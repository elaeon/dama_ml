from itertools import chain, islice
import numpy as np
import pandas as pd
import dask.dataframe as dd
import types
import logging

from collections import defaultdict, deque
from ml.utils.config import get_settings
from ml.utils.numeric_functions import max_type, num_splits, wsrj, max_dtype
from ml.data.abc import AbsDataset
from ml.utils.seq import grouper_chunk


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


def assign_struct_array(it, type_elem, start_i, end_i, dtype, dims):
    length = end_i - start_i
    if dims == 1 or len(dims) == 1 and dims[0] == 1:
        shape = length
    elif len(dims) == 1 and len(dtype) == dims[0]:
        shape = length
    else:
        shape = [length] + dims

    stc_arr = np.empty(shape, dtype=dtype)
    if type_elem == np.ndarray and len(stc_arr.shape) == 1:
        for i, row in enumerate(it):
            stc_arr[i] = tuple(row)
    else:
        for i, row in enumerate(it):
            stc_arr[i] = row
    return stc_arr


class BaseIterator(object):
    def __init__(self, it, length=None, dtypes=None, shape=None, type_elem=None, pushedback=None) -> None:
        self.data = it
        self.pushedback = [] if pushedback is None else pushedback
        self.dtypes = dtypes
        self.dtype = max_dtype(dtypes)
        self.type_elem = type_elem
        self.shape = self.calc_shape(length, shape)
        self.iter_init = True
        self._it = None

    def length(self) -> int:
        if self.shape is not None:
            return self.shape[0]

    def calc_shape(self, length, shape) -> tuple:
        if shape is None:
            return tuple([length])
        elif shape[0] is None and length is not None:
            return tuple([length] + list(shape[1:]))
        elif shape[0] is not None and length is not None:
            return tuple([length] + list(shape[1:]))
        return shape

    @property
    def labels(self) -> list:
        return [c for c, _ in self.dtypes]

    def pushback(self, val) -> None:
        self.pushedback.append(val)

    @staticmethod
    def default_dtypes(dtype) -> list:
        if not isinstance(dtype, list):
            return [("c0", dtype)]
        else:
            return dtype

    def window(self, win_size: int=2):
        win = deque((next(self.data, None) for _ in range(win_size)), maxlen=win_size)
        yield win
        for e in self.data:
            win.append(e)
            yield win

    def flatter(self):
        if self.type_elem == np.ndarray:
            for chunk in self:
                for e in chunk.reshape(-1):
                    if hasattr(e, "__iter__") and len(e) == 1:
                        yield e[0]
                    else:
                        yield e
        elif self.type_elem == pd.DataFrame:
            for chunk in self:
                for e in chunk.values.reshape(-1):
                    yield e
        elif self.type_elem.__module__ == 'builtins':
            for e in chain.from_iterable(self):
                yield e
        else:
            raise Exception("Type of elem {} does not supported".format(self.type_elem))

    def flat(self) -> 'Iterator':
        if self.length() is not None:
            length = self.length() * sum(self.shape[1:])
        else:
            length = None
        return Iterator(self.flatter(), dtypes=self.dtypes, length=length)

    def sample(self, length: int, col=None, weight_fn=None) -> 'Iterator':
        return Iterator(wsrj(self.weights_gen(self, col, weight_fn), length), dtypes=self.dtypes, length=length)

    def weights_gen(self, data, col, weight_fn):
        if not hasattr(self.type_elem, '__iter__') and weight_fn is not None:
            for elem in data:
                yield elem, weight_fn(elem)
        elif col is not None and weight_fn is not None:
            for row in data:
                yield row, weight_fn(row[col])
        else:
            for row in data:
                yield row, 1

    def num_splits(self) -> int:
        return self.length()

    def unique(self) -> dict:
        values = defaultdict(lambda: 0)
        for batch in self:
            u_values, counter = np.unique(batch, return_counts=True)
            for k, v in dict(zip(u_values, counter)).items():
                values[k] += v
        return values

    def __iter__(self) -> 'BaseIterator':
        return self

    def __next__(self):
        if self._it is None and self.length() is None:
            self._it = self.data
        elif self._it is None:
            self._it = islice(self.data, self.num_splits() - len(self.pushedback))

        if len(self.pushedback) > 0:
            return self.pushedback.pop()
        else:
            return next(self._it)


class Iterator(BaseIterator):
    def __init__(self, fn_iter, dtypes=None, length=None) -> None:
        super(Iterator, self).__init__(fn_iter, dtypes=dtypes, length=length)
        if isinstance(fn_iter, types.GeneratorType):
            self.data = fn_iter
            self.is_ds = False
        elif isinstance(fn_iter, Iterator):
            pass
        elif isinstance(fn_iter, pd.DataFrame):
            self.data = fn_iter.itertuples(index=False)
            dtypes = list(zip(fn_iter.columns.values, fn_iter.dtypes.values))
            length = fn_iter.shape[0] if length is None else length
            self.is_ds = False
        elif isinstance(fn_iter, np.ndarray):
            self.data = iter(fn_iter)
            length = fn_iter.shape[0] if length is None else length
            self.is_ds = False
        elif isinstance(fn_iter, AbsDataset):
            self.data = fn_iter
            length = fn_iter.shape[0] if length is None else length
            self.is_ds = True
        else:
            self.data = iter(fn_iter)
            if hasattr(fn_iter, '__len__'):
                length = len(fn_iter)
            self.is_ds = False

        if isinstance(fn_iter, Iterator):
            self.data = fn_iter.data
            length = fn_iter.length if length is None else length
            self.is_ds = False
            self.shape = self.calc_shape(length, fn_iter.shape)
            self.dtype = fn_iter.dtype
            self.type_elem = fn_iter.type_elem
            self.pushedback = fn_iter.pushedback
            self.dtypes = fn_iter.dtypes
        else:
            # obtain dtypes, shape, dtype, type_elem and length
            self.chunk_taste(length, dtypes)

    def chunk_type_elem(self):
        try:
            chunk = next(self)
        except StopIteration:
            return
        else:
            self.pushback(chunk)
            return type(chunk)

    def chunk_taste(self, length, dtypes) -> None:
        """Check for the dtype and global dtype in a chunk"""
        try:
            chunk = next(self)
        except StopIteration:
            self.shape = (0,)
            return

        if isinstance(dtypes, list):
            self.dtypes = self.replace_str_type_to_obj(dtypes)
        elif isinstance(chunk, pd.DataFrame):
            self.dtypes = []
            for c, cdtype in zip(chunk.columns.values, chunk.dtypes.values):
                self.dtypes.append((c, cdtype))
        elif isinstance(chunk, np.ndarray):
            self.dtypes = self.default_dtypes(chunk.dtype)
        else:  # scalars
            if type(chunk).__module__ == 'builtins':
                if hasattr(chunk, "__iter__"):
                    type_e = max_type(chunk)
                    if type_e == list or type_e == tuple or type_e == str or type_e == np.ndarray:
                        self.dtypes = self.default_dtypes(np.dtype("|O"))
                else:
                    if dtypes is not None:
                        self.dtypes = dtypes
                    else:
                        self.dtypes = self.default_dtypes(np.dtype(type(chunk)))

                    if type(chunk) == str:
                        self.dtypes = self.default_dtypes(np.dtype("|O"))
            else:
                self.dtypes = self.default_dtypes(chunk.dtype)

        try:
            shape = [None] + list(chunk.shape)
        except AttributeError:
            if hasattr(chunk, '__iter__') and not isinstance(chunk, str):
                shape = (None, len(chunk))
            else:
                shape = (None,)

        self.shape = self.calc_shape(length, shape)
        self.dtype = max_dtype(self.dtypes)
        self.pushback(chunk)
        self.type_elem = type(chunk)

    @staticmethod
    def replace_str_type_to_obj(dtype) -> list:
        if hasattr(dtype, '__iter__'):
            dtype_tmp = []
            for c, dtp in dtype:
                if dtp == "str" or dtp == str:
                    dtype_tmp.append((c, "|O"))
                else:
                    dtype_tmp.append((c, dtp))
        else:
            if dtype == "str" or dtype == str:
                dtype_tmp = "|O"
            else:
                dtype_tmp = dtype
        return dtype_tmp

    def batchs(self, batch_size: int, batch_type: str="array") -> BaseIterator:
        if batch_size > 0:
            if self.is_ds:
                if batch_type == "df":
                    return BatchDataFrame(self, batch_size=batch_size)
                elif batch_type == "array":
                    return BatchArray(self, batch_size=batch_size)
                else:
                    return BatchStructured(self, batch_size=batch_size)
            else:
                return BatchIt(self, batch_size=batch_size, batch_type=batch_type)
        else:
            return self

    def __getitem__(self, key) -> 'Iterator':
        if isinstance(key, slice):
            if key.stop is not None:
                return Iterator(self, dtypes=self.dtypes, length=key.stop)
        return NotImplemented

    def __setitem__(self, key, value):
        return NotImplemented


class BatchIterator(BaseIterator):
    def __init__(self, it: Iterator, batch_size: int=258, batch_type: str='array'):
        super(BatchIterator, self).__init__(it, dtypes=it.dtypes, length=it.length)
        self.batch_size = batch_size
        self.shape = it.shape
        self.type_elem = pd.DataFrame if batch_type is "df" else np.ndarray
        self.batch_type = batch_type

    def clean_batchs(self) -> Iterator:
        def cleaner():
            for chunk in self:
                for row in chunk:
                    yield row
        return Iterator(cleaner(), dtypes=self.dtypes, length=self.length())

    def batch_shape(self) -> list:
        shape = self.data.shape
        if len(shape) == 1:
            return [self.batch_size]
        else:
            i_features = shape[1:]
            return [self.batch_size] + list(i_features)

    def cut_batch(self, length: int):
        end = 0
        for batch in self:
            end += batch.shape[0]
            if end > length:
                mod = length % self.batch_size
                if mod > 0:
                    batch = batch[:mod]
                yield batch
                break
            yield batch

    def num_splits(self) -> int:
        return num_splits(self.length(), self.batch_size)

    def calc_shape(self, length, shape) -> tuple:
        if shape is None:
            return tuple([length])
        elif shape[0] is None and length is not None:
            return tuple([length] + list(shape[1:]))
        elif shape[0] is not None and length is not None:
            return tuple([length] + list(shape[1:]))
        return shape

    def flat(self) -> Iterator:
        if self.length() is not None:
            length = self.length() * sum(self.shape[1:])
        else:
            length = None
        return Iterator(self.flatter(), dtypes=self.dtypes,
                        length=length).batchs(batch_size=self.batch_size, batch_type=self.batch_type)

    def sample(self, length: int, col: list=None, weight_fn=None) -> BaseIterator:
        data = self.clean_batchs()
        return Iterator(wsrj(self.weights_gen(data, col, weight_fn), length), dtypes=self.dtypes, length=length)

    def run(self) -> Iterator:
        return self.data

    def __next__(self):
        if self._it is None:
            self._it = self.run()
        if len(self.pushedback) > 0:
            return self.pushedback.pop()
        else:
            return next(self._it)

    def __iter__(self) -> 'BatchIterator':
        return self

    def __getitem__(self, key) -> 'BatchIterator':
        if isinstance(key, slice):
            if key.stop is not None:
                return BatchIterator(BaseIterator(self.cut_batch(key.stop), dtypes=self.dtypes,
                                                  length=key.stop, shape=self.shape),
                                     batch_type=self.batch_type, batch_size=self.batch_size)
        return NotImplemented


class BatchIt(BatchIterator):
    def batch_from_it_flat(self, shape):
        for smx in grouper_chunk(self.batch_size, self.data):
            smx_a = np.empty(shape, dtype=self.data.dtypes)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row[0]
            yield smx_a[:i+1]

    def batch_from_it_array(self, shape):
        for smx in grouper_chunk(self.batch_size, self.data):
            smx_a = np.empty(shape, dtype=self.data.dtype)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row
            yield smx_a[:i+1]

    def batch_from_it_structured(self, shape):
        start_i = 0
        end_i = 0
        if len(shape) > 1:
            dims = shape[1:]
        else:
            dims = 1
        if self.length() is None:
            for smx in grouper_chunk(self.batch_size, self.data):
                end_i += shape[0]
                yield assign_struct_array(smx, self.data.type_elem, start_i, end_i, self.data.dtypes, dims)
                start_i = end_i
        else:
            for smx in grouper_chunk(self.batch_size, self.data):
                end_i += shape[0]
                if end_i > self.length():
                    end_i = self.length()
                yield assign_struct_array(smx, self.data.type_elem, start_i, end_i, self.data.dtypes, dims)
                start_i = end_i

    def batch_from_it_df(self, shape):
        start_i = 0
        end_i = 0
        columns = self.data.labels
        for stc_array in self.batch_from_it_structured(shape):
            end_i += stc_array.shape[0]
            yield pd.DataFrame(stc_array, index=np.arange(start_i, end_i), columns=columns)
            start_i = end_i

    def run(self):
        batch_shape = self.batch_shape()
        if self.batch_type == "df":
            return self.batch_from_it_df(batch_shape)
        elif self.batch_type == "array":
            if len(self.data.shape) == 2 and self.data.shape[1] == 1:
                return self.batch_from_it_flat(batch_shape)
            else:
                return self.batch_from_it_array(batch_shape)
        else:
            return self.batch_from_it_structured(batch_shape)


class BatchArray(BatchIterator):
    def run(self):
        init = 0
        end = self.batch_size
        length = self.batch_size
        while length > 0:
            batch = self.data.data[init:end].to_ndarray(dtype=self.dtype)
            yield batch
            init = end
            end += self.batch_size
            length = batch.shape[0]


class BatchDataFrame(BatchIterator):
    def run(self):
        init = 0
        end = self.batch_size
        length = self.batch_size
        while length > 0:
            batch = self.data.data[init:end].to_df(init_i=init, end_i=end)
            yield batch
            init = end
            end += self.batch_size
            length = batch.shape[0]


class BatchStructured(BatchIterator):
    def run(self):
        init = 0
        end = self.batch_size
        length = self.batch_size
        while length > 0:
            batch = self.data.data[init:end].to_structured()
            yield batch
            init = end
            end += self.batch_size
            length = batch.shape[0]


class DaskIterator(object):
    def __init__(self, fn_iter) -> None:
        if isinstance(fn_iter, dd.DataFrame):
            self.data = fn_iter
            self.length = None
            self.dtype = None

    def to_memory(self):
        df = self.data.compute()
        self.length = df.shape[0]
        self.dtype = list(zip(df.columns.values, df.dtypes.values))
        return df
