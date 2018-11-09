from itertools import chain, islice
import numpy as np
import pandas as pd
import dask.dataframe as dd
import psycopg2
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


def assign_struct_array2df(it, type_elem, start_i, end_i, dtype, columns):
    length = end_i - start_i
    stc_arr = np.empty(length, dtype=dtype)
    if hasattr(type_elem, "__iter__"):
        for i, row in enumerate(it):
            stc_arr[i] = tuple(row)
    else:
        for i, row in enumerate(it):
            stc_arr[i] = row
    smx = pd.DataFrame(stc_arr, index=np.arange(start_i, end_i), columns=columns)
    return smx


class BaseIterator(object):
    def __init__(self, it, length=None, dtypes=None, shape=None, type_elem=None) -> None:
        self.data = it
        self.pushedback = []
        self.dtypes = dtypes
        self.dtype = max_dtype(dtypes)
        self.type_elem = type_elem
        self.shape = self.calc_shape(length, shape)
        self.iter_init = True
        self.dims = len(self.shape)

    def length(self) -> int:
        if self.shape is not None:
            return self.shape[0]

    def calc_shape(self, length, shape) -> tuple:
        if shape is None:
            return (length, )
        elif shape[0] is None and length is not None:
            return tuple([length] + list(shape[1:]))
        elif shape[0] is not None and length is not None:
            return tuple([length] + list(shape[1:]))
        return shape

    @property
    def columns(self):
        if isinstance(self.dtypes, list):
            return [c for c, _ in self.dtypes]
        else:
            if self.dims is not None:
                return ["c"+str(i) for i in range(self.dims)]
            else:
                return ["c0"]

    def pushback(self, val) -> None:
        self.pushedback.append(val)

    @staticmethod
    def default_dtypes(dtype):
        return [("c0", dtype)]

    def buffer(self, buffer_size: int):
        while True:
            buffer = []
            for elem in self:
                if len(buffer) < buffer_size:
                    buffer.append(elem)
                else:
                    self.pushback(elem)
                    break
            try:
                v = next(self)
                if v is not None:
                    self.pushback(v)
            except StopIteration:
                yield buffer
                break
            yield buffer

    def window(self, win_size=2):
        win = deque((next(self.data, None) for _ in range(win_size)), maxlen=win_size)
        yield win
        for e in self.data:
            win.append(e)
            yield win

    def flat(self, df=False):
        def _iter():
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

        it = Iterator(_iter(), dtypes=self.dtypes)
        if self.length is not None:
            it.set_length(self.length*sum(self.shape[1:]))

        if self.has_batchs:
            it = it.batchs(self.batch_size, df=df)
        return it

    def sample(self, k: int, col=None, weight_fn=None):
        if self.has_batchs:
            data = self.clean_chunks()
        else:
            data = self
        it = Iterator(wsrj(self.weights_gen(data, col, weight_fn), k), dtypes=self.dtypes)
        it.set_length(k)
        return it

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

    # def set_length(self, length: int):
    #    self.length = length
    #    if length is not None and self.features_dim is not None:
    #        self.shape = [length] + list(self.features_dim)

    def num_splits(self) -> int:
        return self.length()

    def unique(self) -> dict:
        values = defaultdict(lambda: 0)
        for batch in self:
            u_values, counter = np.unique(batch, return_counts=True)
            for k, v in dict(zip(u_values, counter)).items():
                values[k] += v
        return values

    def __iter__(self):
        if self.length() is None or self.iter_init is False:
            return self
        else:
            self.iter_init = False
            return islice(self, self.num_splits())

    def __next__(self):
        if len(self.pushedback) > 0:
            return self.pushedback.pop()
        try:
            return next(self.data)
        except StopIteration:
            if hasattr(self.data, 'close'):
                self.data.close()
            raise StopIteration
        except psycopg2.InterfaceError:
            raise StopIteration


class Iterator(BaseIterator):
    def __init__(self, fn_iter, dtypes=None, length=None) -> None:
        super(Iterator, self).__init__(fn_iter, dtypes=dtypes, length=length)
        if isinstance(fn_iter, types.GeneratorType):  # or isinstance(fn_iter, psycopg2.extensions.cursor):
            self.data = fn_iter
            self.is_ds = False
        if isinstance(fn_iter, Iterator):
            self.data = fn_iter
            length = fn_iter.length if length is None else length
            self.is_ds = False
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
            self.is_ds = False

        # to obtain dtypes, shape, dtype, type_elem and length
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
        self.dims = len(self.shape)

    @staticmethod
    def replace_str_type_to_obj(dtype):
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

    def batchs(self, batch_size: int, df=True):
        if batch_size > 0:
            if self.is_ds:
                if df is True:
                    return BatchDataFrame(self, batch_size=batch_size, length=self.length())
                else:
                    return BatchArray(self, batch_size=batch_size, length=self.length())
            else:
                return BatchIt(self, batch_size=batch_size, length=self.length(), df=df)
        else:
            return self

    @staticmethod
    def check_datatime(dtype: list):
        cols = []
        for col_i, (_, type_) in enumerate(dtype):
            if type_ == np.dtype('<M8[ns]'):
                cols.append(col_i)
        return cols

    def _concat_aux(self, it):
        from collections import deque
        
        if self.type_elem == np.ndarray:
            concat_fn = np.concatenate
        elif self.type_elem is None:
            concat_fn = None
        else:
            concat_fn = pd.concat

        self_it = deque(self)
        try:
            last_chunk = self_it.pop()
        except IndexError:
            for chunk in it:
                yield chunk
        else:
            for chunk in self_it:
                yield chunk

            for chunk in it:
                r = abs(last_chunk.shape[0] - self.batch_size)
                yield concat_fn((last_chunk, chunk[:r]))
                last_chunk = chunk[r:]

            if last_chunk.shape[0] > 0:
                yield last_chunk

    def concat(self, it):
        if not self.has_batchs and not it.has_batchs:
            it_c = Iterator(chain(self, it), batch_size=self.batch_size)
            if self.length is not None and it.length is not None:
                it_c.set_length(self.length+it.length)
            return it_c
        elif self.batch_size == it.batch_size:
            it_c = Iterator(self._concat_aux(it), batch_size=self.batch_size)
            if self.length is not None and it.length is not None:
                it_c.set_length(self.length+it.length)
            return it_c
        else:
            raise Exception("I can't concatenate two iterables with differents chunks size")

    def __getitem__(self, key) -> BaseIterator:
        if isinstance(key, slice):
            if key.stop is not None:
                return Iterator(self.data, dtypes=self.dtypes, length=key.stop)
        return NotImplemented

    def __setitem__(self, key, value):
        return NotImplemented


class BatchIterator(BaseIterator):
    def __init__(self, it: Iterator, batch_size: int=258, length=None, df: bool=False):
        super(BatchIterator, self).__init__(it, dtypes=it.dtypes, length=it.length)
        self.batch_size = batch_size
        self.shape = it.shape
        self.df = df
        self.type_elem = pd.DataFrame if df is True else np.ndarray

    def clean_chunks(self) -> BaseIterator:
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

    def cut_batch(self, length):
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
            return (length, )
        elif shape[0] is None and length is not None:
            return tuple([length] + list(shape[1:]))
        elif shape[0] is not None and length is not None:
            return tuple([length] + list(shape[1:]))
        return shape

    def run(self) -> BaseIterator:
        return self.data

    def __next__(self):
        return next(self.run())

    def __iter__(self) -> BaseIterator:
        return self.run()

    def __getitem__(self, key) -> BaseIterator:
        if isinstance(key, slice):
            if key.stop is not None:
                return BatchIterator(BaseIterator(self.cut_batch(key.stop), dtypes=self.dtypes,
                                                  length=key.stop, shape=self.shape),
                                     df=self.df, batch_size=self.batch_size)
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
            smx_a = np.empty(shape, dtype=self.data.dtypes)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row
            yield smx_a[:i+1]

    def batch_from_it_df(self, shape):
        start_i = 0
        end_i = 0
        columns = self.data.columns
        for smx in grouper_chunk(self.batch_size, self.data):
            end_i += shape[0]
            yield assign_struct_array2df(smx, self.data.type_elem, start_i, end_i, self.data.dtypes,
                                         columns)
            start_i = end_i

    def run(self):
        batch_shape = self.batch_shape()
        if self.df is True:
            return self.batch_from_it_df(batch_shape)
        else:
            if len(self.data.shape) == 2 and self.data.shape[1] == 1:
                return self.batch_from_it_flat(batch_shape)
            else:
                return self.batch_from_it_array(batch_shape)


class BatchArray(BatchIterator):
    def run(self):
        init = 0
        end = self.batch_size
        length = self.batch_size
        while length > 0:
            batch = self.data[init:end].to_ndarray(dtype=self.dtype)
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
            batch = self.data[init:end].to_df(init_i=init, end_i=end)
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
