from itertools import chain, islice
import numpy as np
import pandas as pd
import types

from collections import defaultdict, deque
from ml.utils.numeric_functions import max_type, num_splits, wsrj, max_dtype
from ml.utils.seq import grouper_chunk
from ml.utils.basic import StructArray, isnamedtupleinstance, Shape
from ml.utils.logger import log_config
from ml.abc.data import AbsDataset
from ml.utils.numeric_functions import nested_shape


log = log_config(__name__)


def assign_struct_array(it, type_elem, start_i, end_i, dtype, dims):
    length = end_i - start_i
    if isinstance(dims, list) and len(dims) == 1 and len(dtype) == dims[0] and dims[0] > 1:
        shape = length
    elif isinstance(dims, list):
        shape = [length] + dims
    else:
        shape = length

    stc_arr = np.empty(shape, dtype=dtype)
    if type_elem == np.ndarray and len(stc_arr.shape) == 1:
        for i, row in enumerate(it):
            stc_arr[i] = tuple(row)
    else:
        for i, row in enumerate(it):
            stc_arr[i] = row
    return stc_arr


class BaseIterator(object):
    def __init__(self, it, length: int = np.inf, dtypes: list = None, shape=None,
                 type_elem=None, pushedback=None) -> None:
        self.data = it
        self.pushedback = [] if pushedback is None else pushedback
        self.dtypes = dtypes
        self.dtype = max_dtype(dtypes)
        self.type_elem = type_elem
        if isinstance(it, StructArray):
            self.shape = self.calc_shape_stc(length, it.shape)
        else:
            if isinstance(shape, Shape):
                self.shape = self.calc_shape_stc(length, shape)
            # elif length != np.inf:
            #    print(shape)
            #    self.shape = (length,)
            else:
                self.shape = None
            #    self.shape = (np.inf,)
        self.iter_init = True
        self._it = None

    @property
    def length(self):
        if self.shape is not None:
            if isinstance(self.shape, Shape):
                return self.shape.max_length
            elif self.shape[0] is not None:
                return self.shape[0]
        return np.inf

    @staticmethod
    def calc_shape_stc(length: int, shape: Shape) -> Shape:
        length_shape = {}
        for group, g_shape in shape.items():
            group_length = shape[group][0]
            group_length = length if group_length > length else shape[group][0]
            length_shape[group] = tuple([group_length] + list(g_shape[1:]))
        return Shape(length_shape)

    @property
    def groups(self) -> list:
        return [c for c, _ in self.dtypes]

    def pushback(self, val) -> None:
        self.pushedback.append(val)

    @staticmethod
    def default_dtypes(dtype) -> list:
        if not isinstance(dtype, list):
            return [("c0", dtype)]
        else:
            return dtype

    def window(self, win_size: int = 2):
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
        if self.length != np.inf:
            length = self.length * sum(self.shape[1:])
        else:
            length = self.length
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
        return self.length

    def unique(self) -> dict:
        values = defaultdict(lambda: 0)
        for batch in self:
            u_values, counter = np.unique(batch, return_counts=True)
            for k, v in dict(zip(u_values, counter)).items():
                values[k] += v
        return values

    def is_multidim(self) -> bool:
        return isinstance(self.shape, dict)

    def _cycle_it(self):
        while True:
            for elem in self:
                yield elem
            self.reset_it()

    def __iter__(self) -> 'BaseIterator':
        return self

    def __next__(self):
        if self._it is None and (self.length == np.inf or self.num_splits() == 0):
            self._it = self.data
        elif self._it is None and self.num_splits() > 0:
            self._it = islice(self.data, self.num_splits() - len(self.pushedback))

        if len(self.pushedback) > 0:
            return self.pushedback.pop()
        else:
            return next(self._it)

    def reset_it(self):
        self._it = None

    def to_iter(self):
        groups = self.groups
        for batch in self:
            row = []
            for group in groups:
                row.append(batch[group])
            yield row


class Iterator(BaseIterator):
    def __init__(self, fn_iter, dtypes: list = None, length: int = np.inf) -> None:
        super(Iterator, self).__init__(fn_iter, dtypes=dtypes, length=length)
        if isinstance(fn_iter, types.GeneratorType):
            self.data = fn_iter
        elif isinstance(fn_iter, Iterator):
            pass
        elif isinstance(fn_iter, dict):
            self.data = StructArray(fn_iter.items())
            dtypes = self.data.dtypes
            length = len(self.data)
        elif isinstance(fn_iter, pd.DataFrame):
            self.data = fn_iter.itertuples(index=False)
            dtypes = list(zip(fn_iter.columns.values, fn_iter.dtypes.values))
            length = fn_iter.shape[0] if length == np.inf else length
        elif isinstance(fn_iter, np.ndarray):
            self.data = iter(fn_iter)
            length = fn_iter.shape[0] if length == np.inf else length
        elif isinstance(fn_iter, StructArray) or isinstance(fn_iter, AbsDataset):
            self.data = fn_iter
            dtypes = fn_iter.dtypes
            length = len(fn_iter) if length == np.inf else length
        else:
            self.data = iter(fn_iter)
            if hasattr(fn_iter, '__len__'):
                length = len(fn_iter)

        if isinstance(fn_iter, Iterator):
            self.data = fn_iter.data
            length = fn_iter.length if length == np.inf else length
            self.shape = self.calc_shape_stc(length, fn_iter.shape)
            self.dtype = fn_iter.dtype
            self.type_elem = fn_iter.type_elem
            self.pushedback = fn_iter.pushedback
            self.dtypes = fn_iter.dtypes
        else:
            # obtain dtypes, shape, dtype, type_elem and length
            self.chunk_taste(length, dtypes)

    def chunk_taste(self, length, dtypes) -> None:
        """Check for the dtype and global dtype in a chunk"""
        try:
            chunk = next(self)
        except StopIteration:
            self.shape = Shape({"c0": (0, )})
            self.dtypes = None
            self.dtype = None
        else:
            self.dtypes = self._define_dtypes(chunk, dtypes)
            self.shape = self._define_shape(chunk, length)
            self.dtype = max_dtype(self.dtypes)
            self.pushback(chunk)
            self.type_elem = type(chunk)

    def _define_shape(self, chunk, length) -> Shape:
        try:
            shape = chunk.shape
        except AttributeError:
            if isnamedtupleinstance(chunk):
                shape = ()
            elif hasattr(chunk, '__iter__') and not isinstance(chunk, str):
                shape = Shape(nested_shape(chunk, self.dtypes))
            else:
                shape = None

        if isinstance(self.shape, Shape):
            return self.shape
        elif not isinstance(shape, Shape):
            shapes = {}
            if shape is None:
                shape = []
            for group, _ in self.dtypes:
                shapes[group] = tuple([length] + list(shape))
            return self.calc_shape_stc(length, Shape(shapes))
        elif isinstance(shape, Shape):
            shapes = {}
            for group, _ in self.dtypes:
                shapes[group] = tuple([length] + list(shape[group]))
            return self.calc_shape_stc(length, Shape(shapes))

    def _define_dtypes(self, chunk, dtypes) -> list:
        if isinstance(dtypes, list):
            return self.replace_str_type_to_obj(dtypes)
        elif isinstance(chunk, pd.DataFrame):
            ndtypes = []
            for c, cdtype in zip(chunk.columns.values, chunk.dtypes.values):
                ndtypes.append((c, cdtype))
            return ndtypes
        elif isinstance(chunk, np.ndarray):
            return self.default_dtypes(chunk.dtype)
        else:  # scalars
            if type(chunk).__module__ == 'builtins':
                if hasattr(chunk, "__iter__"):
                    type_e = max_type(chunk)
                    if type_e == list or type_e == tuple or type_e == str or type_e == np.ndarray:
                        return self.default_dtypes(np.dtype("|O"))
                    elif type_e == float or type_e == int:
                        return self.default_dtypes(np.dtype(type_e))
                else:
                    if dtypes is not None:
                        return dtypes
                    else:
                        return self.default_dtypes(np.dtype(type(chunk)))
            else:
                return self.default_dtypes(chunk.dtype)

    @staticmethod
    def replace_str_type_to_obj(dtype) -> list:
        if hasattr(dtype, '__iter__'):
            dtype_tmp = []
            for c, dtp in dtype:
                if dtp == "str" or dtp == str:
                    dtype_tmp.append((c, np.dtype("O")))
                else:
                    dtype_tmp.append((c, dtp))
        else:
            if dtype == "str" or dtype == str:
                dtype_tmp = "|O"
            else:
                dtype_tmp = dtype
        return dtype_tmp

    def batchs(self, batch_size: int, batch_type: str = "array") -> 'BatchIterator':
        if batch_size > 0:
            if self.type_elem == AbsDataset or self.type_elem == StructArray:
                if batch_type == "df":
                    return BatchDataFrame(self, batch_size=batch_size)
                elif batch_type == "array":
                    return BatchArray(self, batch_size=batch_size)
                else:
                    return BatchStructured(self, batch_size=batch_size)
            else:
                if batch_type == "df":
                    return BatchItDataFrame(self, batch_size=batch_size)
                elif batch_type == "array":
                    return BatchItArray(self, batch_size=batch_size)
                else:
                    return BatchItStructured(self, batch_size=batch_size)
        else:
            return self

    def __getitem__(self, key) -> 'Iterator':
        if isinstance(key, slice):
            if key.stop is not None:
                return Iterator(self, dtypes=self.dtypes, length=key.stop)
        return NotImplemented

    def __setitem__(self, key, value):
        return NotImplemented

    def cycle(self):
        shape = self.shape.change_length(np.inf)
        return BaseIterator(self._cycle_it(), dtypes=self.dtypes, type_elem=StructArray,
                            pushedback=self.pushedback, shape=shape)


class BatchIterator(BaseIterator):
    batch_type = None

    def __init__(self, it: Iterator, batch_size: int = 258, static: bool = False):
        super(BatchIterator, self).__init__(it, dtypes=it.dtypes, length=it.length, type_elem=it.type_elem)
        self.batch_size = batch_size
        self.shape = it.shape
        self.static = static

    def clean_batchs(self) -> Iterator:
        def cleaner():
            for chunk in self:
                for row in chunk:
                    yield row
        return Iterator(cleaner(), dtypes=self.dtypes, length=self.length)

    def batch_shape(self) -> list:
        shape = self.data.shape
        if isinstance(shape, Shape):
            if len(shape) == 1:
                shape = shape.to_tuple()

        if len(shape) == 1:
            if self.length != np.inf and self.length < self.batch_size:
                return [self.length]
            else:
                return [self.batch_size]
        else:
            i_features = shape[1:]
            if self.length != np.inf and self.length < self.batch_size:
                return [self.length] + list(i_features)
            else:
                return [self.batch_size] + list(i_features)

    def cut_batch(self, length: int):
        end = 0
        for batch in self:
            end += batch.shape[0]
            if end > length:
                mod = length % self.batch_size
                if mod > 0:
                    yield batch[:mod]
                break
            yield batch

    def num_splits(self) -> int:
        return num_splits(self.length, self.batch_size)

    def flat(self) -> Iterator:
        if self.length != np.inf:
            length = self.length * sum(self.shape[1:])
        else:
            length = self.length
        return Iterator(self.flatter(), dtypes=self.dtypes, length=length)

    def sample(self, length: int, col: list = None, weight_fn=None) -> Iterator:
        data = self.clean_batchs()
        return Iterator(wsrj(self.weights_gen(data, col, weight_fn), length), dtypes=self.dtypes, length=length)

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
                return self.builder(BaseIterator(self.cut_batch(key.stop), dtypes=self.dtypes,
                                                 length=key.stop, shape=self.shape),
                                    batch_size=self.batch_size)
        return NotImplemented

    def run(self):
        if self.static is True:
            return self.data
        else:
            batch_shape = self.batch_shape()
            return self.batch_from_it(batch_shape)

    def batch_from_it(self, batch_shape):
        for data in self.data:
            yield data

    @classmethod
    def builder(cls, it: BaseIterator, batch_size: int):
        return cls(it, batch_size=batch_size, static=True)

    @classmethod
    def from_batchs(cls, iterable: iter, dtypes: list = None, from_batch_size: int = 0, length: int = None):
        it = Iterator(iterable, dtypes=dtypes, length=length)
        batcher_len = num_splits(length, from_batch_size)
        shape_dict = {}
        for group, shape in it.shape.items():
            shape_dict[group] = [shape[0]] + list(shape[2:])
        shape = Shape(shape_dict)
        return cls.builder(BaseIterator(it[:batcher_len], shape=shape, dtypes=dtypes), batch_size=from_batch_size)

    def cycle(self):
        return BatchIterator.from_batchs(self._cycle_it(),  dtypes=self.dtypes, from_batch_size=self.batch_size,
                                         length=np.inf)


class BatchSlice(BatchIterator):
    def batch_from_it(self, shape=None):
        init = 0
        end = self.batch_size
        while True:
            batch = self.data.data[init:end]
            init = end
            end += self.batch_size
            if len(batch) > 0:
                yield batch
            else:
                break


class BatchItArray(BatchIterator):
    batch_type = 'array'

    def batch_from_it_flat(self, shape):
        for smx in grouper_chunk(self.batch_size, self.data):
            smx_a = np.empty(shape, dtype=self.data.dtypes)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row[0]
            yield smx_a[:i+1]

    def batch_from_it(self, shape):
        for smx in grouper_chunk(self.batch_size, self.data):
            smx_a = np.empty(shape, dtype=self.data.dtype)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row
            yield smx_a[:i+1]

    def run(self):
        if self.static is True:
            return self.data
        else:
            batch_shape = self.batch_shape()
            shape = self.data.shape.to_tuple()
            if len(shape) == 2 and shape[1] == 1:
                return self.batch_from_it_flat(batch_shape)
            else:
                return self.batch_from_it(batch_shape)


class BatchItStructured(BatchIterator):
    batch_type = 'structured'

    def batch_from_it(self, shape):
        for start_i, end_i, stc_array in BatchItDataFrame.str_array(shape, self.batch_size,
                                                                    self.data, self.data.dtypes):
            group_array = []
            for group in self.groups:
                group_array.append((group, stc_array[group]))
            yield StructArray(group_array)


class BatchItDataFrame(BatchIterator):
    batch_type = 'df'

    @staticmethod
    def str_array(shape, batch_size, data, dtypes):
        start_i = 0
        end_i = 0
        if len(shape) > 1:
            dims = shape[1:]
        else:
            dims = 1

        for smx in grouper_chunk(batch_size, data):
            end_i += shape[0]
            if data.length is not None and data.length < end_i:
                end_i = data.length
            array = assign_struct_array(smx, data.type_elem, start_i, end_i, dtypes, dims)
            yield start_i, end_i, array
            start_i = end_i

    def batch_from_it(self, shape):
        if len(shape) > 1 and len(self.groups) < shape[1]:
            self.dtypes = [("c{}".format(i), self.dtype) for i in range(shape[1])]
            columns = self.groups
        else:
            columns = self.groups

        for start_i, end_i, stc_array in BatchItDataFrame.str_array(shape[:1], self.batch_size, self.data, self.dtypes):
            yield pd.DataFrame(stc_array, index=np.arange(start_i, end_i), columns=columns)


class BatchArray(BatchSlice):
    batch_type = "array"

    def run(self):
        for batch in self.batch_from_it():
            yield batch.to_ndarray(dtype=self.dtype)


class BatchDataFrame(BatchSlice):
    batch_type = "df"

    def run(self):
        init = 0
        end = self.batch_size
        for batch in self.batch_from_it():
            yield batch.to_df(init_i=init, end_i=end)
            init = end
            end += self.batch_size


class BatchStructured(BatchSlice):
    batch_type = "structured"

    def run(self):
        return self.batch_from_it()
