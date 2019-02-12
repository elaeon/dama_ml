from itertools import chain, islice
import numpy as np
import pandas as pd
import types

from collections import defaultdict, deque
from ml.utils.numeric_functions import max_type, num_splits, wsrj, max_dtype
from ml.utils.seq import grouper_chunk
from ml.utils.core import Shape
from ml.utils.miscellaneous import isnamedtupleinstance
from ml.utils.logger import log_config
from ml.abc.data import AbsData
from ml.abc.group import AbsGroup
from ml.utils.numeric_functions import nested_shape
from ml.data.groups.core import DaGroup, StcArrayGroup, TupleGroup
from ml.fmtypes import Slice, DEFAUL_GROUP_NAME
from numbers import Number, Integral

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
    elif len(dtype) == 1:
        group = dtype.names[0]
        for i, row in enumerate(it):
            if isnamedtupleinstance(row):
                stc_arr[i] = row
            else:
                stc_arr[group][i] = row
    elif type_elem == str or type_elem == np.str_ or type_elem == int or type_elem == np.string_\
            or type_elem == np.object or type_elem == object or type_elem == float:
        for (group, (_, _)), row in zip(dtype.fields.items(), it):
            stc_arr[group] = row
    else:
        for i, row in enumerate(it):
            stc_arr[i] = row
    return stc_arr


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


class BaseIterator(object):
    def __init__(self, it, length: int = np.inf, dtypes: np.dtype = None, shape=None,
                 type_elem=None, pushedback=None) -> None:
        self.data = it
        self.pushedback = [] if pushedback is None else pushedback
        self.dtypes = dtypes
        self.dtype = max_dtype(dtypes)
        self.type_elem = type_elem
        if isinstance(shape, Shape):
            self.shape = self.calc_shape_stc(length, shape)
        else:
            self.shape = None
        self.rewind = False

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
    def groups(self) -> tuple:
        return self.dtypes.names

    def pushback(self, val) -> None:
        self.pushedback.append(val)

    @staticmethod
    def default_dtypes(dtype: np.dtype) -> np.dtype:
        if dtype.fields is None:
            return np.dtype([(DEFAUL_GROUP_NAME, dtype)])
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
            for array in self:
                for elem in array.reshape(-1):
                    if hasattr(elem, "__iter__") and len(elem) == 1:
                        yield elem[0]
                    else:
                        yield elem
        elif self.type_elem == Slice:
            for data in self:
                for elem in data.batch.to_ndarray().reshape(-1):
                    yield elem
        elif self.type_elem == pd.DataFrame:
            for data in self:
                for elem in data.values.reshape(-1):
                    yield elem
        elif self.type_elem.__module__ == 'builtins':
            for elem in chain.from_iterable(self):
                yield elem
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
        elif col is not None:
            for row in data:
                yield row[col], 1
        else:
            for row in data:
                yield row, 1

    def num_splits(self) -> int:
        return self.length

    def is_multidim(self) -> bool:
        return isinstance(self.shape, dict)

    def _cycle_it(self):
        if self.rewind is False:
            raise Exception("This object is not cyclical, try to saved it in an AbsData")

        while True:
            for elem in self:
                yield elem

    def __iter__(self) -> 'BaseIterator':
        if isinstance(self.data, AbsData):
            self.data.data.counter = 0
            if len(self.pushedback) > 0:
                self.pushedback.pop()
        return self

    def __next__(self):
        if len(self.pushedback) > 0:
            return self.pushedback.pop()
        else:
            return next(self.data)

    def to_iter(self):
        groups = self.groups
        for batch in self:
            row = []
            for group in groups:
                row.append(batch[group])
            yield row

    def __getitem__(self, key) -> 'Iterator':
        if isinstance(key, slice):
            if key.stop is not None:
                if self.rewind is False:
                    _it = islice(self.data, key.stop)
                else:
                    _it = self
                return Iterator(_it, dtypes=self.dtypes, length=key.stop)
        return NotImplemented


class Iterator(BaseIterator):
    def __init__(self, fn_iter, dtypes: np.dtype = None, length: int = np.inf) -> None:
        super(Iterator, self).__init__(fn_iter, dtypes=dtypes, length=length)
        if isinstance(fn_iter, types.GeneratorType):
            self.data = fn_iter
            self.rewind = False
        elif isinstance(fn_iter, Iterator):
            self.data = fn_iter.data
            length = fn_iter.length if length == np.inf else length
            self.shape = self.calc_shape_stc(length, fn_iter.shape)
            self.dtype = fn_iter.dtype
            self.type_elem = fn_iter.type_elem
            self.pushedback = fn_iter.pushedback
            self.dtypes = fn_iter.dtypes
            self.rewind = fn_iter.rewind
            return
        elif isinstance(fn_iter, pd.DataFrame):
            self.data = fn_iter.itertuples(index=False)
            dtypes = np.dtype(list(zip(fn_iter.columns.values, fn_iter.dtypes.values)))
            length = fn_iter.shape[0] if length == np.inf else length
            self.rewind = False
        elif isinstance(fn_iter, np.ndarray):
            self.data = iter(fn_iter)
            length = fn_iter.shape[0] if length == np.inf else length
            self.rewind = False
        elif isinstance(fn_iter, AbsData) or isinstance(fn_iter, AbsGroup) or type(fn_iter) == DaGroup:
            self.data = fn_iter
            self.dtypes = fn_iter.dtypes
            self.dtype = fn_iter.dtype
            length = len(fn_iter) if length == np.inf else length
            self.rewind = True
            self.shape = self.calc_shape_stc(length, fn_iter.shape)
            return
        else:
            self.data = iter(fn_iter)
            self.rewind = False
            if hasattr(fn_iter, '__len__'):
                length = len(fn_iter)

        # obtain dtypes, shape, dtype, type_elem and length
        self.chunk_taste(length, dtypes)

    def chunk_taste(self, length, dtypes) -> None:
        """Check for the dtype and global dtype in a chunk"""
        try:
            elem = next(self)
        except StopIteration:
            self.shape = Shape({DEFAUL_GROUP_NAME: (0, )})
            self.dtypes = None
            self.dtype = None
        else:
            self.dtypes = self._define_dtypes(elem, dtypes)
            self.shape = self._define_shape(elem, length)
            self.dtype = max_dtype(self.dtypes)
            self.pushback(elem)
            self.type_elem = type(elem)

    def _define_shape(self, elem, length) -> Shape:
        try:
            shape = elem.shape
        except AttributeError:
            if isnamedtupleinstance(elem):
                shape = ()
            elif hasattr(elem, '__iter__') and not isinstance(elem, str):
                shape = Shape(nested_shape(elem, self.dtypes))
            else:
                shape = None

        if not isinstance(shape, Shape):
            shapes = {}
            if shape is None:
                shape = []

            if len(self.dtypes) > 1:
                for group in self.groups:
                    shapes[group] = tuple([length] + list(shape[1:]))
            else:
                for group in self.groups:
                    shapes[group] = tuple([length] + list(shape))
            return self.calc_shape_stc(length, Shape(shapes))
        elif isinstance(shape, Shape):
            shapes = {}
            for group in self.groups:
                shapes[group] = tuple([length] + list(shape[group]))
            return self.calc_shape_stc(length, Shape(shapes))

    def _define_dtypes(self, chunk, dtypes) -> np.dtype:
        if isinstance(dtypes, np.dtype):
            return self.replace_str_type_to_obj(dtypes)
        elif isinstance(chunk, pd.DataFrame):
            ndtypes = []
            for c, cdtype in zip(chunk.columns.values, chunk.dtypes.values):
                ndtypes.append((c, cdtype))
            return np.dtype(ndtypes)
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
                if isnamedtupleinstance(chunk):
                    dtypes_l = []
                    for v, field in zip(chunk, chunk._fields):
                        dtypes_l.append((field, np.dtype(type(v))))
                    return np.dtype(dtypes_l)
                else:
                    return self.default_dtypes(chunk.dtype)

    @staticmethod
    def replace_str_type_to_obj(dtype: np.dtype) -> np.dtype:
        if dtype.fields is not None:
            dtype_tmp = []
            for c, (dtp, _) in dtype.fields.items():
                if dtp == "str" or dtp == str:
                    dtype_tmp.append((c, np.dtype("O")))
                else:
                    dtype_tmp.append((c, dtp))
        else:
            if dtype == "str" or dtype == str:
                dtype_tmp = "|O"
            else:
                dtype_tmp = dtype
        return np.dtype(dtype_tmp)

    def batchs(self, batch_size: int) -> 'BatchIterator':
        if batch_size > 0:
            if isinstance(self.data, AbsData) or isinstance(self.data, AbsGroup):
                return BatchGroup(self, batch_size=batch_size)
            else:
                return BatchItGroup(self, batch_size=batch_size)
        else:
            return self

    def __getitem__(self, key) -> 'Iterator':
        if isinstance(key, slice):
            if key.stop is not None:
                if self.rewind is False:
                    _it = islice(self, key.stop)
                else:
                    _it = self
                return Iterator(_it, dtypes=self.dtypes, length=key.stop)
            else:
                return self
        return NotImplemented

    def __setitem__(self, key, value):
        return NotImplemented

    def cycle(self):
        shape = self.shape.change_length(np.inf)
        return BaseIterator(self._cycle_it(), dtypes=self.dtypes, type_elem=self.type_elem, shape=shape)

    def to_slice(self, batch_size):
        start = 0
        end = batch_size
        for elem in self:
            batch = DaGroup(TupleGroup(elem, dtypes=self.dtypes))  # fixme
            yield Slice(batch=batch, slice=slice(start, end))
            start = end
            end += batch_size


class BatchIterator(BaseIterator):
    type_elem = None

    def __init__(self, it: Iterator, batch_size: int = 258, static: bool = False):
        super(BatchIterator, self).__init__(it, dtypes=it.dtypes, length=it.length, type_elem=self.type_elem)
        self.batch_size = batch_size
        self.shape = it.shape
        self.static = static
        self._it = self.run()

    def clean_batchs(self) -> Iterator:
        def cleaner():
            for chunk in self:
                for row in chunk.batch:
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
        for slice_obj in self:
            end += slice_obj.batch.shape[0]
            if end >= length:
                mod = self.batch_size - (end - length)
                if mod > 0:
                    yield Slice(batch=slice_obj.batch[:mod], slice=slice(slice_obj.slice.start, length))
                break
            else:
                yield slice_obj

    def unique(self) -> dict:
        values = defaultdict(lambda: 0)
        group = self.groups[0]
        for batch in self:
            u_values, counter = np.unique(batch.batch[group].to_ndarray(), return_counts=True)
            for k, v in dict(zip(u_values, counter)).items():
                values[k] += v
        return values

    def num_splits(self) -> int:
        return num_splits(self.length, self.batch_size)

    def flat(self) -> Iterator:
        if self.length != np.inf:
            length = self.length * sum(self.shape[1:])
        else:
            length = self.length
        return Iterator(self.flatter(), dtypes=self.dtypes, length=length)

    def sample(self, length: int, col: str = None, weight_fn=None) -> Iterator:
        data = self.clean_batchs()
        return Iterator(wsrj(self.weights_gen(data, col, weight_fn), length), dtypes=self.dtypes, length=length)

    def __next__(self):
        return next(self._it)

    def __iter__(self) -> 'BatchIterator':
        self._it = self.run()
        return self

    def __len__(self):
        return self.num_splits()

    def __getitem__(self, key) -> 'BatchIterator':
        if isinstance(key, slice):
            if key.stop is not None:
                return self.builder(BaseIterator(self.cut_batch(key.stop), dtypes=self.dtypes,
                                                 length=key.stop, shape=self.shape),
                                    batch_size=self.batch_size)
            else:
                return self
        return NotImplemented

    def run(self):
        if self.static is True:
            return self.data
        else:
            batch_shape = self.batch_shape()
            return self.batch_from_it(batch_shape)

    def batch_from_it(self, batch_shape):
        return NotImplemented

    def _cycle_it(self):
        while True:
            for elem in self:
                yield elem

    def to_iter(self, raw: bool = False):
        if raw is True:
            for slice_obj in self:
                yield slice_obj.batch.to_ndarray()
        else:
            for slice_obj in self:
                yield slice_obj.batch

    @classmethod
    def builder(cls, it: BaseIterator, batch_size: int):
        return cls(it, batch_size=batch_size, static=True)

    @classmethod
    def from_batchs(cls, iterable: iter, dtypes: np.dtype = None, from_batch_size: int = 0,
                    length: int = None, to_slice=False):
        it = Iterator(iterable, dtypes=dtypes, length=length)
        batcher_len = num_splits(length, from_batch_size)
        shape_dict = {}
        for group, shape in it.shape.items():
            shape_dict[group] = [shape[0]] + list(shape[2:])
        shape = Shape(shape_dict)
        if batcher_len == 0:
            batcher_len = None

        if it.type_elem != Slice and to_slice is True:
            iterator = it[:batcher_len].to_slice(from_batch_size)
        else:
            iterator = it[:batcher_len]
        return cls.builder(BaseIterator(iterator, shape=shape, dtypes=dtypes, type_elem=it.type_elem),
                           batch_size=from_batch_size)

    def cycle(self):
        return BatchIterator.from_batchs(self._cycle_it(),  dtypes=self.dtypes, from_batch_size=self.batch_size,
                                         length=np.inf)

    def only_data(self):
        return BatchIterator.from_batchs(self.to_iter(raw=True),  dtypes=self.dtypes, from_batch_size=self.batch_size,
                                         length=self.length)


class BatchGroup(BatchIterator):
    type_elem = Slice

    def batch_from_it(self, shape=None):
        init = 0
        end = self.batch_size
        while True:
            batch = self.data.data[init:end]  # Always return a Dagroup
            if len(batch) > 0:
                yield Slice(batch=batch, slice=slice(init, end))
                init = end
                end += self.batch_size
            else:
                break


class BatchItGroup(BatchIterator):
    type_elem = Slice

    def batch_from_it(self, shape):
        for start_i, end_i, stc_array in str_array(shape, self.batch_size, self.data, self.data.dtypes):
            da_group = DaGroup(StcArrayGroup(stc_array))
            yield Slice(batch=da_group, slice=slice(start_i, end_i))
