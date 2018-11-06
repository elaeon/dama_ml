from itertools import chain, islice
import operator
import types
import numpy as np
import pandas as pd
import dask.dataframe as dd
import psycopg2
import logging

from collections import defaultdict, deque
from ml.utils.config import get_settings
from ml.utils.numeric_functions import max_type, num_splits, wsrj
from ml.utils.batcher import BatchWrapper, cut, assign_struct_array2df
from ml.data.abc import AbsDataset


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


def choice(op):
    def inner(_):
        def view(x, y):
            if isinstance(x, Iterator) and isinstance(y, Iterator):
                return x.stream_operation(op, y)
            elif hasattr(y, "__iter__"):
                return x.stream_operation(op, Iterator(y))
            else:
                return x.scalar_operation(op, y)
        return view
    return inner


class Iterator(object):
    def __init__(self, fn_iter, dtype=None, batch_size=0) -> None:
        if isinstance(fn_iter, types.GeneratorType) or isinstance(fn_iter, psycopg2.extensions.cursor):
            self.data = fn_iter
            self.length = None
            self.is_ds = False
        elif isinstance(fn_iter, Iterator):
            self.data = fn_iter
            self.length = fn_iter.length
            self.is_ds = False
        elif isinstance(fn_iter, pd.DataFrame):
            self.data = fn_iter.itertuples(index=False)
            dtype = list(zip(fn_iter.columns.values, fn_iter.dtypes.values))
            self.length = fn_iter.shape[0]
            self.is_ds = False
        elif isinstance(fn_iter, np.ndarray):
            self.data = iter(fn_iter)
            self.length = fn_iter.shape[0]
            self.is_ds = False
        elif isinstance(fn_iter, AbsDataset):
            self.data = fn_iter
            self.length = fn_iter.shape[0]
            self.is_ds = True
        else:
            self.data = iter(fn_iter)
            self.length = None
            self.is_ds = False

        self.pushedback = []
        self.dtype = None
        self.shape = None
        self.global_dtype = None
        self.type_elem = None
        if batch_size is None:
            batch_size = 0
        self.batch_size = batch_size
        self.has_batchs = True if batch_size > 0 else False
        self.features_dim = None
        self.iter_init = True
        # to obtain dtype, shape, global_dtype and type_elem
        self.chunk_taste(dtype)
    
    @property
    def columns(self):
        if isinstance(self.dtype, list):
            return [c for c, _ in self.dtype]
        else:
            if self.features_dim is not None and len(self.features_dim) > 0:
                return ["c"+str(i) for i in range(self.features_dim[-1])]
            else:
                return ["c0"]

    @property
    def dtypes(self):
        if isinstance(self.dtype, list):
            return self.dtype
        else:
            return [(col, self.dtype) for col in self.columns]

    def pushback(self, val) -> None:
        self.pushedback.append(val)

    def chunk_type_elem(self):
        try:
            chunk = next(self)
        except StopIteration:
            return
        else:
            self.pushback(chunk)
            return type(chunk)

    def chunk_taste(self, dtypes) -> None:
        """Check for the dtype and global dtype in a chunk"""
        try:
            chunk = next(self)
        except StopIteration:
            self.global_dtype = None
            self.dtype = None
            self.shape = (0,)            
            self.type_elem = None
            return

        if isinstance(dtypes, list):
            self.dtype = self.replace_str_type_to_obj(dtypes)
            global_dtype = self._get_global_dtype(self.dtype)
        elif isinstance(chunk, pd.DataFrame):
            self.dtype = []
            for c, cdtype in zip(chunk.columns.values, chunk.dtypes.values):
                self.dtype.append((c, cdtype))
            global_dtype = self._get_global_dtype(self.dtype)
        elif isinstance(chunk, np.ndarray):
            self.dtype = chunk.dtype
            global_dtype = self.dtype
        else:  # scalars
            if type(chunk).__module__ == 'builtins':
                if hasattr(chunk, "__iter__"):
                    type_e = max_type(chunk)
                    if type_e == list or type_e == tuple or type_e == str or type_e == np.ndarray:
                        self.dtype = np.dtype("|O")
                    else:
                        self.dtype = type_e
                else:
                    if dtypes is not None:
                        self.dtype = dtypes
                    else:
                        self.dtype = type(chunk)

                    if self.dtype == str:
                        self.dtype = np.dtype("|O")
            else:
                self.dtype = chunk.dtype
            global_dtype = np.dtype(self.dtype)

        try:
            shape = [self.length] + list(chunk.shape)
        except AttributeError:
            if hasattr(chunk, '__iter__') and not isinstance(chunk, str):
                shape = (self.length, len(chunk))
            else:
                shape = (self.length,)

        self.shape = shape
        self.global_dtype = global_dtype
        self.pushback(chunk)
        self.type_elem = type(chunk)

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

    @staticmethod
    def _get_global_dtype(dtype: list) -> float:
        sizeof = [(np.dtype(cdtype), cdtype) for _, cdtype in dtype]
        return max(sizeof, key=lambda x: x[0])[1]

    def batchs(self, batch_size: int, dtype=None):
        if self.has_batchs is False and batch_size > 0:
            if dtype is None:
                dtype = self.dtype
            it = Iterator(self.chunks_gen(batch_size, dtype),
                          batch_size=batch_size, dtype=dtype)
            it.set_length(self.length)
            return it
        else:
            return self
    
    def chunks_gen(self, batch_size: int, dtype):
        if batch_size < 1:
            batch_size = self.shape[0]

        if len(self.shape) == 1:
            chunk_shape = [batch_size]
        else:
            i_features = self.shape[1:]
            chunk_shape = [batch_size] + list(i_features)

        return BatchWrapper(self, batch_size, dtype).run(chunk_shape)
    
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

    @staticmethod
    def check_datatime(dtype: list):
        cols = []
        for col_i, (_, type_) in enumerate(dtype):
            if type_ == np.dtype('<M8[ns]'):
                cols.append(col_i)
        return cols

    def flat(self):
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
        
        it = Iterator(_iter(), dtype=self.dtype)
        if self.length is not None:
            it.set_length(self.length*sum(self.features_dim))

        if self.has_batchs:
            it = it.batchs(self.batch_size, dtype=self.global_dtype)
        return it

    def sample(self, k: int, col=None, weight_fn=None):
        if self.has_batchs:
            data = self.clean_chunks()
        else:
            data = self
        it = Iterator(wsrj(self.weights_gen(data, col, weight_fn), k), dtype=self.dtype)
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

    def clean_chunks(self):
        if self.has_batchs:
            def cleaner():
                for chunk in self:
                    for row in chunk:
                        yield row
            return Iterator(cleaner(), batch_size=0, dtype=None)
        return self

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, v):
        if hasattr(v, '__iter__'):
            if len(v) == 1:
                self._shape = tuple(v)
                self.features_dim = ()
            else:
                if self.has_batchs:
                    self._shape = tuple([v[0]] + list(v[2:]))
                    self.features_dim = tuple(self._shape[1:])
                else:
                    self._shape = tuple(v)
                    self.features_dim = tuple(v[1:])
        else:
            self._shape = (v,)
            self.features_dim = ()

    def set_length(self, length: int):
        self.length = length
        if length is not None and self.features_dim is not None:
            self._shape = tuple([length] + list(self.features_dim))

    def it_length(self, length: int):
        if self.has_batchs:
            it = Iterator(self.cut_it_chunk(length), dtype=self.dtype,
                          batch_size=self.batch_size)
            it.set_length(length)
            return it
        else:
            self.set_length(length)
            return self

    def scalar_operation(self, op, scalar: float):
        iter_ = map(lambda x: op(x, scalar), self)
        it = Iterator(iter_, dtype=self.dtype, batch_size=self.batch_size)
        it.set_length(self.length)
        return it

    def stream_operation(self, op, stream):
        iter_ = map(lambda x: op(x[0], x[1]), zip(self, stream))
        it = Iterator(iter_, dtype=self.dtype, batch_size=self.batch_size)
        it.set_length(self.length)
        return it

    @choice(operator.add)
    def __add__(self, x):
        return

    @choice(operator.sub)
    def __sub__(self, x):
        return

    @choice(operator.mul)
    def __mul__(self, x):
        return

    @choice(operator.truediv)
    def __truediv__(self, x):
        return

    @choice(pow)
    def __pow__(self, x):
        return

    __rmul__ = __mul__
    __radd__ = __add__

    @choice(operator.iadd)
    def __iadd__(self, x):
        return

    @choice(operator.isub)
    def __isub__(self, x):
        return

    @choice(operator.imul)
    def __imul__(self, x):
        return

    @choice(operator.itruediv)
    def __itruediv__(self, x):
        return

    @choice(operator.ipow)
    def __ipow__(self, x):
        return

    def compose(self, fn, *args, **kwargs):
        iter_ = (fn(x, *args, **kwargs) for x in self)
        return Iterator(iter_, dtype=self.dtype, batch_size=self.batch_size)

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

    def to_ndarray(self, length=None, dtype=None):
        if dtype is None:
            dtype = self.global_dtype
        #if self.has_batchs:
        #    return self._to_narray_chunks(dtype)
        #else:
        #    return self._to_narray_raw(dtype)
        if length is None:
            return self.data.to_ndarray(dtype=dtype)
        else:
            return self.data[:length].ndarray(dtype=dtype)

    @cut
    def _to_narray_chunks(self, dtype):
        smx_a = np.empty(self.shape, dtype=dtype)
        init = 0
        end = 0
        for smx in self:
            end += smx.shape[0]
            smx_a[init:end] = smx
            init = end
        return smx_a, end, self.length

    @cut
    def _to_narray_raw(self, dtype):
        smx_a = np.empty(self.shape, dtype=dtype)
        init = 0
        end = 0
        for smx in self:
            if hasattr(smx, 'shape') and smx.shape == self.shape:
                end += smx.shape[0]
            else:
                end += 1
            smx_a[init:end] = smx
            init = end
        return smx_a, end, self.length

    def to_df(self, length=None):
        if length is None:
            return self.data.to_df()
        else:
            return self.data[:length].to_df()

    def cut_it_chunk(self, length):
        end = 0
        for chunk in self:
            end += chunk.shape[0]
            if end > length:
                mod = length % self.batch_size
                if mod > 0:
                    chunk = chunk[:mod]
                yield chunk
                break     
            yield chunk

    def num_splits(self):
        if self.has_batchs:
            return num_splits(self.length, self.batch_size)
        else:
            return self.length

    def unique(self):
        values = defaultdict(lambda: 0)
        for batch in self:
            u_values, counter = np.unique(batch, return_counts=True)
            for k, v in dict(zip(u_values, counter)).items():
                values[k] += v
        return values

    def __iter__(self):
        if self.length is None or self.iter_init is False:
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

    def __getitem__(self, key):
        return NotImplemented

    def __setitem__(self, key, value):
        return NotImplemented


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
