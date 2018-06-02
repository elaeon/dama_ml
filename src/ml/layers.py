from itertools import izip, imap, chain, tee, islice
import operator
import collections
import types
import numpy as np
import pandas as pd
import psycopg2
import logging
import datetime

from ml.utils.config import get_settings
from ml.utils.seq import grouper_chunk
from ml.utils.numeric_functions import max_type, wsrj, wsr, num_splits

settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


def choice(operator):
    def inner(fn):
        def view(x, y):
            if isinstance(x, IterLayer) and isinstance(y, IterLayer):
                return x.stream_operation(operator, y)
            elif isinstance(y, collections.Iterable):
                return x.stream_operation(operator, IterLayer(y))#, shape=x.length))
            else:
                return x.scalar_operation(operator, y)
        return view
    return inner


class IterLayer(object):
    def __init__(self, fn_iter, dtype=None, chunks_size=0, length=None):
        if isinstance(fn_iter, types.GeneratorType) or isinstance(fn_iter, psycopg2.extensions.cursor):
            self.it = fn_iter
        elif isinstance(fn_iter, IterLayer):
            self.it = fn_iter
        elif isinstance(fn_iter, pd.DataFrame):
            self.it = fn_iter.itertuples(index=False)
            dtype = zip(fn_iter.columns.values, fn_iter.dtypes.values)
        else:
            self.it = (e for e in fn_iter)

        self.pushedback = []
        self.chunks_size = chunks_size
        self.has_chunks = True if chunks_size > 0 else False
        self.features_dim = None
        self.length = length
        #to obtain dtype, shape, global_dtype and type_elem
        self.chunk_taste(dtype)
    
    def columns(self):
        if hasattr(self.dtype, '__iter__'):
            return [c for c, _ in self.dtype]
        else:
            return None

    def pushback(self, val):
        self.pushedback.append(val)

    def chunk_type_elem(self):
        try:
            chunk = next(self)
        except StopIteration:
            return
        else:
            self.pushback(chunk)
            return type(chunk)

    def chunk_taste(self, dtypes):
        """Check for the dtype and global dtype in a chunk"""
        try:
            chunk = next(self)
        except StopIteration:
            self.global_dtype = None
            self.dtype = None
            self.shape = (0,)            
            self.type_elem = None
            return

        if dtypes is not None and hasattr(dtypes, '__iter__'):
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
        else:#scalars
            if type(chunk).__module__ == '__builtin__':
                if hasattr(chunk, '__iter__'):
                    type_e = max_type(chunk)
                    if type_e == list or type_e == tuple or\
                        type_e == str or type_e == unicode or type_e ==  np.ndarray:
                        self.dtype = "|O"
                    else:
                        self.dtype = type_e
                else:
                    self.dtype = type(chunk)
            else:
                self.dtype = chunk.dtype
            global_dtype = self.dtype

        try:
            shape = [None] + list(chunk.shape)
        except AttributeError:
            if hasattr(chunk, '__iter__'):
                shape = (1, len(chunk))
            else:
                shape = (1,)

        self.shape = shape
        self.global_dtype = global_dtype
        self.pushback(chunk)
        self.type_elem = type(chunk)

    def replace_str_type_to_obj(self, dtype):
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

    def _get_global_dtype(self, dtype):
        sizeof = [(np.dtype(cdtype), cdtype) for _, cdtype in dtype]
        return max(sizeof, key=lambda x: x[0])[1]

    def to_chunks(self, chunks_size, dtype=None):
        if self.has_chunks is False and chunks_size > 0:
            if dtype is None:
                dtype = self.dtype
            return IterLayer(self.chunks_gen(chunks_size, dtype),
                chunks_size=chunks_size, dtype=dtype)
        else:
            return self
    
    def chunks_gen(self, chunks_size, dtype):
        if chunks_size < 1:
            chunks_size = self.shape[0]

        if len(self.shape) == 1:
            chunk_shape = [chunks_size]
        else:
            i_features = self.shape[1:]
            chunk_shape = [chunks_size] + list(i_features)

        if not isinstance(dtype, list):
            for smx in grouper_chunk(chunks_size, self):
                smx_a = np.empty(chunk_shape, dtype=dtype)
                for i, row in enumerate(smx):
                    #try:
                    if hasattr(row, '__iter__') and len(row) == 1:
                        smx_a[i] = row[0]
                    else:
                        smx_a[i] = row
                    #except ValueError:
                    #    smx_a[i] = row[0]
                if i + 1 < chunks_size:
                    yield smx_a[:i+1]
                else:
                    yield smx_a
        else:
            columns = [c for c, _ in dtype]
            dt_cols = self.check_datatime(dtype)
            for smx in grouper_chunk(chunks_size, self):
                yield self._assign_struct_array2df(smx, chunk_shape[0], dtype, 
                    dt_cols, columns, chunks_size=chunks_size)
    
    def check_datatime(self, dtype):
        cols = []
        for col_i, (_, type_) in enumerate(dtype):
            if isinstance(type_, datetime.datetime):
                cols.append(col_i)
        return cols

    def to_tuple(self, row, dt_cols):
        if isinstance(row, tuple):
            row = list(row)

        for col_i in dt_cols:
            row[col_i] = datetime.datetime.strptime(row[col_i], "%Y-%m-%d %H:%M:%S")
        return tuple(row)

    def flat(self):
        def _iter():
            if self.type_elem == np.ndarray:
                for chunk in self:
                    for e in chunk.reshape(-1):
                        if hasattr(e, '__iter__') and len(e) == 1:
                            yield e[0]
                        else:
                            yield e
            elif self.type_elem == pd.DataFrame:
                for chunk in self:
                    for e in chunk.values.reshape(-1):
                        yield e
            elif self.type_elem.__module__ == '__builtin__':
                for e in chain.from_iterable(self):
                    yield e
        
        it = IterLayer(_iter(), dtype=self.dtype)
        if self.has_chunks:            
            return it.to_chunks(self.chunks_size, dtype=self.global_dtype)
        else:
            return it

    def sample(self, k, col=None, weight_fn=None):
        if self.has_chunks:
            data = self.clean_chunks()
        else:
            data = self
        return IterLayer(wsrj(self.weights_gen(data, col, weight_fn), k), 
            length=k, dtype=self.dtype)

    def split(self, i):
        if self.type_elem == pd.DataFrame:
            a, b = tee((row.iloc[:, :i], row.iloc[:, i:]) for row in self)
        else:
            a, b = tee((row[:i], row[i:]) for row in self)

        if hasattr(self.dtype, '__iter__'):
            dtype_0 = self.dtype[:i]
            dtype_1 = self.dtype[i:]
        else:
            dtype_0 = self.dtype
            dtype_1 = self.dtype

        it0 = IterLayer((item for item, _ in a), dtype=dtype_0, 
                chunks_size=self.chunks_size)
        it1 = IterLayer((item for _, item in b), dtype=dtype_1, 
                chunks_size=self.chunks_size)
        return it0, it1

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
        if self.has_chunks:
            def cleaner():
                for chunk in self:
                    for row in chunk:
                        yield row
            return IterLayer(cleaner(), chunks_size=0, dtype=None)
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
                self._shape = tuple(v)
                if self.has_chunks:
                    self.features_dim = tuple(v[2:])
                else:
                    self.features_dim = tuple(v[1:])
        else:
            self._shape = (v,)
            self.features_dim = ()

    def length_shape(self, length):
        if length is None and self.length is not None:
            length = self.length
        else:
            self.length = length
        self.shape = [length] + list(self.features_dim)

    def scalar_operation(self, operator, scalar):
        iter_ = imap(lambda x: operator(x, scalar), self)
        return IterLayer(iter_, dtype=self.dtype, 
            chunks_size=self.chunks_size)

    def stream_operation(self, operator, stream):
        iter_ = imap(lambda x: operator(x[0], x[1]), izip(self, stream))
        return IterLayer(iter_, dtype=self.dtype, 
            chunks_size=self.chunks_size)

    @choice(operator.add)
    def __add__(self, x):
        return

    @choice(operator.sub)
    def __sub__(self, x):
        return

    @choice(operator.mul)
    def __mul__(self, x):
        return

    @choice(operator.div)
    def __div__(self, x):
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

    @choice(operator.idiv)
    def __idiv__(self, x):
        return

    @choice(operator.ipow)
    def __ipow__(self, x):
        return

    @classmethod
    def avg(self, iters_b, size, method="arithmetic"):
        iters = iter(iters_b)
        base_iter = next(iters)
        if method == "arithmetic":
            iter_ = (sum(x) / float(size) for x in izip(base_iter, *iters))
        else:
            iter_ = (reduce(operator.mul, x)**(1. / size) for x in izip(base_iter, *iters))
        return IterLayer(iter_, dtype=base_iter.dtype, 
            chunks_size=base_iter.chunks_size)

    @classmethod
    def max_counter(self, iters_b, weights=None):
        def merge(labels, weights):
            if weights is None:
                return ((label, 1) for label in labels)
            else:
                values = {}
                for label, w in izip(labels, weights):
                    values.setdefault(label, 0)
                    values[label] += w
                return values.items()

        iters = iter(iters_b)
        base_iter = next(iters)
        iter_ = (max(merge(x, weights), key=lambda x: x[1])[0] for x in izip(base_iter, *iters))
        return IterLayer(iter_, dtype=base_iter.dtype, 
            chunks_size=base_iter.chunks_size)

    @classmethod
    def concat_n(self, iters):
        if len(iters) > 1:
            base_iter = iters[0]
            length = sum([it.length for it in iters if it.length is not None])
            return IterLayer(chain(*iters), chunks_size=base_iter.chunks_size, 
                length=length)
        elif len(iters) == 1:
            return iters[0]

    def concat_elems(self, data):
        iter_ = (list(chain(x0, x1)) for x0, x1 in izip(self, data))
        return IterLayer(iter_, dtype=self.dtype, chunks_size=self.chunks_size)

    def compose(self, fn, *args, **kwargs):
        iter_ = (fn(x, *args, **kwargs) for x in self)
        return IterLayer(iter_, dtype=self.dtype, 
            chunks_size=self.chunks_size)

    def concat(self, iterlayer):
        return IterLayer(chain(self, iterlayer))

    def to_datamodelset(self, labels, features, size, ltype):
        from ml.ds import DataLabel
        from ml.utils.config import get_settings

        settings = get_settings("ml")

        data = np.zeros((size, features))
        label_m = np.empty(size, dtype=ltype)
        for i, y in enumerate(self):
            row_c = i % labels.shape[0]
            data[i] = y
            label_m[i] = labels[row_c]
        
        dataset = DataLabel(dataset_path=settings["dataset_model_path"])
        with dataset:
            dataset.from_data(data, label_m)
        return dataset

    def to_narray(self, dtype=None):
        if self.shape is None:
            raise Exception("Data shape is None, IterLayer can't be converted to array")
        if dtype is None:
            dtype = self.global_dtype
        
        if self.has_chunks:
            return self._to_narray_chunks(self, dtype)
        else:
            return self._to_narray_raw(self, dtype)

    def _to_narray_chunks(self, it, dtype):
        smx_a = np.empty(self.shape, dtype=dtype)
        init = 0
        end = 0
        for smx in islice(it, num_splits(self.length, self.chunks_size)):
            if isinstance(smx, IterLayer):
                smx = smx.to_narray(self.chunks_size)
                end += smx.shape[0]
            elif hasattr(smx, 'shape'):
                end += smx.shape[0]
            if end > self.length:
                smx = smx[:end-self.length+1]
                end = self.length
            smx_a[init:end] = smx
            init = end
        return smx_a

    def _to_narray_raw(self, it, dtype):
        smx_a = np.empty(self.shape, dtype=dtype)
        init = 0
        end = 0
        for smx in islice(it, self.length):
            if isinstance(smx, IterLayer):
                smx = smx.to_narray(1)
                end += smx.shape[0]
            elif hasattr(smx, 'shape') and smx.shape == self.shape:
                end += smx.shape[0]
            else:
                end += 1
            smx_a[init:end] = smx
            init = end
        return smx_a

    def to_df(self):
        if self.has_chunks:
            def iter_():
                end = 0
                for chunk in islice(self, num_splits(self.length, self.chunks_size)):
                    end += chunk.shape[0]
                    if end > self.length:
                        chunk = chunk.iloc[:end-self.length+1]                    
                    yield chunk
            return pd.concat(iter_(), axis=0, copy=False, ignore_index=True)
        else:
            if hasattr(self.dtype, '__iter__'):
                columns = [c for c, _ in self.dtype]
                dt_cols = self.check_datatime(self.dtype)
                return self._assign_struct_array2df(self, self.length, self.dtype, 
                    dt_cols, columns)
            else:
                return pd.DataFrame((e for e in islice(self, self.length)))

    def _assign_struct_array2df(self, it, length, dtype, dt_cols, columns, chunks_size=0):
        stc_arr = np.empty(length, dtype=dtype)
        i = 0
        for row in islice(it, length):
            try:
                stc_arr[i] = self.to_tuple(row, dt_cols)
            except TypeError:
                stc_arr[i] = row
            i += 1

        smx = pd.DataFrame(stc_arr,
            index=np.arange(0, length), 
            columns=columns)

        if i < chunks_size:
            return smx.iloc[:i]
        else:
            return smx

    def to_memory(self, length=None):
        self.length_shape(length)
        if hasattr(self.dtype, '__iter__'):
            return self.to_df()
        else:
            return self.to_narray()

    def num_splits(self):        
        from ml.utils.numeric_functions import num_splits
        if self.has_chunks:
            return num_splits(self.length, self.chunks_size)
        else:
            return self.length

    def tee(self):
        it0, it1 = tee(self)
        return IterLayer(it0), IterLayer(it1)

    def __iter__(self):
        return self

    def next(self):
        if len(self.pushedback) > 0:
            return self.pushedback.pop()
        try:
            return self.it.next()
        except StopIteration:
            if hasattr(self.it, 'close'):
                self.it.close()
            raise StopIteration
        except psycopg2.InterfaceError:
            raise StopIteration
