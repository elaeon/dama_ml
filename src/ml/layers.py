from itertools import izip, imap, chain, tee
import operator
import collections
import types
import numpy as np
import pandas as pd

from ml.utils.seq import grouper_chunk

class struct_array(np.recarray):
    @property
    def sshape(self):
        return tuple(list(self.shape) + [len(self.dtype.names)])


#class df(pd.DataFrame):
#    @property
#    def sshape(self):
#        return tuple(list(self.shape) + [len(self.dtype.names)])


def choice(operator):
    def inner(fn):
        def view(x, y):
            if isinstance(x, IterLayer) and isinstance(y, IterLayer):
                return x.stream_operation(operator, y)
            elif isinstance(y, collections.Iterable):
                return x.stream_operation(operator, IterLayer(y))
            else:
                return x.scalar_operation(operator, y)
        return view
    return inner


class IterLayer(object):
    def __init__(self, fn_iter, shape=None, dtype=None, has_chunks=False, chunks_size=0):
        if isinstance(fn_iter, types.GeneratorType):
            _fn_iter = fn_iter
        elif isinstance(fn_iter, IterLayer):
            _fn_iter = fn_iter
        elif isinstance(fn_iter, pd.DataFrame):
            _fn_iter = (e for e in fn_iter.values)
            dtype = zip(fn_iter.columns.values, fn_iter.dtypes.values)
        else:
            _fn_iter = (e for e in fn_iter)

        self.it = _fn_iter
        self.pushedback = []
        self.shape = shape
        self.chunks_size = chunks_size
        self.has_chunks = has_chunks
        self.chunk_taste(dtype)

    def pushback(self, val):
        self.pushedback.append(val)

    def chunk_taste(self, dtype):
        chunk = next(self)
        if isinstance(chunk, pd.DataFrame):
            self.dtype = []
            for c, cdtype in zip(chunk.columns.values, chunk.dtypes.values):
                if cdtype == np.dtype("|O"):
                    global_dtype = cdtype
                self.dtype.append((c, cdtype))
            global_dtype = cdtype
        elif hasattr(dtype, '__iter__'):
            self.dtype = []
            for c, cdtype in dtype:
                if cdtype == np.dtype("|O"):
                    global_dtype = cdtype
                self.dtype.append((c, cdtype))
            global_dtype = cdtype
        elif isinstance(chunk, np.ndarray):
            self.dtype = chunk.dtype
            global_dtype = self.dtype
        else:#scalars
            if type(chunk).__module__ == '__builtin__':
                self.dtype = '|O'
            else:
                self.dtype = chunk.dtype
            global_dtype = self.dtype
        self.global_dtype = global_dtype
        self.pushback(chunk)

    def to_chunks(self, chunks_size):
        if self.has_chunks is False:
            return IterLayer(self.gen_chunks(self, chunks_size), shape=self.shape, 
                chunks_size=chunks_size, has_chunks=True)
        else:
            return self
        
    def gen_chunks(self, fn_iter, chunks_size):
        if chunks_size < 1:
            chunks_size = self.shape[0]

        if len(self.shape) == 1:
            chunk_shape = [chunks_size]
        else:
            i_features = self.shape[1:]
            chunk_shape = [chunks_size] + list(i_features)

        if not isinstance(self.dtype, list):
            for smx in grouper_chunk(chunks_size, fn_iter):
                smx_a = np.empty(chunk_shape, dtype=np.dtype(self.dtype))
                for i, row in enumerate(smx):
                    try:
                        smx_a[i] = row
                    except ValueError:
                        smx_a[i] = row[0]
                if i + 1 < chunks_size:
                    yield smx_a[:i+1]
                else:
                    yield smx_a
        else:
            for smx in grouper_chunk(chunks_size, fn_iter):
                smx_a = pd.DataFrame(np.empty(chunk_shape, 
                    dtype=self.dtype[0][1]), columns=[c for c, _ in self.dtype])
                for i, row in enumerate(smx):
                    try:
                        smx_a.values[i, :] = row
                    except ValueError:
                        smx_a.values[i, :] = row[0]
                if i + 1 < chunks_size:
                    yield smx_a[:i+1]
                else:
                    yield smx_a

    def flat(self):
        def _iter():
            for chunk in self:
                for e in chunk.reshape(-1):
                    yield e

        shape = (reduce(operator.mul, self.shape),)
        it = IterLayer(_iter(), shape=shape, dtype=self.dtype)
        if self.has_chunks:
            return it.to_chunks(self.chunks_size)
        else:
            return it

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, v):
        if isinstance(v, list):
            self._shape = tuple(v)
        else:
            self._shape = v

    def scalar_operation(self, operator, scalar):
        iter_ = imap(lambda x: operator(x, scalar), self)
        return IterLayer(iter_, shape=self.shape, dtype=self.dtype, 
            has_chunks=self.has_chunks, chunks_size=self.chunks_size)

    def stream_operation(self, operator, stream):
        iter_ = imap(lambda x: operator(x[0], x[1]), izip(self, stream))
        return IterLayer(iter_, shape=self.shape, dtype=self.dtype, 
            has_chunks=self.has_chunks, chunks_size=self.chunks_size)

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
        return IterLayer(iter_, shape=base_iter.shape, dtype=base_iter.dtype, 
            has_chunks=base_iter.has_chunks, chunks_size=base_iter.chunks_size)

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
        return IterLayer(iter_, shape=base_iter.shape, dtype=base_iter.dtype, 
            has_chunks=base_iter.has_chunks, chunks_size=base_iter.chunks_size)

    @classmethod
    def concat_n(self, iters):
        return IterLayer(chain(*iters))

    def concat_elems(self, data):
        iter_ = (list(chain(x0, x1)) for x0, x1 in izip(self, data))
        return IterLayer(iter_, shape=None, dtype=self.dtype, 
            has_chunks=self.has_chunks, chunks_size=self.chunks_size)

    def compose(self, fn, *args, **kwargs):
        iter_ = (fn(x, *args, **kwargs) for x in self)
        return IterLayer(iter_, shape=self.shape, dtype=self.dtype, 
            has_chunks=self.has_chunks, chunks_size=self.chunks_size)

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
        
        #fixme: add a dataset chunk writer
        dataset = DataLabel(dataset_path=settings["dataset_model_path"])
        with dataset:
            dataset.build_dataset(data, label_m)
        return dataset

    def to_narray(self, dtype=None):
        if self.shape is None:
            raise Exception("Data shape is None, IterLayer can't be converted to array")
        if dtype is None:
            dtype = self.global_dtype
        return self._to_narray(self, self.shape, dtype)

    def _to_narray(self, it, shape, dtype):
        smx_a = np.empty(shape, dtype=dtype)
        init = 0
        end = 0
        for smx in it:
            if len(smx.shape) >= 1 and self.has_chunks:
                end += smx.shape[0]
            else:
                end += 1
            smx_a[init:end] = smx
            init = end
        return smx_a

    def to_df(self):
        if self.has_chunks:
            return pd.concat((chunk for chunk in self), axis=0, copy=False, ignore_index=True)
        else:
            if hasattr(self.dtype, '__iter__'):
                return pd.DataFrame((e for e in self), columns=[c for c, _ in self.dtype])
            else:
                return pd.DataFrame((e for e in self))

    def to_memory(self):
        if hasattr(self.dtype, '__iter__'):
            return self.to_df()
        else:
            return self.to_narray()

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
