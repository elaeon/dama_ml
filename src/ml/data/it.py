from itertools import chain, islice
import types
import numpy as np
import pandas as pd
import dask.dataframe as dd
import psycopg2
import logging

from collections import defaultdict, deque
from ml.utils.config import get_settings
from ml.utils.numeric_functions import max_type, num_splits, wsrj
from ml.utils.batcher import BatchDataFrame, Batch, BatchIt, BatchArray
from ml.data.abc import AbsDataset


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


class Iterator(object):
    def __init__(self, fn_iter, dtypes=None, batch_size: int=0, length=None, pushedback=None) -> None:
        if isinstance(fn_iter, types.GeneratorType) or isinstance(fn_iter, psycopg2.extensions.cursor):
            self.data = fn_iter
            self.length = length
            self.is_ds = False
        elif isinstance(fn_iter, Iterator):
            self.data = fn_iter
            self.length = fn_iter.length if length is None else length
            self.is_ds = False
        elif isinstance(fn_iter, pd.DataFrame):
            self.data = fn_iter.itertuples(index=False)
            dtypes = list(zip(fn_iter.columns.values, fn_iter.dtypes.values))
            self.length = fn_iter.shape[0] if length is None else length
            self.is_ds = False
        elif isinstance(fn_iter, np.ndarray):
            self.data = iter(fn_iter)
            self.length = fn_iter.shape[0] if length is None else length
            self.is_ds = False
        elif isinstance(fn_iter, AbsDataset):
            self.data = fn_iter
            self.length = fn_iter.shape[0] if length is None else length
            self.is_ds = True
        elif isinstance(fn_iter, Batch):
            self.data = fn_iter
            self.length = length
            self.is_ds = False
        else:
            self.data = iter(fn_iter)
            self.length = length
            self.is_ds = False

        self.pushedback = [] if not isinstance(pushedback, list) else pushedback
        self.dtypes = None
        self.shape = None
        self.dtype = None
        self.type_elem = None
        if batch_size is None:
            batch_size = 0
        self.batch_size = batch_size
        self.has_batchs = True if batch_size > 0 else False
        self.features_dim = None
        self.iter_init = True
        # to obtain dtypes, shape, dtype, type_elem and length
        self.chunk_taste(dtypes)
        print(self.data)
    
    @property
    def columns(self):
        if isinstance(self.dtypes, list):
            return [c for c, _ in self.dtypes]
        else:
            if self.features_dim is not None and len(self.features_dim) > 0:
                return ["c"+str(i) for i in range(self.features_dim[-1])]
            else:
                return ["c0"]

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

    def default_dtypes(self, dtype):
        return [("c0", dtype)]

    def chunk_taste(self, dtypes) -> None:
        """Check for the dtype and global dtype in a chunk"""
        try:
            chunk = next(self)
        except StopIteration:
            self.dtype = None
            self.dtypes = None
            self.shape = (0,)            
            self.type_elem = None
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
            shape = [self.length] + list(chunk.shape)
        except AttributeError:
            if hasattr(chunk, '__iter__') and not isinstance(chunk, str):
                shape = (self.length, len(chunk))
            else:
                shape = (self.length,)

        self.shape = shape
        self.dtype = max_type(self.dtypes)
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

    def batchs(self, batch_size: int, df=True):
        if self.has_batchs is False and batch_size > 0:
            return Iterator(self.chunks_gen(batch_size, df),
                            batch_size=batch_size, dtypes=self.dtypes, length=self.length)
        else:
            return self
    
    def chunks_gen(self, batch_size: int, df: bool=False):
        if self.is_ds:
            if df is True:
                return BatchDataFrame(self, batch_size, df)
            else:
                return BatchArray(self, batch_size, df)
        else:
            return BatchIt(self, batch_size, df)

    
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
            it.set_length(self.length*sum(self.features_dim))

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

    def clean_chunks(self):
        if self.has_batchs:
            def cleaner():
                for chunk in self:
                    for row in chunk:
                        yield row
            return Iterator(cleaner(), batch_size=0, dtypes=None)
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

    # @cut
    # def _to_narray_chunks(self, dtype):
    #    smx_a = np.empty(self.shape, dtype=dtype)
    #    init = 0
    #    end = 0
    #    for smx in self:
    #        end += smx.shape[0]
    #        smx_a[init:end] = smx
    #        init = end
    #    return smx_a, end, self.length

    # @cut
    # def _to_narray_raw(self, dtype):
    #    smx_a = np.empty(self.shape, dtype=dtype)
    #    init = 0
    #    end = 0
    #    for smx in self:
    #        if hasattr(smx, 'shape') and smx.shape == self.shape:
    #            end += smx.shape[0]
    #        else:
    #            end += 1
    #        smx_a[init:end] = smx
    #        init = end
    #    return smx_a, end, self.length

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
            print("POP", self.data)
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
        if isinstance(key, slice):
            if key.stop is not None:
                stop = key.stop
                if self.has_batchs:
                    #self.data.it.pushedback = self.pushedback
                    print("PPPP", self.data.it, self.data.it.pushedback)
                    return Iterator(self.data, dtypes=self.dtypes, length=stop,
                                    batch_size=self.batch_size)
                else:
                    return Iterator(self.data, dtypes=self.dtypes, length=stop, pushedback=self.pushedback)
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
