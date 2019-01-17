from .seq import grouper_chunk
import numpy as np
import pandas as pd


def assign_struct_array2df(it, type_elem, start_i, end_i, dtype, columns):
    length = end_i - start_i
    stc_arr = np.empty(length, dtype=dtype)
    i = 0
    if hasattr(type_elem, "__iter__"):
        for i, row in enumerate(it):
            stc_arr[i] = tuple(row)
    else:
        for i, row in enumerate(it):
            stc_arr[i] = row
    smx = pd.DataFrame(stc_arr, index=np.arange(start_i, end_i), columns=columns)
    return smx


class Batch(object):
    def __init__(self, it, batch_size: int=258, df: bool=False):
        self.it = it
        self.batch_size = batch_size
        self.df = df

    def batch_shape(self):
        shape = self.it.shape
        if len(shape) == 1:
            return [self.batch_size]
        else:
            i_features = shape[1:]
            return [self.batch_size] + list(i_features)

    def cut_batch(self, length):
        end = 0
        print("CUT BATCH", self.it, self.it.pushedback)
        for batch in self:
            end += batch.shape[0]
            if end > length:
                mod = length % self.batch_size
                if mod > 0:
                    batch = batch[:mod]
                yield batch
                break
            yield batch

    def __next__(self):
        return next(self.run())

    def __iter__(self):
        print("ITER", self.it, self.it.pushedback)
        return self.run()

    def __getitem__(self, key):
        print("CUT", self.it, self.it.pushedback)
        if isinstance(key, slice):
            if key.stop is not None:
                stop = key.stop
                return self.cut_batch(stop)
        return NotImplemented



class BatchIt(Batch):
    def batch_from_it_flat(self, shape):
        for smx in grouper_chunk(self.batch_size, self.it):
            smx_a = np.empty(shape, dtype=self.it.dtypes)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row[0]
            yield smx_a[:i+1]

    def batch_from_it_array(self, shape):
        for smx in grouper_chunk(self.batch_size, self.it):
            smx_a = np.empty(shape, dtype=self.it.dtypes)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row
            yield smx_a[:i+1]

    def batch_from_it_df(self, shape):
        start_i = 0
        end_i = 0
        columns = self.it.columns
        print("TO DF", self.it, self.it.pushedback)
        for smx in grouper_chunk(self.batch_size, self.it):
            end_i += shape[0]
            yield assign_struct_array2df(smx, self.it.type_elem, start_i, end_i, self.it.dtypes,
                                         columns)
            start_i = end_i

    def run(self):
        batch_shape = self.batch_shape()
        if self.df is True:
            return self.batch_from_it_df(batch_shape)
        else:
            if len(self.it.shape) == 2 and self.it.shape[1] == 1:
                return self.batch_from_it_flat(batch_shape)
            else:
                return self.batch_from_it_array(batch_shape)


class BatchArray(Batch):
    def run(self):
        init = 0
        end = self.batch_size
        length = self.batch_size
        while length > 0:
            batch = self.it.data[init:end].to_ndarray(dtype=self.dtype)
            yield batch
            init = end
            end += self.batch_size
            length = batch.shape[0]


class BatchDataFrame(Batch):
    def run(self):
        init = 0
        end = self.batch_size
        length = self.batch_size
        while length > 0:
            batch = self.it.data[init:end].to_df(init_i=init, end_i=end)
            yield batch
            init = end
            end += self.batch_size
            length = batch.shape[0]

