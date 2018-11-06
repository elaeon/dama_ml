from .seq import grouper_chunk
from .decorators import cut
import numpy as np
import pandas as pd


# cut if length > array size
@cut
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
    return smx, i+1, length


class Batch(object):
    def __init__(self, it, batch_size=258, dtype=None):
        self.it = it
        self.batch_size = batch_size
        self.dtype = dtype if dtype is not None else self.it.dtype


class BatchIt(Batch):
    def batch_from_it_flat(self, shape):
        for smx in grouper_chunk(self.batch_size, self.it):
            smx_a = np.empty(shape, dtype=self.dtype)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row[0]
            yield smx_a[:i+1]

    def batch_from_it_array(self, shape):
        for smx in grouper_chunk(self.batch_size, self.it):
            smx_a = np.empty(shape, dtype=self.dtype)
            i = 0
            for i, row in enumerate(smx):
                smx_a[i] = row
            yield smx_a[:i+1]

    def batch_from_it_df(self, shape):
        start_i = 0
        end_i = 0
        columns = [c for c, _ in self.dtype]
        for smx in grouper_chunk(self.batch_size, self.it):
            end_i += shape[0]
            yield assign_struct_array2df(smx, self.it.type_elem, start_i, end_i, self.dtype,
                                         columns)
            start_i = end_i

    def run(self, shape):
        if isinstance(self.dtype, list):
            return self.batch_from_it_df(shape)
        else:
            if len(self.it.shape) == 2 and self.it.shape[1] == 1:
                return self.batch_from_it_flat(shape)
            else:
                return self.batch_from_it_array(shape)


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


class BatchWrapper(Batch):
    def run(self, shape):
        if self.it.is_ds:
            if isinstance(self.dtype, list):
                return BatchDataFrame(self.it, self.batch_size, self.dtype).run()
            else:
                return BatchArray(self.it, self.batch_size, self.dtype).run()
        else:
            return BatchIt(self.it, self.batch_size, self.dtype).run(shape)
