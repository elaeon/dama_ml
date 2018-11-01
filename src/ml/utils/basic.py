import hashlib
import numpy as np
import pandas as pd
from collections import OrderedDict


class Hash:
    def __init__(self, hash_fn:str='sha1'):
        self.hash_fn = hash_fn
        self.hash = getattr(hashlib, hash_fn)()

    def chunks(self, it):
        for chunk in it:
            self.hash.update(chunk)

    def __str__(self):
        return "${hash_fn}${digest}".format(hash_fn=self.hash_fn, digest=self.hash.hexdigest())


class StructArray:
    def __init__(self, columns):
        self.columns = columns
        self.dtype = self.columns2dtype()
        self.o_columns = OrderedDict(self.columns)

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.start is None:
                start = 0
            else:
                start = key.start

            if key.stop is None:
                stop = self.length
            else:
                stop = key.stop
            return self.convert(self.columns, self.shape, start, stop)
        elif isinstance(key, str):
            columns = [(key, self.o_columns[key])]
            return self.convert_from_columns(columns, self.shape, 0, self.length)
        elif isinstance(key, list):
            try:
                columns = [(col_name, self.o_columns[col_name]) for col_name in key]
            except KeyError:
                columns = [(self.columns[i][0], self.columns[i][1]) for i in key]
            return self.convert_from_columns(columns, self.shape, 0, self.length)
        else:
            shape = list(self.columns[0][1].shape[1:])
            return self.convert_from_index(shape, key)


    def is_multidim(self):
        return len(self.columns) == 1 and len(self.columns[0][1].shape) >= 2

    @property
    def length(self):
        return max([a.shape[0] for _, a in self.columns])

    @property
    def shape(self):
        return tuple([self.length] + list(self.columns[0][1].shape[1:]))

    def columns2dtype(self):
        return [(col_name, array.dtype) for col_name, array in self.columns]

    def convert(self, columns, shape, init, end):
        stc_arr = np.empty(shape, dtype=self.dtype)
        for col_name, array in columns:
            stc_arr[col_name] = array[init:end]
        return stc_arr

    def convert_from_index(self, shape, index):
        stc_arr = np.empty(shape, dtype=self.dtype)
        for col_name, array in self.columns:
            stc_arr[col_name] = array[index]
        return stc_arr

    def convert_from_columns(self, columns, shape, init, end):
        o_dtype = OrderedDict(self.dtype)
        dtype = [(col_name, o_dtype[col_name]) for col_name, _ in columns]

        stc_arr = np.empty(shape, dtype=dtype)
        for col_name, array in columns:
            stc_arr[col_name] = array[init:end]
        return stc_arr

    def to_df(self, start_i=0, end_i=None):
        if end_i is None:
            end_i = self.length
        # if len(dataset.shape) > 2:
        #    dataset = dataset.reshape(dataset.shape[0], -1)
        if self.is_multidim():
            columns = ["c{}".format(i) for i in range(self.shape[1])]
            data = self[:][self.columns[0][0]]
        else:
            data = self[:]
            columns = [col_name for col_name, _ in self.dtype]
        return pd.DataFrame(data,
                   index=np.arange(start_i, end_i),
                   columns=columns)


def unique_dtypes(dtypes):
    return np.unique([dtype.name for _, dtype in dtypes])


def labels2num(self):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(self.labels)
    return le