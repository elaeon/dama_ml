import hashlib
import numpy as np
import pandas as pd

from collections import OrderedDict
from .decorators import cache
from .numeric_functions import max_dtype


class Hash:
    def __init__(self, hash_fn: str='sha1'):
        self.hash_fn = hash_fn
        self.hash = getattr(hashlib, hash_fn)()

    def update(self, it):
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
            return self.convert(self.columns, key.start, key.stop)
        elif isinstance(key, str):
            columns = [(key, self.o_columns[key])]
            return self.convert_from_columns(columns, 0, self.length)
        elif isinstance(key, list):
            try:
                columns = [(col_name, self.o_columns[col_name]) for col_name in key]
            except KeyError:
                columns = [(self.columns[i][0], self.columns[i][1]) for i in key]
            return self.convert_from_columns(columns, 0, self.length)
        else:
            return self.convert_from_index(key)

    def is_multidim(self):
        return len(self.columns) == 1 and len(self.columns[0][1].shape) >= 2

    @property
    @cache
    def length(self):
        return max([a.shape[0] for _, a in self.columns])

    @property
    def shape(self):
        return tuple([self.length] + list(self.columns[0][1].shape[1:]))

    def columns2dtype(self):
        return [(col_name, array.dtype) for col_name, array in self.columns]

    def convert(self, columns, start_i, end_i):
        if start_i is None:
            start_i = 0

        if end_i is None:
            end_i = self.length

        ncolumns = [(col_name, array[start_i:end_i]) for col_name, array in columns]
        return StructArray(ncolumns)

    def convert_from_index(self, index):
        ncolumns = [(col_name, array[index:index+1]) for col_name, array in self.columns]
        return StructArray(ncolumns)

    @staticmethod
    def convert_from_columns(columns, start_i, end_i):
        ncolumns = [(col_name, array[start_i:end_i]) for col_name, array in columns]
        return StructArray(ncolumns)

    @staticmethod
    def _array_builder(shape, dtype, columns, start_i: int, end_i: int):
        stc_arr = np.empty(shape, dtype=dtype)
        for col_name, array in columns:
            stc_arr[col_name] = array[start_i:end_i]
        return stc_arr

    def to_df(self, init_i: int=0, end_i=None):
        columns = [col_name for col_name, _ in self.dtype]
        data = StructArray._array_builder(self.shape, self.dtype, self.columns, 0, self.length)
        if end_i is None:
            end_i = self.length
        size = abs(end_i - init_i)
        if size > data.shape[0]:
            end_i = init_i + data.shape[0]
        return pd.DataFrame(data, index=np.arange(init_i, end_i), columns=columns)

    def to_ndarray(self, dtype=None):
        if dtype is None:
            dtype = self.global_dtype
        shape = self.shape
        array = StructArray._array_builder(shape, self.dtype, self.columns, 0, self.length)

        if len(shape) == 1:
            shape = list(shape) + [len(self.columns)]

        ndarray = np.empty(shape, dtype=dtype)
        if self.is_multidim():
            columns = [col_name for col_name, _ in self.dtype]
            for i, row in enumerate(array):
                ndarray[i] = row[columns]
        else:
            for i, row in enumerate(array):
                ndarray[i] = tuple(row)
        return ndarray

    @property
    @cache
    def global_dtype(self):
        return max_dtype(self.dtype)


def unique_dtypes(dtypes):
    return np.unique([dtype.name for _, dtype in dtypes])


def labels2num(self):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(self.labels)
    return le
