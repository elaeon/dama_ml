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
    def __init__(self, columns, labels=None):
        self.columns = columns
        self.dtypes = self.columns2dtype()
        self.dtype = max_dtype(self.dtypes)
        self.o_columns = OrderedDict(self.columns)
        self.labels = labels

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.convert(self.columns, key.start, key.stop)
        elif isinstance(key, str):
            columns = [(key, self.o_columns[key])]
            return self.convert_from_columns(columns, 0, self.length())
        elif isinstance(key, list):
            try:
                columns = [(col_name, self.o_columns[col_name]) for col_name in key]
            except KeyError:
                columns = [(self.columns[i][0], self.columns[i][1]) for i in key]
            return self.convert_from_columns(columns, 0, self.length())
        else:
            return self.convert_from_index(key)

    def is_multidim(self) -> bool:
        return len(self.columns[0][1].shape) >= 2

    @cache
    def length(self) -> int:
        return max([a.shape[0] for _, a in self.columns])

    @property
    def struct_shape(self) -> tuple:
        if len(self.columns) > 1:
            return tuple([self.length()])  # + [1])
        else:
            return tuple([self.length()] + list(self.columns[0][1].shape[1:]))

    @property
    def shape(self):
        if len(self.columns) > 1:
            return tuple([self.length()] + [len(self.columns)])
        else:
            return tuple([self.length()] + list(self.columns[0][1].shape[1:]))

    def columns2dtype(self) -> list:
        return [(col_name, array.dtype) for col_name, array in self.columns]

    def convert(self, columns, start_i, end_i):
        if start_i is None:
            start_i = 0

        if end_i is None:
            end_i = self.length()

        ncolumns = [(col_name, array[start_i:end_i]) for col_name, array in columns]
        return StructArray(ncolumns, labels=self.labels)

    def convert_from_index(self, index: int):
        ncolumns = [(col_name, array[index:index+1]) for col_name, array in self.columns]
        return StructArray(ncolumns)

    @staticmethod
    def convert_from_columns(columns: list, start_i: int, end_i: int):
        ncolumns = [(col_name, array[start_i:end_i]) for col_name, array in columns]
        return StructArray(ncolumns)

    @staticmethod
    def _array_builder(shape, dtypes, columns: list, start_i: int, end_i: int):
        stc_arr = np.empty(shape, dtype=dtypes)
        for col_name, array in columns:
            stc_arr[col_name] = array[start_i:end_i]
        return stc_arr

    def to_df(self, init_i: int=0, end_i=None) -> pd.DataFrame:
        data = StructArray._array_builder(self.struct_shape, self.dtypes, self.columns, 0, self.length())
        if end_i is None:
            end_i = self.length()
        size = abs(end_i - init_i)
        if size > data.shape[0]:
            end_i = init_i + data.shape[0]
        if len(self.dtypes) == 1 and len(data.shape) == 2:
            if self.labels is None:
                self.labels = ["c"+str(i) for i in range(data.shape[1])]
            return pd.DataFrame(data["c0"], index=np.arange(init_i, end_i), columns=self.labels)
        else:
            if self.labels is None:
                self.labels = [col_name for col_name, _ in self.dtypes]
            return pd.DataFrame(data, index=np.arange(init_i, end_i), columns=self.labels)

    def to_ndarray(self, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = self.dtype
        ndarray = np.empty(self.shape, dtype=dtype)
        if len(self.columns) == 1:
            col_name, array = self.columns[0]
            ndarray[:] = array[0:self.length()]
        else:
            if not self.is_multidim():
                for i, (col_name, array) in enumerate(self.columns):
                    ndarray[:, i] = array[0:self.length()]
            else:
                return NotImplemented
        return ndarray

    def to_structured(self):
        return StructArray._array_builder(self.struct_shape, self.dtypes, self.columns, 0, self.length())


def unique_dtypes(dtypes) -> np.ndarray:
    return np.unique([dtype.name for _, dtype in dtypes])


def labels2num(self):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(self.labels)
    return le
