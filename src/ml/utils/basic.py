import hashlib
import numpy as np
import pandas as pd
import xarray as xr
import numbers

from collections import OrderedDict
from .decorators import cache
from .numeric_functions import max_dtype


class Hash:
    def __init__(self, hash_fn: str='sha1'):
        self.hash_fn = hash_fn
        self.hash = getattr(hashlib, hash_fn)()

    def update(self, it):
        if it.dtype == np.dtype('<M8[ns]'):
            for chunk in it:
                self.hash.update(chunk.astype('object'))
        else:
            for chunk in it:
                self.hash.update(chunk)

    def __str__(self):
        return "${hash_fn}${digest}".format(hash_fn=self.hash_fn, digest=self.hash.hexdigest())


class StructArray:
    def __init__(self, labels_data):
        self.labels_data = labels_data
        self.dtypes = self.columns2dtype()
        self.dtype = max_dtype(self.dtypes)
        self.o_columns = OrderedDict(self.labels_data)
        self.groups = list(self.o_columns.keys())
        self.counter = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.convert(self.labels_data, key.start, key.stop)
        elif isinstance(key, str):
            columns = [(key, self.o_columns[key])]
            return self.convert_from_columns(columns, 0, len(self))
        elif isinstance(key, list):
            try:
                columns = [(col_name, self.o_columns[col_name]) for col_name in key]
            except KeyError:
                columns = [(self.labels_data[i][0], self.labels_data[i][1]) for i in key]
            return self.convert_from_columns(columns, 0, len(self))
        elif isinstance(key, np.ndarray) and key.dtype == np.dtype('int'):
            return self.convert_from_array(self.labels_data, key)
        else:
            return self.convert_from_index(key)

    def rename_group(self, old, new):
        self.o_columns[new] = self.o_columns[old]
        del self.o_columns[old]
        self.labels_data = list(self.o_columns.items())

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        try:
            elem = self[self.counter]
        except IndexError:
            self.counter = 0
            raise StopIteration
        else:
            self.counter += 1
            return elem

    def __add__(self, other: 'StructArray') -> 'StructArray':
        if other == 0:
            return self
        groups = {}
        groups.update(self.o_columns)
        groups.update(other.o_columns)
        return StructArray(groups.items())

    def __radd__(self, other):
        return self.__add__(other)

    @cache
    def __len__(self):
        if len(self.labels_data) > 0:
            values = [len(a) for _, a in self.labels_data if hasattr(a, '__len__')]
            if len(values) > 0:
                return max(values)
            else:
                return 1
        else:
            return 0

    def is_multidim(self) -> bool:
        shape_values = list(self.shape.values())
        if len(shape_values) == 0:
            return False

        group_shape_0 = shape_values[0]
        for group_shape in shape_values[1:]:
            if group_shape != group_shape_0:
                return True
        return False

    @property
    def struct_shape(self) -> tuple:
        if len(self.labels_data) > 1:
            return tuple([len(self)])
        else:
            return tuple([len(self)] + list(self.labels_data[0][1].shape[1:]))

    @property
    @cache
    def shape(self) -> 'Shape':
        shapes = {}
        for group, data in self.labels_data:
            if hasattr(data, 'shape'):
                shape = data.shape
                if isinstance(shape, Shape):
                    shapes[group] = shape[group]
                elif len(shape) >= 1:
                    shapes[group] = shape
                else:
                    shapes[group] = ()
            else:
                shapes[group] = ()
        return Shape(shapes)

    @property
    def ushape(self) -> tuple:
        return self.shape.to_tuple()

    def columns2dtype(self) -> list:
        dtypes = []
        for col_name, array in self.labels_data:
            try:
                dtypes.append((col_name, array.dtype))
            except AttributeError:
                dtypes.append((col_name, np.dtype(type(array))))
        return dtypes

    def convert(self, labels_data, start_i, end_i):
        if start_i is None:
            start_i = 0

        if end_i is None:
            end_i = len(self)

        sub_labels_data = [(group, array[start_i:end_i]) for group, array in labels_data]
        return StructArray(sub_labels_data)

    def convert_from_array(self, labels_data, index_array):
        sub_labels_data = [(label, array[index_array]) for label, array in labels_data]
        return StructArray(sub_labels_data)

    def convert_from_index(self, index: int):
        sub_labels_data = []
        for group, array in self.labels_data:
            sub_labels_data.append((group, array[index]))
        return StructArray(sub_labels_data)

    @staticmethod
    def convert_from_columns(labels_data: list, start_i: int, end_i: int):
        sub_labels_data = []
        for label, array in labels_data:
            try:
                sub_labels_data.append((label, array[start_i:end_i]))
            except IndexError:
                if isinstance(array, str) or isinstance(array, numbers.Number):
                    sub_labels_data.append((label, array))
                else:
                    raise IndexError
        return StructArray(sub_labels_data)

    @staticmethod
    def _array_builder(shape, dtypes, columns: list, start_i: int, end_i: int):
        stc_arr = np.empty(shape, dtype=dtypes)
        for col_name, array in columns:
            stc_arr[col_name] = array[start_i:end_i]
        return stc_arr

    def to_df(self, init_i: int=0, end_i=None) -> pd.DataFrame:
        data = StructArray._array_builder(self.struct_shape, self.dtypes, self.labels_data, 0, len(self))
        if end_i is None:
            end_i = len(self)
        size = abs(end_i - init_i)
        if size > data.shape[0]:
            end_i = init_i + data.shape[0]
        if len(self.dtypes) == 1 and len(data.shape) == 2:
            columns = ["c"+str(i) for i in range(data.shape[1])]
            group, _ = self.dtypes[0]
            return pd.DataFrame(data[group], index=np.arange(init_i, end_i), columns=columns)
        else:
            return pd.DataFrame(data, index=np.arange(init_i, end_i), columns=self.groups)

    def to_ndarray(self, dtype: list=None) -> np.ndarray:
        if dtype is None:
            dtype = self.dtype
        if not self.is_multidim():
            ushape = self.ushape
            ndarray = np.empty(ushape, dtype=dtype)
            if len(self.labels_data) == 1:
                if ushape[0] == 1:
                    ndarray[0] = self.o_columns[self.groups[0]]
                else:
                    for i, (_, array) in enumerate(self.labels_data):
                        ndarray[:] = array[0:len(self)]
            else:
                if len(ushape) == 1:
                    for i, (_, array) in enumerate(self.labels_data):
                        ndarray[i] = array
                elif ushape[0] == 1:
                    for i, (_, array) in enumerate(self.labels_data):
                        ndarray[:, i] = array
                else:
                    for i, (_, array) in enumerate(self.labels_data):
                        ndarray[:, i] = array[0:len(self)]
        else:
            raise NotImplementedError
        return ndarray

    def to_xrds(self) -> xr.Dataset:
        xr_data = {}
        for group, data in self.labels_data:
            index_dims = ["{}_{}".format(group, i) for i in range(len(data.shape))]
            if isinstance(data, StructArray):
                data_dims = (index_dims, data.to_ndarray())
            else:
                data_dims = (index_dims, data)
            xr_data[group] = data_dims
        return xr.Dataset(xr_data)


class Shape(object):
    def __init__(self, shape: dict):
        self._shape = shape

    def __getitem__(self, item):
        if isinstance(item, numbers.Number) and item == 0:
            return self.max_length
        elif isinstance(item, str):
            return self._shape[item]
        elif isinstance(item, slice):
            return self.to_tuple()[item.start:item.stop]
        else:
            raise IndexError

    def __len__(self):
        return len(self._shape)

    def __eq__(self, other):
        return self.to_tuple() == other

    def __str__(self):
        return str(self._shape)

    def items(self):
        return self._shape.items()

    def values(self):
        return self._shape.values()

    def get_dim_shape(self, dim, shapes):
        values = []
        for shape in shapes:
            try:
                values.append(shape[dim])
            except IndexError:
                pass
        return values

    def to_tuple(self):
        # if we have different lengths return dict of shapes
        shapes = list(self._shape.values())
        if len(shapes) == 0:
            return (0,)

        dim = 0
        nshape = []
        while dim < len(max(shapes)):
            nshape.append(max(self.get_dim_shape(dim, shapes)))
            dim += 1
        num_groups = len(self._shape)
        if num_groups > 1:
            nshape.insert(1, num_groups)
        elif num_groups == 1 and len(nshape) == 0:
            nshape.append(num_groups)
        return tuple(nshape)

    @property
    def max_length(self):
        if len(self._shape) > 0:
            values = [a[0] for a in self._shape.values() if len(a) > 0]
            if len(values) == 0:
                return 0
            else:
                return max(values)
        else:
            return 0

    def change_length(self, length) -> 'Shape':
        shapes = OrderedDict()
        for group, shape in self.items():
            shapes[group] = tuple([length] + list(shape[1:]))
        return Shape(shapes)


def unique_dtypes(dtypes) -> np.ndarray:
    return np.unique([dtype.name for _, dtype in dtypes])


def labels2num(labels):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(labels)
    return le


def isnamedtupleinstance(x):
    f = getattr(x, '_fields', None)
    shape = getattr(x, 'shape', None)
    return f is not None and shape is None  # x.__bases__[0] == tuple
