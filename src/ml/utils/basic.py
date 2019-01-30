import hashlib
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import numbers
import time

from collections import OrderedDict
from .decorators import cache
from .numeric_functions import max_dtype


class Hash:
    def __init__(self, hash_fn: str = 'sha1'):
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

    def __add__(self, other: 'StructArray') -> 'StructArray':
        if other == 0:
            return self
        groups = {}
        groups.update(self.o_columns)
        groups.update(other.o_columns)
        return StructArray(groups.items())

    def __radd__(self, other):
        return self.__add__(other)

    def is_multidim(self) -> bool:
        shape_values = list(self.shape.values())
        if len(shape_values) == 0:
            return False

        group_shape_0 = shape_values[0]
        for group_shape in shape_values[1:]:
            if group_shape != group_shape_0:
                return True
        return False


class Shape(object):
    def __init__(self, shape: dict):
        self._shape = shape

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.to_tuple()[item]
        elif isinstance(item, str):
            return self._shape[item]
        elif isinstance(item, slice):
            return self.to_tuple()[item.start:item.stop]
        else:
            raise IndexError

    def __iter__(self):
        return iter(self.to_tuple())

    def __len__(self):
        """ To add compatibility with shapes that are in tuple form, we define the Shape lenght
        as the lenght of its tuple"""
        return len(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other

    def __str__(self):
        return str(self._shape)

    def groups(self):
        return self._shape.keys()

    def items(self):
        return self._shape.items()

    def values(self):
        return self._shape.values()

    @staticmethod
    def get_dim_shape(dim, shapes) -> list:
        values = []
        for shape in shapes:
            try:
                values.append(shape[dim])
            except IndexError:
                pass
        return values

    @cache
    def to_tuple(self) -> tuple:
        # if we have different lengths return dict of shapes
        shapes = list(self._shape.values())
        if len(shapes) == 0:
            return tuple([0])

        dim = 0
        nshape = []
        while dim < len(max(shapes)):
            nshape.append(max(Shape.get_dim_shape(dim, shapes)))
            dim += 1
        num_groups = len(self._shape)
        if num_groups > 1:
            nshape.insert(1, num_groups)
        elif num_groups == 1 and len(nshape) == 0:
            nshape.append(num_groups)
        return tuple(nshape)

    @property
    def max_length(self) -> int:
        if len(self._shape) > 0:
            values = [a[0] for a in self._shape.values() if len(a) > 0]
            if len(values) > 0:
                return max(values)
        return 0

    def change_length(self, length) -> 'Shape':
        shapes = OrderedDict()
        for group, shape in self.items():
            shapes[group] = tuple([length] + list(shape[1:]))
        return Shape(shapes)


class Array(da.Array):
    @property
    def shape(self) -> Shape:
        tuple_shape = super(Array, self).shape
        shape = Shape({"c0": tuple_shape})
        return shape

    @property
    def dtypes(self) -> list:
        return [("c0", self.dtype)]

    @staticmethod
    def from_da(array):
        return Array(array.dask, chunks=array.chunks, dtype=array.dtype, name=array.name)


class Login(object):
    def __init__(self, username, resource, passwd: str = None):
        self.username = username
        self.passwd = passwd
        self.resource = resource


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


def time2str(date):
    return time.strftime("%a, %d %b %Y %H:%M", time.gmtime(date))
