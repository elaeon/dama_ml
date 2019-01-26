from ml.abc.group import AbsGroup
from ml.utils.basic import Shape
import numpy as np


class StructuredGroup(AbsGroup):
    def __init__(self, conn, name=None):
        super(StructuredGroup, self).__init__(conn, name=name)

    def __getitem__(self, item):
        group = StructuredGroup(self.conn[item])
        if isinstance(item, slice):
            group.slice = item
        elif isinstance(item, int):
            group.slice = slice(item, item + 1)
        else:
            group.slice = self.slice
        return group

    def __setitem__(self, item, value):
        if hasattr(value, "groups"):
            for group in value.groups:
                self.conn[group][item] = value[group]
        #else:
        #    group = list(value.keys())[0]
        #    self.conn[group][item] = value[group]

    @property
    def dtypes(self) -> np.dtype:
        if isinstance(self.conn, np.ndarray):
            if self.conn.dtype.fields is None:
                return self.conn.dtype
            else:
                return np.dtype([(self.inv_map.get(group, group), dtype) for group, (dtype, _) in self.conn.dtype.fields.items()])
        elif isinstance(self.conn, np.void):
            return np.dtype(
                [(self.inv_map.get(group, group), dtype) for group, (dtype, _) in self.conn.dtype.fields.items()])

    @property
    def shape(self) -> Shape:
        if isinstance(self.conn, np.ndarray):
            if self.groups is None:
                shape = dict([("c0", self.conn.shape)])
            else:
                shape = dict([(group, self.conn[group].shape) for group in self.groups])
        else:
            shape = dict([(group, self.conn.shape) for group in self.groups])
        return Shape(shape)

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        return self.conn


class NumpyArrayGroup(AbsGroup):
    def __init__(self, conn, name=None, dtypes=None):
        super(NumpyArrayGroup, self).__init__(conn, name=name, dtypes=dtypes)

    def __getitem__(self, item):
        if isinstance(item, str):
            group = NumpyArrayGroup(self.conn[item])
        elif isinstance(item, slice):
            group = NumpyArrayGroup(self.conn[item], dtypes=self.dtypes)
            group.slice = item
        elif isinstance(item, int):
            group = NumpyArrayGroup(self.conn[item], dtypes=self.dtypes)
            group.slice = slice(item, item + 1)
        else:
            group = NumpyArrayGroup(self.conn[item], dtypes=self.dtypes)
            group.slice = self.slice
        return group

    def __setitem__(self, item, value):
        if hasattr(value, "groups"):
            for group in value.groups:
                self.conn[group][item] = value[group]
        #else:
        #    group = list(value.keys())[0]
        #    self.conn[group][item] = value[group]

    @property
    def dtypes(self) -> np.dtype:
        if isinstance(self.conn, dict):
            return np.dtype([(self.inv_map.get(group, group), elem.dtype) for group, elem in self.conn.items()])
        elif isinstance(self.conn, np.ndarray):
            return np.dtype([(self.inv_map.get(group, group), dtype)
                             for group, (dtype, _) in self.static_dtypes.fields.items()])

    @property
    def shape(self) -> Shape:
        if isinstance(self.conn, dict):
            shape = dict([(group, self.conn[group].shape) for group in self.groups])
        else:
            shape = dict([(group, self.conn.shape) for group in self.groups])
        return Shape(shape)

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        if isinstance(self.conn, dict):
            return self.conn[self.groups[0]]
        else:
            return self.conn