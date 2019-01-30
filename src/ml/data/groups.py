from ml.abc.group import AbsGroup
from ml.utils.basic import Shape
import numpy as np
from collections import OrderedDict
import dask.array as da


class DaGroup(AbsGroup):
    def __init__(self, conn, name=None, dtypes=None, alias_map=None, chunks=None, from_groups=None):
        if isinstance(conn, AbsGroup):
            groups = OrderedDict()
            if from_groups is not None:
                for group in conn.groups:
                    if group in from_groups:
                        groups[group] = conn.conn[group]
            else:
                for group in conn.groups:
                    groups[group] = conn.conn[group]
            conn = self.convert(groups, chunks=chunks)
        else:
            conn = self.convert(conn, chunks=chunks)
        super(DaGroup, self).__init__(conn, name=name, dtypes=dtypes, alias_map=alias_map)

    def convert(self, groups_dict, chunks) -> dict:
        groups = OrderedDict()
        for group, data in groups_dict.items():
            groups[group] = da.from_array(data, chunks=chunks)
        return groups

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(group, data.dtype) for group, data in self.conn.items()])

    def __getitem__(self, item):
        return self.conn[item]

    def __setitem__(self, key, value):
        pass

    @property
    def shape(self) -> 'Shape':
        shape = {group: data.shape for group, data in self.conn.items()}
        return Shape(shape)

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)):
        from ml.data.drivers import Memory
        from ml.data.ds import Data
        with Data(driver=Memory()) as data:
            data.from_data(self)
            return data.to_ndarray(dtype=dtype)

    def store(self, dataset):
        for group in self.groups:
            self[group].store(dataset.data[group])


class StructuredGroup(AbsGroup):
    def __init__(self, conn, name=None):
        super(StructuredGroup, self).__init__(conn, name=name)

    def __getitem__(self, item):
        if isinstance(item, str):
            key = self.alias_map.get(item, item)
            group = StructuredGroup(self.conn[key])
            group.slice = self.slice
            return group
        else:
            group = StructuredGroup(self.conn)
            if isinstance(item, slice):
                group.slice = item
            elif isinstance(item, int):
                group.slice = slice(item, item + 1)
            else:
                group.slice = self.slice
            return group.to_ndarray()

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
        #print(self.conn, self.slice)
        return self.conn[self.slice]


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