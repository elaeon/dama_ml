from ml.abc.group import AbsGroup
from ml.abc.data import AbsData
from ml.utils.basic import Shape
import numpy as np
from collections import OrderedDict
import dask.array as da

class DaGroupDict(OrderedDict):
    pass


class DaGroup(object):
    def __init__(self, conn, chunks=None, from_groups=None):
        if isinstance(conn, AbsGroup):
            groups = OrderedDict()
            if from_groups is not None:
                for group in conn.groups:
                    if group in from_groups:
                        groups[group] = conn.conn[group]
            else:
                for group in conn.groups:
                    groups[group] = conn.get_conn_group(group)
            self.conn = self.convert(groups, chunks=chunks)
        elif isinstance(conn, DaGroupDict):
            self.conn = conn
        elif isinstance(conn, dict):
            self.conn = self.convert(conn, chunks=chunks)
        else:
            raise NotImplementedError("Type {} does not supported".format(type(conn)))

    def convert(self, groups_dict, chunks) -> dict:
        groups = DaGroupDict()
        for group, data in groups_dict.items():
            chunks = data.shape  # fixme
            groups[group] = da.from_array(data, chunks=chunks)
        return groups

    def sample(self, index):
        conn = DaGroupDict()
        for group, values in self.conn.items():
            conn[group] = self.conn[group][index]
        return DaGroup(conn=conn)

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.keys()])

    @property
    def groups(self):
        return self.dtypes.names

    def __getitem__(self, item):
        if isinstance(item, slice):
            dict_conn = DaGroupDict()
            for group in self.groups:
                dict_conn[group] = self.conn[group][item]
            return DaGroup(dict_conn)
        elif isinstance(item, str):
            dict_conn = DaGroupDict()
            dict_conn[item] =  self.conn[item]
            return DaGroup(dict_conn)

    def __setitem__(self, key, value):
        pass

    def __add__(self, other: 'DaGroup') -> 'DaGroup':
        if other == 0:
            return self
        groups = DaGroupDict()
        groups.update(self.conn)
        groups.update(other.conn)
        return DaGroup(groups)

    def __radd__(self, other):
        return self.__add__(other)

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

    def rename_group(self, old_name, new_name):
        self.conn[new_name] = self.conn[old_name]
        del self.conn[old_name]

    def store(self, dataset: AbsData):
        from ml.data.it import Iterator
        if dataset.driver.inblock is True:
            batch_size = 2
            init = 0
            end = batch_size
            init_g = init
            end_g = end
            shape = tuple([batch_size] + list(self.shape.to_tuple())[1:])
            while True:
                total_cols = 0
                data = np.empty(shape, dtype=float)
                for group in self.groups:
                    try:
                        num_cols = self.shape[group][1]
                        slice_grp = (slice(init_g, end_g), slice(total_cols, total_cols + num_cols))
                    except IndexError:
                        num_cols = 1
                        slice_grp = (slice(init_g, end_g), total_cols)
                    data[slice_grp] = self.conn[group][init:end].compute()
                    total_cols += num_cols
                init_g = 0
                end_g = batch_size
                dataset.data[init:end] = data
                if end < self.shape.to_tuple()[0]:
                    init = end
                    end += batch_size
                else:
                    break
        else:
            for group in self.groups:
                self.conn[group].store(dataset.data[group])


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
        return self.conn[self.slice]
