from ml.abc.group import AbsGroup, AbsBaseGroup
from ml.abc.data import AbsData
from ml.utils.core import Shape
from ml.fmtypes import DEFAUL_GROUP_NAME
import numpy as np
from collections import OrderedDict
from ml.utils.decorators import cache
import dask.array as da
import numbers


class DaGroupDict(OrderedDict):
    def __init__(self, *args, map_rename=None, **kwargs):
        super(DaGroupDict, self).__init__(*args, **kwargs)
        if map_rename is None:
            self.map_rename = {}
        else:
            self.map_rename = map_rename

    def rename(self, key, new_key) -> 'DaGroupDict':
        map_rename = self.map_rename.copy()
        map_rename[new_key] = key
        return DaGroupDict(((new_key if k == key else k, v) for k, v in self.items()), map_rename=map_rename)

    def get_oldname(self, name) -> str:
        return self.map_rename.get(name, name)


class DaGroup(AbsGroup):
    def __init__(self, conn, chunks=None, writer_conn=None):
        if isinstance(conn, DaGroupDict):
            reader_conn = conn
        elif isinstance(conn, dict):
            reader_conn = self.convert(conn, chunks=chunks)
        elif isinstance(conn, AbsBaseGroup):  # or AbsGroup:
            groups = OrderedDict()
            for group in conn.groups:
                groups[group] = conn.get_conn(group)
            reader_conn = self.convert(groups, chunks=chunks)
            if writer_conn is None:
                writer_conn = conn
        else:
            raise NotImplementedError("Type {} does not supported".format(type(conn)))
        super(DaGroup, self).__init__(reader_conn, writer_conn=writer_conn)

    def convert(self, groups_dict, chunks) -> dict:
        groups = DaGroupDict()
        for group, data in groups_dict.items():
            chunks = data.shape  # fixme
            lock = False
            print(chunks, "CONVERT CHUNKS")
            groups[group] = da.from_array(data, chunks=chunks, lock=lock)
        return groups

    def sample(self, index):
        return self.set_values(self.groups, index)

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.keys()])

    def set_values(self, groups, item) -> 'DaGroup':
        dict_conn = DaGroupDict(map_rename=self.conn.map_rename)
        for group in groups:
            dict_conn[group] = self.conn[group][item]
        return DaGroup(dict_conn, writer_conn=self.writer_conn)

    def __getitem__(self, item) -> 'DaGroup':
        if isinstance(item, slice):
            return self.set_values(self.groups, item)
        elif isinstance(item, str):
            dict_conn = DaGroupDict(map_rename=self.conn.map_rename)
            dict_conn[item] = self.conn[item]
            return DaGroup(dict_conn, writer_conn=self.writer_conn.get_group(self.conn.get_oldname(item)))
        elif isinstance(item, int):
            return self.set_values(self.groups, item)
        elif isinstance(item, list):
            dict_conn = DaGroupDict(map_rename=self.conn.map_rename)
            for group in item:
                dict_conn[group] = self.conn[group]
            return DaGroup(dict_conn, writer_conn=self.writer_conn)
        elif isinstance(item, np.ndarray) and item.dtype == np.dtype(int):
            return self.sample(item)

    def __setitem__(self, item, value):
        if self.writer_conn.inblock is True:
            self.writer_conn[item] = value
        else:
            if hasattr(value, "groups"):
                for group in value.groups:
                    group = self.conn.get_oldname(group)
                    self.writer_conn.conn[group][item] = value[group].to_ndarray()
            elif hasattr(value, 'batch'):
                for group in value.batch.groups:
                    group = self.conn.get_oldname(group)
                    self.writer_conn.conn[group][item] = value.batch[group].to_ndarray()
            elif isinstance(value, numbers.Number):
                self.writer_conn.conn[item] = value
            elif isinstance(value, np.ndarray):
                self.writer_conn.conn[item] = value
            else:
                if isinstance(item, str):
                    self.writer_conn.conn[item] = value

    def __add__(self, other: 'DaGroup') -> 'DaGroup':
        if other == 0:
            return self
        groups = DaGroupDict()
        groups.update(self.conn)
        groups.update(other.conn)
        return DaGroup(groups, writer_conn=self.writer_conn)

    def __radd__(self, other):
        return self.__add__(other)

    @staticmethod
    def concat(da_groups, axis=0) -> 'DaGroup':
        writers = {group.writer_conn.module_cls_name() for group in da_groups}
        if len(writers) > 1:
            raise Exception
        else:
            if axis == 0:
                groups = [da_group.groups for da_group in da_groups]
                intersection_groups = set(groups[0])
                for group in groups[1:]:
                    intersection_groups = intersection_groups.intersection(set(group))
                da_group_dict = DaGroupDict()
                for group in intersection_groups:
                    da_arrays = [da_group[group].array() for da_group in da_groups]
                    da_array_c = da.concatenate(da_arrays, axis=axis)
                    da_group_dict[group] = da_array_c
                return DaGroup(da_group_dict)
            else:
                raise NotImplementedError

    @property
    def shape(self) -> 'Shape':
        shape = {group: data.shape for group, data in self.conn.items()}
        return Shape(shape)

    def array(self):
        if len(self.groups) == 1:
            return self.conn[self.groups[0]]
        else:
            raise NotImplementedError

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)):
        self.writer_conn.attrs["dtype"] = dtype
        if len(self.groups) == 1:
            computed_array = self.conn[self.groups[0]].compute(dtype=self.dtype)
            if self.writer_conn.base_cls() == AbsBaseGroup and dtype is not None and dtype != self.dtype:
                return computed_array.astype(dtype)
            return computed_array
        else:
            shape = self.shape.to_tuple()
            if dtype is None:
                dtype = self.dtype
            data = np.empty(shape, dtype=dtype)
            total_cols = 0
            for i, group in enumerate(self.groups):
                try:
                    num_cols = self.shape[group][1]
                    slice_grp = (slice(None, None), slice(total_cols, total_cols + num_cols))
                except IndexError:
                    num_cols = 1
                    slice_grp = (slice(None, None), total_cols)
                total_cols += num_cols
                data[slice_grp] = self.conn[group].compute(dtype=dtype)
            return data

    def to_df(self):
        from ml.data.it import Iterator
        import pandas as pd
        shape = self.shape
        if len(shape) > 1 and len(self.groups) < shape[1]:
            dtypes = np.dtype([("c{}".format(i), self.dtype) for i in range(shape[1])])
            columns = self.groups
        else:
            dtypes = self.dtypes
            columns = self.groups

        data = Iterator(self).batchs(batch_size=258)
        stc_arr = np.empty(shape.to_tuple()[0], dtype=dtypes)
        if len(self.groups) == 1:
            for slice_obj in data:
                stc_arr[slice_obj.slice] = slice_obj.batch.to_ndarray()
        else:
            for slice_obj in data:
                for group, (dtype, _) in self.dtypes.fields.items():
                    stc_arr[group][slice_obj.slice] = slice_obj.batch[group].to_ndarray(dtype)
        return pd.DataFrame(stc_arr, index=np.arange(0, stc_arr.shape[0]), columns=columns)

    def rename_group(self, old_name, new_name):
        self.conn = self.conn.rename(old_name, new_name)

    def store(self, dataset: AbsData):
        self.writer_conn = dataset.data.writer_conn
        if self.writer_conn.inblock is True:
            from ml.data.it import Iterator
            for e in Iterator(self).batchs(batch_size=258):
                dataset.data[e.slice] = e.batch.to_ndarray()
        else:
            for group in self.groups:
                self.conn[group].store(dataset.data[group])

    @staticmethod
    def from_da(da_array: da.Array, group_name: str = DEFAUL_GROUP_NAME):
        da_group_dict = DaGroupDict()
        da_group_dict[group_name] = da.Array(da_array.dask, chunks=da_array.chunks,
                                             dtype=da_array.dtype, name=da_array.name)
        return DaGroup(da_group_dict)


class StcArrayGroup(AbsBaseGroup):
    inblock = False

    @property
    def dtypes(self) -> np.dtype:
        return self.conn.dtype

    def get_group(self, group) -> AbsBaseGroup:
        return StcArrayGroup(self.conn[group])

    def get_conn(self, group):
        return self.conn[group]


class TupleGroup(AbsGroup):
    inblock = False

    def __init__(self, conn, dtypes=None):
        self.dtypes = dtypes
        super(TupleGroup, self).__init__(conn)

    def __getitem__(self, item):
        if isinstance(item, str):
            for index, (group, (dtype, _)) in enumerate(self.dtypes.fields.items()):
                if group == item:
                    return TupleGroup(self.conn[index], dtypes=np.dtype([(group, dtype)]))
        elif isinstance(item, list) or isinstance(item, tuple):
            if isinstance(self.conn, np.ndarray):
                return self.conn[item]
            else:
                raise NotImplementedError
        elif isinstance(item, int):
            raise NotImplementedError
        elif isinstance(item, slice):
            raise NotImplementedError

    def __setitem__(self, item, value):
        pass

    def __iter__(self):
        pass

    def get_group(self, group):
        return self[group]

    def get_conn(self, group):
        return self[group]

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        if self.dtype is None:
            return np.asarray([])
        pass

    def to_df(self):
        pass

    @property
    @cache
    def shape(self) -> Shape:
        shape = {}
        for index, group in enumerate(self.groups):
            shape[group] = self.conn[index].shape
        return Shape(shape)

    @property
    def dtypes(self) -> np.dtype:
        return self.dtypes_cache

    @dtypes.setter
    def dtypes(self, v):
        self.dtypes_cache = v
