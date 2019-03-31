from dama.utils.miscellaneous import filter_dtypes
from dama.utils.decorators import cache
from dama.fmtypes import DEFAUL_GROUP_NAME
from dama.exceptions import NotChunksFound
from dama.utils.core import Shape, Chunks
from dama.abc.conn import AbsConn
from dama.abc.driver import AbsDriver
from collections import OrderedDict
from numbers import Number
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import numpy as np


__all__ = ['GroupManager', 'ListConn', 'DaskDfConn']


class GroupManager(AbsConn):
    def __init__(self):
        super(GroupManager, self).__init__(OrderedDict(), None)
        self.counter = 0

    def __add__(self, other: 'GroupManager') -> 'GroupManager':
        if isinstance(other, Number) and other == 0:
            return self
        groups = GroupManager()
        groups.update(self)
        groups.update(other)
        return groups

    def __radd__(self, other):
        return self.__add__(other)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.manager_from_groups(self.groups, item)
        elif isinstance(item, str):
            dict_conn = GroupManager()
            dict_conn[item] = self.conn[item]
            return dict_conn
        elif isinstance(item, int):
            return self.manager_from_groups(self.groups, item)
        elif isinstance(item, list):
            dict_conn = GroupManager()
            for group in item:
                dict_conn[group] = self.conn[group]
            return dict_conn
        elif isinstance(item, np.ndarray) and item.dtype == np.dtype(int):
            return self.manager_from_groups(self.groups, item)
        elif isinstance(item, da.Array):
            index = [i for i, is_true in enumerate(item.compute()) if is_true]  # fixme generalize masked data
            return self.manager_from_groups(self.groups, index)

    def __setitem__(self, key, value):
        self.conn[key] = value

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        try:
            elem = self._iterator(self.counter)
            self.counter += 1
        except IndexError:
            raise StopIteration
        else:
            return elem

    def __eq__(self, other):
        if len(self.groups) == 1:
            return self.conn[self.groups[0]] == other
        else:
            raise NotImplementedError

    def update(self, group_manager: 'GroupManager'):
        self.conn.update(group_manager.conn)

    def _iterator(self, counter):
        elem = self[counter]
        return elem

    def rename_group(self, old_key, new_key):
        for _ in range(len(self.conn)):
            k, v = self.conn.popitem(False)
            self.conn[new_key if old_key == k else k] = v

    @classmethod
    def convert(cls, groups_items, chunks: Chunks) -> 'GroupManager':
        if chunks is None:
            raise NotChunksFound
        groups = cls()
        if isinstance(groups_items, dict):
            groups_items = groups_items.items()
        for group, data in groups_items:
            lock = False
            groups[group] = da.from_array(data, chunks=chunks[group], lock=lock)
        return groups

    def manager_from_groups(self, groups, item) -> 'GroupManager':
        dict_conn = GroupManager()
        for group in groups:
            dict_conn[group] = self.conn[group][item]
        return dict_conn

    def to_dd(self) -> dd.DataFrame:
        dfs = []
        for group in self.groups:
            df = dd.from_dask_array(self.conn[group], columns=[group])
            dfs.append(df)
        return dd.concat(dfs, axis=1)

    @staticmethod
    def concat(da_groups, axis=0) -> 'GroupManager':
        if axis == 0:
            all_groups = [da_group.groups for da_group in da_groups]
            da_group_dict = GroupManager()
            intersection_groups = set(all_groups[0])
            for group in all_groups[1:]:
                intersection_groups = intersection_groups.intersection(set(group))

            if len(intersection_groups) > 0:
                # to maintain connexions order
                groups = [group for group in all_groups[0] if group in intersection_groups]
                for group in groups:
                    da_arrays = [da_group.conn[group] for da_group in da_groups]
                    da_array_c = da.concatenate(da_arrays, axis=axis)
                    da_group_dict[group] = da_array_c
                return da_group_dict
            else:
                return sum(da_groups)
        else:
            raise NotImplementedError

    @staticmethod
    def from_da(da_array: da.Array, group_name: str = DEFAUL_GROUP_NAME) -> 'GroupManager':
        dagroup_dict = GroupManager()
        dagroup_dict[group_name] = da.Array(da_array.dask, chunks=da_array.chunks,
                                            dtype=da_array.dtype, name=da_array.name)
        return dagroup_dict

    @property
    def chunksize(self) -> Chunks:
        chunks = Chunks()
        for group in self.groups:
            chunks[group] = self.conn[group].chunksize
        return chunks

    def to_ndarray(self, dtype: np.dtype = None) -> np.ndarray:
        self.attrs["dtype"] = dtype
        if len(self.groups) == 1:
            computed_array = self.conn[self.groups[0]].compute(dtype=self.dtype)
            if dtype is not None and dtype != self.dtype:
                return computed_array.astype(dtype)
            return computed_array
        else:
            shape = self.shape.to_tuple()
            if dtype is None:
                dtype = self.dtype
            data = np.empty(shape, dtype=dtype)
            total_cols = 0
            for group in self.groups:
                try:
                    num_cols = self.shape[group][1]
                    slice_grp = (slice(None, None), slice(total_cols, total_cols + num_cols))
                except IndexError:
                    num_cols = 1
                    slice_grp = (slice(None, None), total_cols)
                total_cols += num_cols
                data[slice_grp] = self.conn[group].compute(dtype=dtype)
            return data

    def to_stc_array(self) -> np.ndarray:
        if len(self.groups) == 1:
            computed_array = self.conn[self.groups[0]].compute(dtype=self.dtype)
            return computed_array
        else:
            shape = self.shape
            if len(shape) > 1 and len(self.groups) < shape[1]:
                dtypes = np.dtype([("c{}".format(i), self.dtype) for i in range(shape[1])])
            else:
                dtypes = self.dtypes

            shape = self.shape.to_tuple()
            data = np.empty(shape[0], dtype=dtypes)
            for group in self.groups:
                data[group] = self.conn[group].compute(dtype=self.dtype)
            return data

    def to_df(self) -> pd.DataFrame:
        stc_arr = self.to_stc_array()
        return pd.DataFrame(stc_arr, index=np.arange(0, stc_arr.shape[0]), columns=self.groups)

    @property
    def dtypes(self):
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.keys()])

    @dtypes.setter
    def dtypes(self, v):
        pass

    @property
    def shape(self) -> Shape:
        shape = OrderedDict((group, data.shape) for group, data in self.conn.items())
        return Shape(shape)

    def store(self, driver: AbsDriver):
        for group in self.groups:
            self.conn[group].store(driver[group])

    @property
    def da(self):
        if len(self.groups) == 1:
            return self.conn[self.groups[0]]
        else:
            raise NotImplementedError


class DaskDfConn(dd.DataFrame, AbsConn):
    def __init__(self, conn, dtypes=None):
        dd.DataFrame.__init__(self, conn.dask, conn._name, conn._meta, conn.divisions)
        AbsConn.__init__(self, conn, dtypes)
        # super(DaskDfConn, self).__init__(conn.dask, conn._name, conn._meta, conn.divisions)

    # def __getitem__(self, item):
    #    dd = self.conn[item]
    #    return DaskDfConn(dd, dd.dtypes)

    # def __setitem__(self, key, value):
    #    self.conn[key] = value

    def __getattr__(self, key):
        if key in self.columns:
            return self[key]
        else:
            return self.__getattribute__(key)

    def base_cls(self):
        return self.__class__.__bases__[0]

    def cast(self, value):
        return value

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @property
    @cache
    def shape(self) -> Shape:
        length = self.conn.shape[0].compute()
        shape = dict([(group, (length,)) for group in self.groups])
        return Shape(shape)

    def to_ndarray(self, dtype: np.dtype = None) -> np.ndarray:
        return self.conn.compute(dtype=dtype).values

    def to_df(self):
        return self.conn.compute()

    def store(self, driver: AbsDriver):
        for group in self.groups:
            self.conn[group].store(driver[group])

    @property
    def chunksize(self) -> Chunks:
        chunks = Chunks()
        chunksize = self.conn.chunksize
        for group in self.groups:
            chunks[group] = chunksize
        return chunks

    @property
    def dtypes(self):
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.columns])

    # @dtypes.setter
    # def dtypes(self, v):
    #     pass


class ListConn(AbsConn):
    def __getitem__(self, item):
        if isinstance(item, str):
            dtypes = filter_dtypes(item, self.dtypes)
            i = self.groups.index(item)
            return ListConn([self.conn[i]], dtypes=dtypes)
        else:
            return self.conn[item]

    def __setitem__(self, key, value):
        try:
            self.conn[key] = value
        except IndexError:
            if len(self.conn) == 0:
                self.conn.append(None)

            for _ in range(abs(key - len(self.conn))):
                self.conn.append(None)
            self.conn[key] = value

    def to_ndarray(self, dtype: np.dtype = None):
        if len(self.dtypes) == 1:
            return self.conn
        else:
            array = np.empty(self.shape.to_tuple(), dtype=self.dtype)
            for i, elem in enumerate(self.conn):
                array[:, i] = elem
        return array

    @property
    @cache
    def shape(self):
        shape = {}
        for index, group in enumerate(self.groups):
            shape[group] = self.conn[index].shape
        return Shape(shape)

    def store(self, driver: AbsDriver):
        pass

    def chunksize(self) -> Chunks:
        pass
