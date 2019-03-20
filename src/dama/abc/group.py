from abc import ABC, abstractmethod
from collections import OrderedDict
from dama.utils.numeric_functions import max_dtype
from dama.utils.core import Shape, Chunks
from dama.fmtypes import Slice, DEFAUL_GROUP_NAME
from dama.exceptions import NotChunksFound
from dama.utils.decorators import cache
from dama.abc.data import AbsData
from numbers import Number
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd


class DaGroupDict(OrderedDict):
    def __init__(self, *args, map_rename=None, **kwargs):
        super(DaGroupDict, self).__init__(*args, **kwargs)
        self.attrs = Attrs()
        #if map_rename is None:
        #    self.map_rename = {}
        #else:
        #    self.map_rename = map_rename

    def __add__(self, other: 'DaGroupDict') -> 'DaGroupDict':
        if isinstance(other, Number) and other == 0:
            return self
        groups = DaGroupDict()
        groups.update(self)
        groups.update(other)
        return groups

    def __radd__(self, other):
        return self.__add__(other)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.set_values(self.groups, item)
        elif isinstance(item, str):
            dict_conn = DaGroupDict()
            dict_conn[item] = super(DaGroupDict, self).__getitem__(item)
            return dict_conn
        elif isinstance(item, int):
            return self.set_values(self.groups, item)
        elif isinstance(item, list):
            dict_conn = DaGroupDict()
            for group in item:
                dict_conn[group] =  super(DaGroupDict, self).__getitem__(group)
            return dict_conn
        elif isinstance(item, np.ndarray) and item.dtype == np.dtype(int):
            return self.set_values(self.groups, item)
        elif isinstance(item, da.Array):
            index = [i for i, is_true in enumerate(item.compute()) if is_true]  # fixme generalize masked data
            return self.set_values(self.groups, index)

    def rename_group(self, key, new_key):
        #self.map_rename[new_key] = key
        self[new_key] = super(DaGroupDict, self).__getitem__(key)
        del self[key]

    #def get_oldname(self, name) -> str:
    #    return self.map_rename.get(name, name)

    @property
    def groups(self):
        return tuple(self.keys())

    @classmethod
    def convert(cls, groups_items, chunks: Chunks) -> 'DaGroupDict':
        if chunks is None:
            raise NotChunksFound
        groups = cls()
        if isinstance(groups_items, dict):
            groups_items = groups_items.items()
        for group, data in groups_items:
            lock = False
            groups[group] = da.from_array(data, chunks=chunks[group], lock=lock)
        return groups

    def set_values(self, groups, item) -> 'DaGroupDict':
        dict_conn = DaGroupDict()
        for group in groups:
            dict_conn[group] = super(DaGroupDict, self).__getitem__(group)[item]
        return dict_conn

    def to_dd(self) -> dd.DataFrame:
        dfs = []
        for group in self.groups:
            df = dd.from_dask_array(self[group], columns=[group])
            dfs.append(df)
        return dd.concat(dfs, axis=1)

    @staticmethod
    def concat(da_groups, axis=0) -> 'DaGroupDict':
        if axis == 0:
            all_groups = [da_group.groups for da_group in da_groups]
            da_group_dict = DaGroupDict()
            intersection_groups = set(all_groups[0])
            for group in all_groups[1:]:
                intersection_groups = intersection_groups.intersection(set(group))

            if len(intersection_groups) > 0:
                groups = [group for group in all_groups[0] if group in intersection_groups]  # to maintain groups order
                for group in groups:
                    da_arrays = [da_group[group].darray for da_group in da_groups]
                    da_array_c = da.concatenate(da_arrays, axis=axis)
                    da_group_dict[group] = da_array_c
                return da_group_dict
            else:
                return sum(da_groups)
        else:
            raise NotImplementedError

    @staticmethod
    def from_da(da_array: da.Array, group_name: str = DEFAUL_GROUP_NAME) -> 'DaGroupDict':
        dagroup_dict = DaGroupDict()
        dagroup_dict[group_name] = da.Array(da_array.dask, chunks=da_array.chunks,
                                            dtype=da_array.dtype, name=da_array.name)
        return DaGroupDict(dagroup_dict=dagroup_dict)

    @property
    def chunksize(self) -> Chunks:
        chunks = Chunks()
        for group in self.groups:
            chunks[group] = self[group].chunksize
        return chunks

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        self.attrs["dtype"] = dtype
        if len(self.groups) == 1:
            computed_array = super(DaGroupDict, self).__getitem__(self.groups[0]).compute(dtype=self.dtype)
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
        return np.dtype([(group, super(DaGroupDict, self).__getitem__(group).dtype) for group in self.keys()])

    @property
    def dtype(self) -> np.dtype:
        return max_dtype(self.dtypes)


class AbsGroup(ABC):
    inblock = None
    dtypes = None
    conn = None

    @abstractmethod
    def get_group(self, group):
        return NotImplemented

    @abstractmethod
    def get_conn(self, group):
        return NotImplemented

    @property
    def groups(self) -> tuple:
        return self.dtypes.names

    @property
    def dtype(self) -> np.dtype:
        return max_dtype(self.dtypes)

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
    def shape(self) -> Shape:
        return NotImplemented

    def set(self, item, value):
        return NotImplemented


class AbsDaskGroup(AbsGroup):
    def __init__(self, conn, dtypes, chunks: Chunks=None):
        self.conn = conn
        self.attrs = Attrs()
        self.dtypes = dtypes

    @abstractmethod
    def get_group(self, group):
        return NotImplemented

    @abstractmethod
    def get_conn(self, group):
        return NotImplemented

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
        shape = OrderedDict([(group, (length,)) for group in self.groups])
        return Shape(shape)

    def set(self, item, value):
        pass


class AbsDictGroup(AbsGroup):
    inblock = None

    def __init__(self, conn, dtypes, chunks: Chunks=None):
        self.conn = conn
        # self.attrs = Attrs()
        self.dtypes = dtypes
        groups = [(group, self.get_conn(group)) for group in self.groups]
        self.manager = DaGroupDict.convert(groups, chunks=chunks)

    @abstractmethod
    def __getitem__(self, item):
        return NotImplemented

    @abstractmethod
    def __setitem__(self, item, value):
        return NotImplemented

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

    def _iterator(self, counter):
        elem = self[counter]
        return elem

    def __len__(self):
        return self.shape.to_tuple()[0]

    def __repr__(self):
        return "{} {}".format(self.cls_name(), self.shape)

    def dtypes_from_groups(self, groups) -> np.dtype:
        if not isinstance(groups, list) and not isinstance(groups, tuple):
            groups = [groups]
        return np.dtype([(group, dtype) for group, (dtype, _) in self.dtypes.fields.items() if group in groups])

    @abstractmethod
    def get_group(self, group):
        return NotImplemented

    @abstractmethod
    def get_conn(self, group):
        return NotImplemented

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
    def shape(self) -> Shape:
        shape = OrderedDict((group, data.shape) for group, data in self.manager.items())
        return Shape(shape)

    def set(self, item, value):
        #from dama.groups.core import DaGroup
        if self.inblock is True:
            self[item] = value
        else:
            if type(value) == AbsDictGroup:
                for group in value.groups:
                    group = value.conn.get_oldname(group)
                    self.conn[group][item] = value[group].to_ndarray()
            elif type(value) == Slice:
                for group in value.batch.groups:
                    group = value.batch.conn.get_oldname(group)
                    self.conn[group][item] = value.batch[group].to_ndarray()
            elif isinstance(value, Number):
                self.conn[item] = value
            elif isinstance(value, np.ndarray):
                self.conn[item] = value
            else:
                if isinstance(item, str):
                    self.conn[item] = value

    def store(self, dataset: AbsData):
        self.write_to_group = dataset.driver.absgroup
        if self.write_to_group.inblock is True:
            from dama.data.it import Iterator
            data = Iterator(self).batchs(chunks=self.manager.chunksize)
            dataset.batchs_writer(data)
        else:
            for group in self.groups:
                self.conn[group].store(dataset.driver.absgroup.get_conn(group))

    def items(self):
        return [(group, self.conn[group]) for group in self.groups]


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Attrs(dict, metaclass=Singleton):
    pass


class AbsGroupX(AbsDictGroup):
    __slots__ = ['conn', 'counter', 'attrs']

    #def __init__(self, conn, dtypes, chunks: Chunks):
    #    super(AbsGroupX, self).__init__(conn, dtypes, chunks)
    #    self.counter = 0

    #@abstractmethod
    #def __getitem__(self, item):
    #    return NotImplemented

    #@abstractmethod
    #def __setitem__(self, item, value):
    #    return NotImplemented

    #def __iter__(self):
    #    self.counter = 0
    #    return self

    #def __next__(self):
    #    try:
    #        elem = self._iterator(self.counter)
    #        self.counter += 1
    #    except IndexError:
    #        raise StopIteration
    #    else:
    #        return elem

    #def _iterator(self, counter):
    #    elem = self[counter]
    #    return elem

    #def __len__(self):
    #    return self.shape.to_tuple()[0]

    #def __repr__(self):
    #    return "{} {}".format(self.cls_name(), self.shape)

    #def get_group(self, group):
    #    return self[group]

    #def get_conn(self, group):
    #    return self[group]

    #@property
    #@abstractmethod
    #def shape(self) -> Shape:
    #    return NotImplemented

    #@abstractmethod
    #def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
    #    return NotImplemented

    #@abstractmethod
    #def to_df(self) -> pd.DataFrame:
    #    return NotImplemented

    #def items(self):
    #    return [(group, self.conn[group]) for group in self.groups]