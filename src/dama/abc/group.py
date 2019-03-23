from abc import ABC, abstractmethod
from collections import OrderedDict
from dama.utils.numeric_functions import max_dtype
from dama.utils.core import Shape, Chunks
from dama.fmtypes import DEFAUL_GROUP_NAME
from dama.exceptions import NotChunksFound
from dama.utils.decorators import cache
from numbers import Number
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd


class Manager:
    pass


class DaGroupDict(OrderedDict, Manager):
    def __init__(self, *args, **kwargs):
        super(DaGroupDict, self).__init__(*args, **kwargs)
        self.attrs = Attrs()
        self.counter = 0

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
                dict_conn[group] = super(DaGroupDict, self).__getitem__(group)
            return dict_conn
        elif isinstance(item, np.ndarray) and item.dtype == np.dtype(int):
            return self.set_values(self.groups, item)
        elif isinstance(item, da.Array):
            index = [i for i, is_true in enumerate(item.compute()) if is_true]  # fixme generalize masked data
            return self.set_values(self.groups, index)

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
            return (super(DaGroupDict, self).__getitem__(self.groups[0]) == other)
        else:
            raise NotImplementedError

    @property
    def size(self):
        return self.shape[0]

    def update(self, *args, **kwargs):
        if not args:
            raise TypeError("descriptor 'update' of 'MutableMapping' object "
                            "needs an argument")
        other = args[0]
        for key, value in other.items():
            self[key] = value

    def getitem(self, group):
        return super(DaGroupDict, self).__getitem__(group)

    def popitem(self, last: bool = ...):
        if not self:
            raise KeyError('dictionary is empty')
        key = next(reversed(self) if last else iter(self.keys()))
        value = self.pop(key)
        return key, value

    def pop(self, key):
        value = super(DaGroupDict, self).__getitem__(key)
        del self[key]
        return value

    def _iterator(self, counter):
        elem = self[counter]
        return elem

    def rename_group(self, old_key, new_key):
        for _ in range(len(self)):
            k, v = self.popitem(False)
            self[new_key if old_key == k else k] = v

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
                    da_arrays = [da_group.getitem(group) for da_group in da_groups]
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
        return DaGroupDict(dagroup_dict)

    @property
    def chunksize(self) -> Chunks:
        chunks = Chunks()
        for group in self.groups:
            chunks[group] = self.getitem(group).chunksize
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
                data[slice_grp] = super(DaGroupDict, self).__getitem__(group).compute(dtype=dtype)
            return data

    def to_stc_array(self) -> np.ndarray:
        if len(self.groups) == 1:
            computed_array = super(DaGroupDict, self).__getitem__(self.groups[0]).compute(dtype=self.dtype)
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
                data[group] = super(DaGroupDict, self).__getitem__(group).compute(dtype=self.dtype)
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

    @property
    def shape(self) -> Shape:
        shape = OrderedDict((group, data.shape) for group, data in self.items())
        return Shape(shape)


class AbsConn(ABC):
    inblock = None
    dtypes = None
    conn = None

    def __init__(self, conn, dtypes):
        self.conn = conn
        self.attrs = Attrs()
        self.dtypes = dtypes

    #@abstractmethod
    #def get_group(self, group):
    #    return NotImplemented

    #@abstractmethod
    #def get_conn(self, group):
    #    return NotImplemented

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


class AbsDaskGroup(AbsConn):
    def __init__(self, conn, dtypes, chunks: Chunks=None):
        super(AbsDaskGroup, self).__init__(conn, dtypes)

    #@abstractmethod
    #def get_group(self, group):
    #    return NotImplemented

    #@abstractmethod
    #def get_conn(self, group):
    #    return NotImplemented

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


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Attrs(dict, metaclass=Singleton):
    pass



class AbsGroupX:
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