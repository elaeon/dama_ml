from dama.abc.group import AbsConn
from dama.utils.core import Shape
import numpy as np
from collections import OrderedDict
from dama.utils.decorators import cache


class DaGroup:
    def __init__(self, abs_source: AbsConn=None, write_to_group=None, chunks=None):
        pass
#    def __init__(self, abs_source: AbsGroup=None, write_to_group=None):
#        self.abs_source = abs_source
#        self.write_to_group = write_to_group
        #if isinstance(dagroup_dict, DaGroupDict):
        #    reader_conn = dagroup_dict
#        if isinstance(abs_source, AbsGroup):
            #if is_dask_collection(abs_source.conn):
#            reader_conn = abs_source
            #elif isinstance(abs_source.conn, DaGroupDict):
            #    reader_conn = dagroup_dict
            #else:
            #    raise Exception
            #else:
            #    groups = [(group, abs_source.get_conn(group)) for group in abs_source.groups]
            #    reader_conn = DaGroup.convert(groups, chunks=chunks)
#        else:
#            raise NotImplementedError("Type {} is not supported".format(type(abs_source)))
#        super(DaGroup, self).__init__(reader_conn, dtypes=reader_conn.dtypes)

    #@staticmethod
    #def convert(groups_items, chunks: Chunks) -> DaGroupDict:
    #    if chunks is None:
    #        raise NotChunksFound
    #    groups = DaGroupDict()
    #    if isinstance(groups_items, dict):
    #        groups_items = groups_items.items()
    #    for group, data in groups_items:
    #        lock = False
    #        groups[group] = da.from_array(data, chunks=chunks[group], lock=lock)
    #    return groups

    #def sample(self, index):
    #    return self.set_values(self.groups, index)

    #def set_values(self, groups, item) -> 'DaGroup':
    #    dict_conn = DaGroupDict(map_rename=self.conn.map_rename)
    #    for group in groups:
    #        dict_conn[group] = self.conn[group][item]
    #    return DaGroup(dagroup_dict=dict_conn, write_to_group=self.write_to_group, abs_source=self.abs_source)

    #def __getitem__(self, item) -> 'DaGroup':
    #    pass
        #if isinstance(item, slice):
        #    return self.set_values(self.groups, item)
        #elif isinstance(item, str):
        #    dict_conn = DaGroupDict(map_rename=self.conn.map_rename)
        #    dict_conn[item] = self.conn[item]
        #    if self.write_to_group is None:
        #        return DaGroup(dagroup_dict=dict_conn, abs_source=self.abs_source)
        #    else:
        #        return DaGroup(dagroup_dict=dict_conn, abs_source=self.abs_source,
        #                       write_to_group=self.write_to_group.get_group(self.conn.get_oldname(item)))
        #elif isinstance(item, int):
        #    return self.set_values(self.groups, item)
        #elif isinstance(item, list):
        #    dict_conn = DaGroupDict(map_rename=self.conn.map_rename)
        #    for group in item:
        #        dict_conn[group] = self.conn[group]
        #    return DaGroup(dagroup_dict=dict_conn, write_to_group=self.write_to_group, abs_source=self.abs_source)
        #elif isinstance(item, np.ndarray) and item.dtype == np.dtype(int):
        #    return self.sample(item)
        #elif isinstance(item, da.Array):
        #    index = [i for i, is_true in enumerate(item.compute()) if is_true]  # fixme generalize masked data
        #    return self.sample(index)

    #def __setitem__(self, item, value):
    #    self.write_to_group.set(item, value)

    #def set(self, item, value):
    #    if isinstance(self.conn, DaGroupDict):
    #        self.conn[item] = value
    #    else:
    #        raise NotImplementedError

    #def __add__(self, other: 'DaGroup') -> 'DaGroup':
    #    if isinstance(other, Number) and other == 0:
    #        return self
    #    groups = DaGroupDict()
    #    groups.update(self.conn)
    #    groups.update(other.conn)
    #    return DaGroup(dagroup_dict=groups, write_to_group=self.write_to_group, abs_source=self.abs_source)

    #def __radd__(self, other):
    #    return self.__add__(other)

    #def __eq__(self, other):
    #    if len(self.groups) == 1:
    #        return self.conn[self.groups[0]] == other
    #    else:
    #        raise NotImplementedError

    #def to_dd(self) -> dd.DataFrame:
    #    return self.abs_source.to_dd()
        #dfs = []
        #for group in self.groups:
        #    df = dd.from_dask_array(self.conn[group], columns=[group])
        #    dfs.append(df)
        #return dd.concat(dfs, axis=1)

    #@staticmethod
    #def concat(self, da_groups, axis=0) -> 'DaGroup':
    #    return self.abs_source.concat(da_groups, axis=axis)
        #if axis == 0:
        #    all_groups = [da_group.groups for da_group in da_groups]
        #    da_group_dict = DaGroupDict()
        #    intersection_groups = set(all_groups[0])
        #    for group in all_groups[1:]:
        #        intersection_groups = intersection_groups.intersection(set(group))

        #    if len(intersection_groups) > 0:
        #        groups = [group for group in all_groups[0] if group in intersection_groups]  # to maintain groups order
        #        for group in groups:
        #            da_arrays = [da_group[group].darray for da_group in da_groups]
        #            da_array_c = da.concatenate(da_arrays, axis=axis)
        #            da_group_dict[group] = da_array_c
        #        return DaGroup(dagroup_dict=da_group_dict)
        #    else:
        #        return sum(da_groups)
        #else:
        #    raise NotImplementedError

    #@property
    #def shape(self) -> Shape:
    #    shape = OrderedDict((group, data.shape) for group, data in self.conn.items())
    #    return Shape(shape)

    #@property
    #def chunksize(self) -> Chunks:
    #    chunks = Chunks()
    #    for group in self.groups:
    #        chunks[group] = self.conn[group].chunksize
    #    return chunks

    #@property
    #def darray(self):
    #    if len(self.groups) == 1:
    #        return self.conn[self.groups[0]]
    #    else:
    #        raise NotImplementedError("I can't return a dask array with two groups.")

    #def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
    #    if self.abs_source is not None:
    #        self.abs_source.attrs["dtype"] = dtype
    #    if len(self.groups) == 1:
    #        computed_array = self.conn[self.groups[0]].compute(dtype=self.dtype)
    #        if dtype is not None and dtype != self.dtype:
    #            return computed_array.astype(dtype)
    #        return computed_array
    #    else:
    #        shape = self.shape.to_tuple()
    #        if dtype is None:
    #            dtype = self.dtype
    #        data = np.empty(shape, dtype=dtype)
    #        total_cols = 0
    #        for group in self.groups:
    #            try:
    #                num_cols = self.shape[group][1]
    #                slice_grp = (slice(None, None), slice(total_cols, total_cols + num_cols))
    #            except IndexError:
    #                num_cols = 1
    #                slice_grp = (slice(None, None), total_cols)
    #            total_cols += num_cols
    #            data[slice_grp] = self.conn[group].compute(dtype=dtype)
    #        return data

    #def to_stc_array(self) -> np.ndarray:
    #    if len(self.groups) == 1:
    #        computed_array = self.conn[self.groups[0]].compute(dtype=self.dtype)
    #        return computed_array
    #    else:
    #        shape = self.shape
    #        if len(shape) > 1 and len(self.groups) < shape[1]:
    #            dtypes = np.dtype([("c{}".format(i), self.dtype) for i in range(shape[1])])
    #        else:
    #            dtypes = self.dtypes

    #        shape = self.shape.to_tuple()
    #        data = np.empty(shape[0], dtype=dtypes)
    #        for group in self.groups:
    #            data[group] = self.conn[group].compute(dtype=self.dtype)
    #        return data

    #def to_df(self) -> pd.DataFrame:
    #    stc_arr = self.to_stc_array()
    #    return pd.DataFrame(stc_arr, index=np.arange(0, stc_arr.shape[0]), columns=self.groups)

    #def rename_group(self, old_name, new_name):
    #    self.conn = self.conn.rename(old_name, new_name)

    #def store(self, dataset: AbsData):
    #    self.write_to_group = dataset.driver.absgroup
    #    if self.write_to_group.inblock is True:
    #        from dama.data.it import Iterator
    #        data = Iterator(self).batchs(chunks=self.chunksize)
    #        dataset.batchs_writer(data)
    #    else:
    #        for group in self.groups:
    #            self.conn[group].store(dataset.driver.absgroup.get_conn(group))

    #@staticmethod
    #def from_da(da_array: da.Array, group_name: str = DEFAUL_GROUP_NAME):
    #    dagroup_dict = DaGroupDict()
    #    dagroup_dict[group_name] = da.Array(da_array.dask, chunks=da_array.chunks,
    #                                         dtype=da_array.dtype, name=da_array.name)
    #    return DaGroup(dagroup_dict=dagroup_dict)


class StcArrayGroup(AbsConn):
    inblock = False

    def get_group(self, group) -> AbsConn:
        dtypes = self.dtypes_from_groups(group)
        return StcArrayGroup(self.conn[group], dtypes)

    def get_conn(self, group):
        return self.conn[group]

    @staticmethod
    def fit_shape(shape: Shape) -> Shape:
        _shape = OrderedDict()
        if len(shape.groups()) == 1:
            key = list(shape.groups())[0]
            _shape[key] = shape[key]
        else:
            for group, shape_tuple in shape.items():
                if len(shape_tuple) == 2:  # and shape_tuple[1] == 1:
                    _shape[group] = tuple(shape_tuple[:1])
                elif len(shape_tuple) == 1:
                    _shape[group] = shape_tuple
                else:
                    raise Exception
        return Shape(_shape)


class TupleGroup(AbsConn):
    inblock = False

    def __init__(self, conn, dtypes=None):
        super(TupleGroup, self).__init__(conn, dtypes)

    def __getitem__(self, item):
        if isinstance(item, str):
            if isinstance(self.conn, tuple):
                dtypes = self.dtypes_from_groups(item)
                index = 0
                for i, group in enumerate(self.groups):
                    if item == group:
                        index = i
                        break
                return TupleGroup(self.conn[index], dtypes=dtypes)
            else:
                raise NotImplementedError
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

    def to_df(self):
        pass

    @property
    @cache
    def shape(self) -> Shape:
        shape = OrderedDict()
        for index, group in enumerate(self.groups):
            shape[group] = self.conn[index].shape
        return Shape(shape)
