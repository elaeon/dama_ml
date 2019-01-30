import zarr
import h5py
import numpy as np
import os
import numbers
from zarr.core import Array as ZArray
from zarr.hierarchy import Group as ZGroup
from h5py.highlevel import Group as H5Group

from numcodecs import MsgPack
from ml.abc.driver import AbsDriver
from ml.utils.basic import Shape
from ml.utils.files import rm
from ml.utils.numeric_functions import max_dtype
from ml.abc.group import AbsGroup
from ml.data.groups import DaGroup
from ml.utils.logger import log_config

log = log_config(__name__)


class HDF5(AbsDriver):
    persistent = True
    ext = 'h5'
    data_tag = "data"

    def __contains__(self, item):
        return item in self.conn

    def __getitem__(self, item):
        return HDF5Group(self.conn[self.data_tag])

    def enter(self, url):
        if self.conn is None:
            self.conn = h5py.File(url, mode=self.mode)
            self.attrs = self.conn.attrs

    def exit(self):
        self.conn.close()
        self.conn = None
        self.attrs = None

    def require_dataset(self, level: str, group: str, shape: tuple, dtype: np.dtype) -> None:
        if dtype == np.dtype("O") or dtype.type == np.str_:
            dtype = h5py.special_dtype(vlen=str)

        self.conn[level].require_dataset(group, shape, dtype=dtype, chunks=True, exact=True,
                                         **self.compressor_params)

    def destroy(self, scope):
        rm(scope)

    def exists(self, scope):
        return os.path.exists(scope)

    def set_schema(self, name, dtypes):
        self.conn.require_group("metadata")
        self.require_dataset("metadata", "dtypes", (len(dtypes), 2), dtype=np.dtype('object'))
        for i, (group, dtype) in enumerate(dtypes):
            self.conn["metadata"]["dtypes"][i] = (group, dtype.str)

    def set_data_shape(self, shape):
        self.conn.require_group(self.data_tag)
        for group, dtype in self.dtypes("metadata"):
            self.require_dataset(self.data_tag, group, shape[group], dtype)

    def dtypes(self, name) -> list:
        dtypes = self.conn["metadata"]["dtypes"]
        return [(col, np.dtype(dtype)) for col, dtype in dtypes]


class Zarr(AbsDriver):
    persistent = True
    ext = "zarr"
    data_tag = "data"
    metadata_tag = "metadata"

    def __contains__(self, item):
        return item in self.conn

    @property
    def data(self):
        return ZarrGroup(self.conn[self.data_tag])

    def enter(self, url):
        if self.conn is None:
            self.conn = zarr.open(url, mode=self.mode)
            self.attrs = self.conn.attrs

    def exit(self):
        self.conn = None
        self.attrs = None

    def require_dataset(self, level: str, group: str, shape: tuple, dtype: np.dtype) -> None:
        if dtype == np.dtype("O"):
            object_codec = MsgPack()
        else:
            object_codec = None
        self.conn[level].require_dataset(group, shape, dtype=dtype, chunks=True,
                                         exact=True, object_codec=object_codec,
                                         compressor=self.compressor)

    def destroy(self, scope):
        rm(scope)

    def exists(self, scope):
        return os.path.exists(scope)

    def set_schema(self, dtypes:np.dtype):
        if self.metadata_tag in self.conn:
           log.debug("Rewriting dtypes")
        self.conn.require_group(self.metadata_tag)
        self.require_dataset(self.metadata_tag, "dtypes", (len(dtypes), 2), dtype=np.dtype('object'))
        for i, (group, (dtype, _)) in enumerate(dtypes.fields.items()):
            self.conn[self.metadata_tag]["dtypes"][i] = (group, dtype.str)

    def set_data_shape(self, shape):
        self.conn.require_group(self.data_tag)
        dtypes = self.dtypes
        if dtypes is not None:
            for group, (dtype, _) in dtypes.fields.items():
                self.require_dataset(self.data_tag, group, shape[group], dtype)

    @property
    def dtypes(self) -> np.dtype:
        if self.metadata_tag in self.conn:
            dtypes = self.conn[self.metadata_tag]["dtypes"]
            return np.dtype([(col, np.dtype(dtype)) for col, dtype in dtypes])

    def spaces(self) -> list:
        return list(self.conn.keys())


class Memory(Zarr):
    presistent = False

    def enter(self, url=None):
        if self.conn is None:
            self.conn = zarr.group()
            self.attrs = self.conn.attrs

    def exit(self):
        pass



class ZarrGroup(AbsGroup):
    def __init__(self, conn, name=None, dtypes=None, index=None, alias_map=None):
        super(ZarrGroup, self).__init__(conn, name=name, dtypes=dtypes, index=index,
                                        alias_map=alias_map)

    def __getitem__(self, item):
        if isinstance(item, str):
            if isinstance(self.conn, ZGroup):
                key = self.alias_map.get(item, item)
                group = ZarrGroup(self.conn[key], name=item, alias_map=self.alias_map.copy())
                if self.slice.stop is not None:
                    group.slice = self.slice
                else:
                    group.slice = slice(0, self.shape[key][0])
            else:
                group = ZarrGroup(self.conn, name=item, alias_map=self.alias_map.copy())
                group.slice = self.slice
            return group
        elif isinstance(item, slice):
            group = ZarrGroup(self.conn, name=self.name, alias_map=self.alias_map.copy(), dtypes=self.dtypes)
            group.slice = item
            return group.to_ndarray()
        elif isinstance(item, int):
            if item >= len(self):
                raise IndexError("index {} is out of bounds with size {}".format(item, len(self.conn)))
            group = ZarrGroup(self.conn, name=self.name, alias_map=self.alias_map.copy(),
                              dtypes=self.dtypes)
            group.slice = slice(item, item + 1)
            return group
        elif isinstance(item, list):
            from ml.data.groups import DaGroup
            if isinstance(self.conn, ZGroup):
                return DaGroup(self, chunks=(10,), from_groups=item)
            else:
                raise NotImplementedError
        elif isinstance(item, tuple):
            group = ZarrGroup(self.conn, name=self.name, alias_map=self.alias_map.copy(), dtypes=self.dtypes)
            group.slice = item # self.slice.update(item)
            return group.to_ndarray()

    def __setitem__(self, item, value):
        if hasattr(value, "groups"):
            for group in value.groups:
                self.conn[group][item] = value[group].to_ndarray()
        elif hasattr(value, 'batch'):
            #print(value.batch, item, value.batch.dtype)
            for group in value.batch.dtype.names:
                self.conn[group][item] = value.batch[group]
        elif isinstance(value, numbers.Number):
            self.conn[item] = value
        elif isinstance(value, np.ndarray):
            self.conn[item] = value
        else:
            if isinstance(item, str):
                self.conn[item] = value

    @property
    def dtypes(self) -> np.dtype:
        if isinstance(self.conn, ZGroup):
            return np.dtype([(self.inv_map.get(group, group), self.conn[group].dtype) for group in self.conn.keys()])
        elif isinstance(self.conn, ZArray):
            return np.dtype([(self.inv_map.get(self.name, self.name), self.conn.dtype)])
        else:
            return np.dtype([(self.inv_map.get(group, group), dtype)
                             for group, (dtype, _) in self.static_dtypes.fields.items()])

    @property
    def shape(self) -> Shape:
        if isinstance(self.conn, ZGroup):
            if self.slice.stop is None:
                shape = dict([(self.alias_map.get(group, group), self.conn[self.alias_map.get(group, group)].shape) for group in self.groups])
            else:
                shape = dict([(self.alias_map.get(group, group), self.conn[self.alias_map.get(group, group)][self.slice].shape) for group in self.groups])
        else:
            if self.slice.stop is None:
                shape = dict([(self.alias_map.get(group, group), self.conn.shape) for group in self.groups])
            else:
                shape = dict([(self.alias_map.get(group, group), self.conn[self.slice].shape) for group in self.groups])
        return Shape(shape)

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        #if isinstance(self.slice.idx, list):
        #    shape = [len(self.slice.idx)] + list(self.shape.to_tuple())[1:]
        #    array = np.empty(shape, dtype=self.dtype)
        #    for i, idx in enumerate(self.slice.idx):
        #        array[i] = self.conn[idx]
        #    return array

        if self.dtype is None:
            return np.asarray([])

        if isinstance(self.conn, ZArray):
            array = self.conn[self.slice]
        else:
            if len(self.groups) == 1:
                array = self.conn[self.groups[0]][self.slice]
            else:
                array = np.empty(self.shape.to_tuple(), dtype=self.dtype)
                for i, group in enumerate(self.groups):
                    array[:, i] = self.conn[group][self.slice]
                return array

        if dtype is not None and self.dtype != dtype:
            return array.astype(dtype)
        else:
            return array

    def to_dagroup(self, chunks=None) -> DaGroup:
        return DaGroup(self, chunks=chunks)

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
            init = 0
            end = data.batch_size
            for e in data:
                stc_arr[init:end] = e
                init = end
                end += data.batch_size
        else:
            for i, group in enumerate(self.groups):
                init = 0
                end = data.batch_size
                for e in data:
                    stc_arr[group][init:end] = e[:, i]
                    init = end
                    end += data.batch_size
        return pd.DataFrame(stc_arr, index=np.arange(0, stc_arr.shape[0]), columns=columns)


class HDF5Group(object):
    def __init__(self, conn, name=None):
        self.conn = conn
        self.name = name

    def __getitem__(self, item):
        if isinstance(self.conn, H5Group):
            return HDF5Group(self.conn[item], name=item)
        else:
            return HDF5Group(self.conn[item], name=self.name)

    def __setitem__(self, item, value):
        if hasattr(value, "groups"):
            for group in value.groups:
                self.conn[group][item] = value[group]

    def __len__(self):
        return self.shape.to_tuple()[0]

    @property
    def dtypes(self):
        if isinstance(self.conn, H5Group):
            return [(key, self.conn[key].dtype) for key in self.conn.keys()]
        else:
            return [(self.name, self.conn.dtype)]

    @property
    def dtype(self) -> np.dtype:
        return max_dtype(self.dtypes)

    @property
    def shape(self) -> Shape:
        shape = dict([(group, self.conn.shape) for group, _ in self.dtypes])
        return Shape(shape)

    def compute(self, chunksize=(258,)):
        return self.conn