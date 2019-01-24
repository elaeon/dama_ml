import zarr
import h5py
import numpy as np
import os
from zarr.core import Array as ZArray
from zarr.hierarchy import Group as ZGroup
from h5py.highlevel import Group as H5Group

from numcodecs import MsgPack
from ml.abc.driver import AbsDriver
from ml.utils.basic import Shape
from ml.utils.files import rm
from ml.utils.numeric_functions import max_dtype


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

    def __contains__(self, item):
        return item in self.conn

    def __getitem__(self, item):
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


class Memory(Zarr):
    presistent = False

    def enter(self, url=None):
        if self.conn is None:
            self.conn = zarr.group()
            self.attrs = self.conn.attrs

    def exit(self):
        pass


class Empty(object):
    def __init__(self, dtypes):
        self.dtypes = dtypes

    def __len__(self):
        return 0

    def compute(self):
        return None


class ZarrGroup(object):
    def __init__(self, conn, name=None, end_node=False, dtypes=None, index=None, alias_map=None):
        self.conn = conn
        self.name = name
        self.end_node = end_node
        self.static_dtypes = dtypes
        self.index = index
        if alias_map is None:
            self.alias_map = {}
            self.inv_map = {}
        else:
            self.alias_map = alias_map
            self.inv_map = {value: key for key, value in self.alias_map.items()}

    def set_alias(self, name, alias):
        self.alias_map[alias] = name
        self.inv_map[name] = alias

    def __getitem__(self, item):
        if isinstance(self.conn, ZGroup):
            return ZarrGroup(self.conn[self.alias_map.get(item, item)], name=item, alias_map=self.alias_map.copy())
        elif isinstance(self.conn, ZArray) and not self.end_node:
            return ZarrGroup(self.conn[item], name=self.name, end_node=True, alias_map=self.alias_map.copy(),
                             dtypes=self.dtypes)
        elif self.end_node:
            if isinstance(item, int) and hasattr(self.conn, 'len') and item >= len(self.conn):
                raise IndexError("index {} is out of bounds with size {}".format(item, len(self.conn)))
            elif isinstance(item, int):
                return ZarrGroup(self.conn[item], name=self.name, end_node=True, dtypes=self.dtypes,
                                 alias_map=self.alias_map.copy())
            elif isinstance(item, int) and item < len(self.conn):
                return ZarrGroup(self.conn[item], name=self.name, end_node=True, dtypes=self.dtypes,
                                 alias_map=self.alias_map.copy())
            elif isinstance(item, slice):
                return ZarrGroup(self.conn[item], name=self.name, end_node=True, dtypes=self.dtypes,
                                 alias_map=self.alias_map.copy())
            elif isinstance(item, list) or isinstance(item, np.ndarray):
                return ZarrGroup(self.conn, name=self.name, end_node=True, dtypes=self.dtypes, index=item,
                                 alias_map=self.alias_map.copy())
            elif isinstance(item, tuple):
                return ZarrGroup(self.conn[item], name=self.name, end_node=True, dtypes=self.dtypes,
                                 alias_map=self.alias_map.copy())
            else:
                return Empty(dtypes=self.dtypes)


    def __setitem__(self, item, value):
        if hasattr(value, "groups"):
            for group in value.groups:
                self.conn[group][item] = value[group]
        else:
            group = list(value.keys())[0]
            self.conn[group][item] = value[group]

    def __len__(self):
        return self.shape.to_tuple()[0]

    @property
    def dtypes(self) -> list:
        if isinstance(self.conn, ZGroup):
            return [(self.inv_map.get(key, key), self.conn[key].dtype) for key in self.conn.keys()]
        elif hasattr(self.conn, 'dtype'):
            return [(self.inv_map.get(self.name, self.name), self.conn.dtype)]
        else:
            return [(self.inv_map.get(key, key), dtype) for key, dtype in self.static_dtypes]

    @property
    def dtype(self) -> np.dtype:
        return max_dtype(self.dtypes)

    @property
    def shape(self) -> Shape:
        shape = dict([(group, self.conn.shape) for group, _ in self.dtypes])
        return Shape(shape)

    def compute(self, chunksize=(258,)):
        if self.index is not None:
            shape = [len(self.index)] + list(self.shape.to_tuple())[1:]
            array = np.empty(shape, dtype=self.dtype)
            for i, idx in enumerate(self.index):
                array[i] = self.conn[idx]
            return array
        elif isinstance(self.conn, ZArray):
            return self.conn[...]
        else:
            return self.conn


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