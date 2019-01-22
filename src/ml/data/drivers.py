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


class ZarrGroup(object):
    def __init__(self, conn, name=None):
        self.conn = conn
        self.name = name

    def __getitem__(self, item):
        if isinstance(self.conn, ZGroup):
            return ZarrGroup(self.conn[item], name=item)
        else:
            return ZarrGroup(self.conn[item], name=self.name)

    def __setitem__(self, item, value):
        if hasattr(value, "groups"):
            for group in value.groups:
                self.conn[group][item] = value[group]

    def __len__(self):
        return self.shape.to_tuple()[0]

    @property
    def dtypes(self):
        if isinstance(self.conn, ZGroup):
            return [(key, self.conn[key].dtype) for key in self.conn.keys()]
        else:
            return [(self.name, self.conn.dtype)]

    @property
    def shape(self) -> Shape:
        shape = dict([(group, self.conn.shape) for group, _ in self.dtypes])
        return Shape(shape)

    def compute(self, chunksize=(258,)):
        if isinstance(self.conn, ZArray):
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
    def shape(self) -> Shape:
        shape = dict([(group, self.conn.shape) for group, _ in self.dtypes])
        return Shape(shape)

    def compute(self, chunksize=(258,)):
        return self.conn