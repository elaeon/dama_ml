import zarr
import h5py
import numpy as np
import os
from zarr.hierarchy import Group as ZGroup
from h5py.highlevel import Group as H5Group

from numcodecs import MsgPack
from ml.abc.driver import AbsDriver
from ml.utils.files import rm
from ml.abc.group import AbsBaseGroup
from ml.data.groups import DaGroup
from ml.utils.logger import log_config

log = log_config(__name__)


class HDF5(AbsDriver):
    persistent = True
    ext = 'h5'
    data_tag = "data"
    metadata_tag = "metadata"

    def __contains__(self, item):
        return item in self.conn

    @property
    def data(self) -> DaGroup:
        return DaGroup(HDF5Group(self.conn[self.data_tag]))

    def enter(self):
        if self.conn is None:
            self.conn = h5py.File(self.login.url, mode=self.mode)
            self.attrs = self.conn.attrs

    def exit(self):
        self.conn.close()
        self.conn = None
        self.attrs = None

    def require_dataset(self, level: str, group: str, shape: tuple, dtype: np.dtype) -> None:
        if dtype == np.dtype("O") or dtype.type == np.str_:
            dtype = h5py.special_dtype(vlen=str)
        elif dtype == np.dtype("datetime64[ns]"):
            dtype = np.dtype("int8")

        self.conn[level].require_dataset(group, shape, dtype=dtype, chunks=True, exact=True,
                                         **self.compressor_params)

    def destroy(self):
        rm(self.login.url)

    def exists(self):
        return os.path.exists(self.login.url)

    def set_schema(self, dtypes: np.dtype):
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


class Zarr(AbsDriver):
    persistent = True
    ext = "zarr"
    data_tag = "data"
    metadata_tag = "metadata"

    def __contains__(self, item):
        return item in self.conn

    @property
    def data(self) -> DaGroup:
        return DaGroup(ZarrGroup(self.conn[self.data_tag]))

    def enter(self):
        if self.conn is None:
            self.conn = zarr.open(self.login.url, mode=self.mode)
            self.attrs = self.conn.attrs
        return self

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

    def destroy(self):
        rm(self.login.url)

    def exists(self):
        return os.path.exists(self.login.url)

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

    def destroy(self):
        pass



class ZarrGroup(AbsBaseGroup):
    inblock = False

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.keys()])

    def get_group(self, group) -> AbsBaseGroup:
        if isinstance(self.conn, ZGroup):
            return ZarrGroup(self.conn[group])
        else:
            return ZarrGroup(self.conn)

    def get_conn(self, group):
        if isinstance(self.conn, ZGroup):
            return self.conn[group]
        else:
            return self.conn


class HDF5Group(AbsBaseGroup):
    inblock = False

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.keys()])

    def get_group(self, group) -> AbsBaseGroup:
        if isinstance(self.conn, H5Group):
            return ZarrGroup(self.conn[group])
        else:
            return ZarrGroup(self.conn)

    def get_conn(self, group):
        if isinstance(self.conn, H5Group):
            return self.conn[group]
        else:
            return self.conn

    def cast(self, value):
        if value.dtype == np.dtype("datetime64[ns]"):
            return value.astype("int8")
        else:
            return value