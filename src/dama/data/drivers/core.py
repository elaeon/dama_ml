import zarr
import h5py
import numpy as np
import os

from numcodecs import MsgPack
from dama.abc.driver import AbsDriver
from dama.utils.files import rm
from dama.utils.logger import log_config
from dama.data.groups.hdf5 import HDF5Group
from dama.data.groups.zarr import ZarrGroup


log = log_config(__name__)


class DataDoesNotFound(Exception):
    pass


class HDF5(AbsDriver):
    persistent = True
    ext = 'h5'
    data_tag = "data"
    metadata_tag = "metadata"

    def __contains__(self, item):
        return item in self.conn

    @property
    def absgroup(self):
        try:
            return HDF5Group(self.conn[self.data_tag])
        except KeyError:
            raise DataDoesNotFound

    def open(self):
        if self.conn is None:
            self.conn = h5py.File(self.url, mode=self.mode)
            self.attrs = self.conn.attrs

    def close(self):
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
        rm(self.url)

    def exists(self):
        return os.path.exists(self.url)

    def set_schema(self, dtypes: np.dtype, idx: list = None, unique_key: str = None):
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
    def absgroup(self):
        return ZarrGroup(self.conn[self.data_tag])

    def open(self):
        if self.conn is None:
            self.conn = zarr.open(self.url, mode=self.mode)
            self.attrs = self.conn.attrs
        return self

    def close(self):
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
        rm(self.url)

    def exists(self):
        return os.path.exists(self.url)

    def set_schema(self, dtypes: np.dtype, idx: list = None, unique_key=None):
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
    ext = None
    persistent = False

    def open(self):
        if self.conn is None:
            self.conn = zarr.group()
            self.attrs = self.conn.attrs

    def close(self):
        pass

    def destroy(self):
        pass
