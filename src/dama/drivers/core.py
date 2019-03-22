import zarr
import h5py
import numpy as np
import os

from numcodecs import MsgPack
from dama.abc.driver import AbsDriver
from dama.utils.files import rm
from dama.utils.logger import log_config
from dama.abc.group import DaGroupDict
from dama.utils.core import Chunks, Shape


log = log_config(__name__)


class HDF5(AbsDriver):
    persistent = True
    ext = 'h5'
    data_tag = "data"
    metadata_tag = "metadata"
    insert_by_rows = False

    def __getitem__(self, item):
        return self.conn[self.data_tag][item]

    def __setitem__(self, key, value):
        self.conn[self.data_tag][key] = value

    def __contains__(self, item):
        return item in self.conn

    def manager(self, chunks: Chunks):
        self.chunksize = chunks
        groups = [(group, self[group]) for group in self.groups]
        return DaGroupDict.convert(groups, chunks=chunks)

    def absgroup(self):
        pass

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

    @property
    def shape(self) -> Shape:
        shape = {}
        for group in self.groups:
            shape[group] = self[group].shape
        return Shape(shape)

    def cast(self, value):
        if value.dtype == np.dtype("datetime64[ns]"):
            return value.astype("int8")
        else:
            return value

    def absconn(self):
        pass


class Zarr(AbsDriver):
    persistent = True
    ext = "zarr"
    data_tag = "data"
    metadata_tag = "metadata"
    insert_by_rows = False

    def __getitem__(self, item):
        return self.conn[self.data_tag][item]

    def __setitem__(self, key, value):
        self.conn[self.data_tag][key] = value

    def __contains__(self, item):
        return item in self.conn

    def manager(self, chunks: Chunks):
        self.chunksize = chunks
        if self.groups is not None:
            groups = [(group, self[group]) for group in self.groups]
            return DaGroupDict.convert(groups, chunks=chunks)

    def absgroup(self):
        pass

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

    @property
    def shape(self) -> Shape:
        shape = {}
        for group in self.groups:
            shape[group] = self[group].shape
        return Shape(shape)

    def spaces(self) -> list:
        return list(self.conn.keys())

    def cast(self, value):
        return value

    def absconn(self):
        pass


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

    def absconn(self):
        pass


class StcArray(AbsDriver):
    persistent = False
    ext = None
    insert_by_rows = False

    def __getitem__(self, item):
        return self.conn[item]

    def __setitem__(self, key, value):
        self.conn[key] = value

    def __contains__(self, item):
        return item in self.conn

    def manager(self, chunks: Chunks):
        # self.chunksize = chunks
        groups = [(group, self[group]) for group in self.groups]
        return DaGroupDict.convert(groups, chunks=chunks)

    def absconn(self):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def destroy(self):
        pass

    @property
    def dtypes(self):
        return self.conn.dtype

    def exists(self):
        return self.conn is not None

    def set_data_shape(self, shape):
        pass

    def set_schema(self, dtypes: np.dtype, idx: list = None, unique_key=None):
        pass

    def spaces(self):
        pass

    @staticmethod
    def fit_shape(shape: Shape) -> Shape:
        _shape = {}
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
