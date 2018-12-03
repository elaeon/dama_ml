import zarr
import h5py
import numpy as np
from numcodecs import MsgPack
from ml.abc.driver import AbsDriver


class HDF5(AbsDriver):
    persistent = True
    ext = 'h5'

    def enter(self, url):
        if self.f is None:
            self.f = h5py.File(url, mode=self.mode)
            self.attrs = self.f.attrs

    def exit(self):
        self.f.close()
        self.f = None
        self.attrs = None

    def require_group(self, *args, **kwargs):
        return self.f.require_group(*args, **kwargs)

    def require_dataset(self, group:str, name:str, shape: tuple, dtype: np.dtype) -> None:
        self.f[group].require_dataset(name, shape, dtype=dtype, chunks=True,
                                      exact=True,
                                      **self.compressor_params)

    def auto_dtype(self, dtype: np.dtype):
        if dtype == np.dtype("O") or dtype.type == np.str_:
            return h5py.special_dtype(vlen=str)
        else:
            return dtype


class Zarr(AbsDriver):
    persistent = True
    ext = "zarr"

    def enter(self, url):
        if self.f is None:
            self.f = zarr.open(url, mode=self.mode)
            self.attrs = self.f.attrs

    def exit(self):
        self.f = None
        self.attrs = None

    def require_group(self, *args, **kwargs):
        return self.f.require_group(*args, **kwargs)

    def require_dataset(self, group:str, name:str, shape: tuple, dtype: np.dtype) -> None:
        if dtype == np.dtype("O"):
            object_codec = MsgPack()
        else:
            object_codec = None
        self.f[group].require_dataset(name, shape, dtype=dtype, chunks=True,
                                      exact=True, object_codec=object_codec,
                                      compressor=self.compressor)

    def auto_dtype(self, dtype: np.dtype):
        return dtype


class Memory(Zarr):
    presistent = False

    def enter(self, url=None):
        if self.f is None:
            self.f = zarr.group()
            self.attrs = self.f.attrs

    def exit(self):
        pass