from h5py.highlevel import Group as NativeH5Group
import numpy as np
from ml.abc.group import AbsBaseGroup


class HDF5Group(AbsBaseGroup):
    inblock = False

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.keys()])

    def get_group(self, group) -> AbsBaseGroup:
        if isinstance(self.conn, NativeH5Group):
            return HDF5Group(self.conn[group])
        else:
            return HDF5Group(self.conn)

    def get_conn(self, group):
        if isinstance(self.conn, NativeH5Group):
            return self.conn[group]
        else:
            return self.conn

    def cast(self, value):
        if value.dtype == np.dtype("datetime64[ns]"):
            return value.astype("int8")
        else:
            return value