from h5py import Group as NativeH5Group
import numpy as np
from dama.abc.group import AbsDictGroup


class HDF5Group(AbsDictGroup):
    inblock = False

    def get_group(self, group) -> AbsDictGroup:
        if isinstance(self.conn, NativeH5Group):
            dtypes = self.dtypes_from_groups(group)
            return HDF5Group(self.conn[group], dtypes)
        else:
            return HDF5Group(self.conn, self.dtypes)

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