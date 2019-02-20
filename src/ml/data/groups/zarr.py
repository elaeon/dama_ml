import numpy as np
from ml.abc.group import AbsBaseGroup
from zarr.hierarchy import Group as NativeZGroup


class ZarrGroup(AbsBaseGroup):
    inblock = False

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(group, self.conn[group].dtype) for group in self.conn.keys()])

    def get_group(self, group) -> AbsBaseGroup:
        if isinstance(self.conn, NativeZGroup):
            return ZarrGroup(self.conn[group])
        else:
            return ZarrGroup(self.conn)

    def get_conn(self, group):
        if isinstance(self.conn, NativeZGroup):
            return self.conn[group]
        else:
            return self.conn
