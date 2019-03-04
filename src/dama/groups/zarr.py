from dama.abc.group import AbsBaseGroup
from zarr.hierarchy import Group as NativeZGroup


class ZarrGroup(AbsBaseGroup):
    inblock = False

    def get_group(self, group) -> AbsBaseGroup:
        if isinstance(self.conn, NativeZGroup):
            dtypes = self.dtypes_from_groups(group)
            return ZarrGroup(self.conn[group], dtypes)
        else:
            return ZarrGroup(self.conn, self.dtypes)

    def get_conn(self, group):
        if isinstance(self.conn, NativeZGroup):
            return self.conn[group]
        else:
            return self.conn
