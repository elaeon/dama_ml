from dama.abc.group import AbsDictGroup
from zarr.hierarchy import Group as NativeZGroup


class ZarrGroup(AbsDictGroup):
    inblock = False

    def __getitem__(self, item):
        return self.conn[item]

    def __setitem__(self, key, value):
        self.conn[key] = value

    def get_group(self, group) -> AbsDictGroup:
        if isinstance(self.conn, NativeZGroup):
            dtypes = self.dtypes_from_groups(group)
            return ZarrGroup(self.conn[group], dtypes)
        else:
            return ZarrGroup(self.conn, self.dtypes)

    def get_conn(self, group):
        if isinstance(self.conn, NativeZGroup):
            return self[group]
        else:
            return self.conn
