from collections import namedtuple
Fmtype = namedtuple('Fmtype', 'id name type db_type')

BOOLEAN = Fmtype(id=0, name="boolean", type=bool, db_type="BOOLEAN")
NANBOOLEAN = Fmtype(id=1, name="nan boolean", type=int, db_type="INTEGER")
ORDINAL = Fmtype(id=2, name="ordinal", type=int, db_type="INTEGER")
CATEGORICAL = Fmtype(id=3, name="categorical", type=int, db_type="INTEGER")
DENSE = Fmtype(id=4, name="dense", type=float, db_type="FLOAT")
TEXT = Fmtype(id=5, name="text", type=str, db_type="TEXT")

fmtypes_map = {
    BOOLEAN.id: BOOLEAN.name,
    NANBOOLEAN.id: NANBOOLEAN.name,
    ORDINAL.id: ORDINAL.name,
    CATEGORICAL.id: CATEGORICAL.name,
    DENSE.id: DENSE.name
}

class FmtypesT(object):
    def __init__(self):
        self.fmtypes = {}

    def add(self, key, fmtype):
        self.fmtypes[key] = fmtype.id

    def fmtypes_fill(self, size, default=DENSE):
        full_fmtypes =  [default.id for _ in range(size)]
        for col, fmtype in self.fmtypes.items():
            full_fmtypes[col] = fmtype
        self.fmtypes =  full_fmtypes
