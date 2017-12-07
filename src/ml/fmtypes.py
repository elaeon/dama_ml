from collections import namedtuple
Fmtype = namedtuple('Fmtype', 'id name')

BOOLEAN = Fmtype(id=0, name="boolean")
NANBOOLEAN = Fmtype(id=1, name="nan boolean")
ORDINAL = Fmtype(id=2, name="ordinal")
CATEGORICAL = Fmtype(id=3, name="categorical")
DENSE = Fmtype(id=4, name="dense")

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
