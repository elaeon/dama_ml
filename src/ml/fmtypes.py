from collections import namedtuple
from numpy import dtype

Fmtype = namedtuple('Fmtype', 'name dtype db_type')

BOOLEAN = Fmtype(name="boolean", dtype=dtype(bool), db_type="BOOLEAN")
NANBOOLEAN = Fmtype(name="nan boolean", dtype=dtype(int), db_type="INTEGER")
ORDINAL = Fmtype(name="int", dtype=dtype(int), db_type="INTEGER")
DENSE = Fmtype(name="float", dtype=dtype(float), db_type="FLOAT")
TEXT = Fmtype(name="text", dtype=dtype(object), db_type="TEXT")
DATETIME = Fmtype(name="datetime", dtype=dtype("datetime64[ns]"), db_type="TIMESTAMP")

fmtypes_map = {
    BOOLEAN.name: BOOLEAN,
    NANBOOLEAN.name: NANBOOLEAN,
    ORDINAL.name: ORDINAL,
    DENSE.name: DENSE,
    TEXT.name: TEXT,
    DATETIME.name: DATETIME
}

# class FmtypesT(object):
#    def __init__(self):
#        self.fmtypes = {}

#    def add(self, key, fmtype):
#        self.fmtypes[key] = fmtype.id

#    def fmtypes_fill(self, size, default=DENSE):
#        full_fmtypes =  [default.id for _ in range(size)]
#        for col, fmtype in self.fmtypes.items():
#            full_fmtypes[col] = fmtype
#        self.fmtypes =  full_fmtypes
