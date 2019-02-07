from collections import namedtuple
from numpy import dtype

Fmtype = namedtuple('Fmtype', 'name dtype db_type')

DEFAUL_GROUP_NAME = "g0"
BOOLEAN = Fmtype(name="boolean", dtype=dtype(bool), db_type="BOOLEAN")
NANBOOLEAN = Fmtype(name="nan boolean", dtype=dtype(int), db_type="INTEGER")
ORDINAL = Fmtype(name="int", dtype=dtype(int), db_type="INTEGER")
DENSE = Fmtype(name="float", dtype=dtype(float), db_type="FLOAT")
TEXT = Fmtype(name="text", dtype=dtype(object), db_type="TEXT")
DATETIME = Fmtype(name="datetime", dtype=dtype("datetime64[ns]"), db_type="TIMESTAMP")

fmtypes_map = {
    BOOLEAN.dtype: BOOLEAN,
    NANBOOLEAN.dtype: NANBOOLEAN,
    ORDINAL.dtype: ORDINAL,
    DENSE.dtype: DENSE,
    TEXT.dtype: TEXT,
    DATETIME.dtype: DATETIME
}
