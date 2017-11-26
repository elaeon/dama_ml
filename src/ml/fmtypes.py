from collections import namedtuple
Fmtype = namedtuple('Fmtype', 'id name')

BOOLEAN = Fmtype(id=0, name="boolean")
NANBOOLEAN = Fmtype(id=1, name="nan boolean")
ORDINAL = Fmtype(id=2, name="ordinal")
CARDINAL = Fmtype(id=3, name="cardinal")
DENSE = Fmtype(id=4, name="dense")

