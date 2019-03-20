from dama.abc.group import AbsDaskGroup
import numpy as np


class CSVGroup(AbsDaskGroup):
    inblock = True
    block = True

    def __getitem__(self, item):
        pass

    def __setitem__(self, item, value):
        pass

    def __iter__(self):
        pass

    def get_group(self, group):
        return self[group]

    def get_conn(self, group):
        return self[group]

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        pass

    def to_df(self):
        pass
