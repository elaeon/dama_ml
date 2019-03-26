import numpy as np
import os
import pandas as pd

from dama.abc.driver import AbsDriver
from dama.abc.conn import AbsConn
from dama.connexions.core import DaskDfConn
from dama.utils.files import rm
from dama.utils.logger import log_config
from dama.utils.core import Chunks, Shape
from dask.dataframe.io.csv import make_reader

log = log_config(__name__)


class CSV(AbsDriver):
    persistent = True
    ext = "csv"
    data_tag = None
    metadata_tag = "metadata"

    def __getitem__(self, item):
        return self.conn[item]

    def __setitem__(self, key, value):
        self.conn[key] = value

    def absconn(self):
        pass

    def manager(self, chunks: Chunks) -> AbsConn:
        return DaskDfConn(self.conn)

    def open(self):
        if self.conn is None:
            if self.mode == "r":
                self.conn = make_reader(pd.read_csv, 'read_csv', 'CSV')(self.url)
            else:
                raise NotImplementedError
            self.attrs = {}
        return self

    def close(self):
        self.conn = None
        self.attrs = None

    def require_dataset(self, level: str, group: str, shape: tuple, dtype: np.dtype) -> None:
        pass

    def destroy(self):
        rm(self.url)

    def exists(self):
        return os.path.exists(self.url)

    def set_schema(self, dtypes: np.dtype, idx: list = None, unique_key=None):
        pass

    def set_data_shape(self, shape):
        pass

    @property
    def dtypes(self) -> np.dtype:
        return np.dtype([(col, np.dtype(dtype)) for col, dtype in self.conn.dtypes.items()])

    def spaces(self) -> list:
        return list(self.conn.columns)

    def shape(self) -> Shape:
        length = self.conn.shape[0].compute()
        shape = dict([(group, (length,)) for group in self.groups])
        return Shape(shape)
