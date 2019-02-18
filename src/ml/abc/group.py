from abc import ABC, abstractmethod
from ml.utils.numeric_functions import max_dtype
from ml.utils.core import Shape
import numpy as np
import pandas as pd


class AbsBaseGroup(ABC):
    def __init__(self, conn):
        self.conn = conn
        self.attrs = Attrs()

    @property
    @abstractmethod
    def dtypes(self):
        return NotImplemented

    @abstractmethod
    def get_group(self, group):
        return NotImplemented

    @abstractmethod
    def get_conn(self, group):
        return NotImplemented

    @property
    def groups(self) -> tuple:
        return self.dtypes.names

    @property
    def dtype(self) -> np.dtype:
        return max_dtype(self.dtypes)

    def base_cls(self):
        return self.__class__.__bases__[0]

    def cast(self, value):
        return value

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @classmethod
    def cls_name(cls):
        return cls.__name__


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Attrs(dict, metaclass=Singleton):
    pass


class AbsGroup(AbsBaseGroup):
    __slots__ = ['conn', 'writer_conn', 'counter', 'attrs']

    def __init__(self, conn, writer_conn=None):
        super(AbsGroup, self).__init__(conn)
        self.writer_conn = writer_conn
        self.counter = 0

    @abstractmethod
    def __getitem__(self, item):
        return NotImplemented

    @abstractmethod
    def __setitem__(self, item, value):
        return NotImplemented

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        try:
            elem = self._iterator(self.counter)
            self.counter += 1
        except IndexError:
            raise StopIteration
        else:
            return elem

    def _iterator(self, counter):
        elem = self[counter]
        return elem

    def __len__(self):
        return self.shape.to_tuple()[0]

    def __repr__(self):
        return "{} {}".format(self.cls_name(), self.shape)

    def get_group(self, group):
        return self[group]

    def get_conn(self, group):
        return self[group]

    @property
    @abstractmethod
    def dtypes(self) -> np.dtype:
        return NotImplemented

    @property
    @abstractmethod
    def shape(self) -> Shape:
        return NotImplemented

    @abstractmethod
    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        return NotImplemented

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        return NotImplemented

    def items(self):
        return [(group, self.conn[group]) for group in self.groups]