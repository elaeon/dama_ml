from abc import ABC, abstractmethod
from dama.utils.numeric_functions import max_dtype
from dama.utils.core import Shape, Chunks
import numpy as np


__all__ = ['AbsConn']


class AbsConn(ABC):
    dtypes = None
    conn = None
    attrs = None

    def __init__(self, conn, dtypes=None):
        self.conn = conn
        self.attrs = Attrs()
        if dtypes is not None:
            self.dtypes = dtypes

    @abstractmethod
    def __getitem__(self, item):
        return NotImplemented

    @abstractmethod
    def __setitem__(self, key, value):
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

    @property
    @abstractmethod
    def shape(self) -> Shape:
        return NotImplemented

    @property
    def size(self) -> int:
        return self.shape[0]

    @abstractmethod
    def to_ndarray(self, dtype: np.dtype = None) -> np.ndarray:
        return NotImplemented

    @abstractmethod
    def store(self, driver: 'AbsDriver'):
        return NotImplemented

    @property
    @abstractmethod
    def chunksize(self) -> Chunks:
        return NotImplemented


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Attrs(dict, metaclass=Singleton):
    pass
