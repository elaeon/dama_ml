from abc import ABC, abstractmethod
from ml.utils.numeric_functions import max_dtype
from ml.utils.basic import Shape
import numpy as np


class AbsGroup(ABC):

    def __init__(self, conn, name=None, end_node=False, dtypes=None, index=None, alias_map=None):
        self.conn = conn
        self.name = name
        self.end_node = end_node
        self.static_dtypes = dtypes
        self.index = index
        self.slice = slice(0, np.inf)
        self.counter = 0
        if alias_map is None:
            self.alias_map = {}
            self.inv_map = {}
        else:
            self.alias_map = alias_map
            self.inv_map = {value: key for key, value in self.alias_map.items()}

    def set_alias(self, name, alias):
        self.alias_map[alias] = name
        self.inv_map[name] = alias

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
        if elem.dtypes is None:  # fixme
            return elem.conn
        elif len(elem.groups) == 1:
            array = elem.to_ndarray()
            if len(elem.shape[elem.groups[0]]) == 0:  # fixme
                array = array[0]
            return array
        else:
            return elem

    def __len__(self):
        return self.shape.to_tuple()[0]

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.slice)

    @property
    @abstractmethod
    def dtypes(self) -> np.dtype:
        return NotImplemented

    @property
    def dtype(self) -> np.dtype:
        return max_dtype(self.dtypes)

    @property
    @abstractmethod
    def shape(self) -> Shape:
        return NotImplemented

    @abstractmethod
    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        return NotImplemented

    @property
    def groups(self) -> tuple:
        return self.dtypes.names
