from abc import ABC, abstractmethod
from ml.utils.numeric_functions import max_dtype
from ml.utils.basic import Shape
import numpy as np


class Slice(object):
    def __init__(self, start=None, stop=None, step=None, index=None):
        self.slice = slice(start, stop, step)
        self.index = index

    def update(self, item) -> 'Slice':
        if isinstance(item, slice) and self.index is None:
            return Slice(item.start, item.stop, item.step)
        elif isinstance(item, slice) and self.index is not None:
            index = self.index[item]
            return Slice(item.start, item.stop, item.step, index=index)
        elif isinstance(item, list):
            index = self.range_index(item, self.slice)
            return Slice(self.slice.start, self.slice.stop, self.slice.step, index=index)
        else:
            return Slice(self.slice.start, self.slice.stop, self.slice.step)

    def range_index(self, index, item):
        nindex = []
        start, stop = self.start_stop(item)
        for elem in index:
            if start <= elem <= stop:
                nindex.append(elem)
        return nindex

    def start_stop(self, item):
        if item.start is None:
            start = 0
        else:
            start = item.start

        if item.stop is None:
            stop = np.inf
        else:
            stop = item.stop
        return start, stop

    @property
    def stop(self):
        return self.slice.stop

    @property
    def start(self):
        return self.slice.start

    @property
    def idx(self):
        if self.index is not None:
            return self.index
        else:
            return self.slice

    def __str__(self):
        if self.index is not None:
            if len(self.index) > 3:
                return "{} ...".format(",".join([str(e) for e in self.index[:3]]))
            else:
                return "{}".format(",".join([str(e) for e in self.index]))
        else:
            return str(self.slice)


class AbsGroup(ABC):

    def __init__(self, conn, name=None, dtypes=None, index=None, alias_map=None):
        self.conn = conn
        self.name = name
        self.static_dtypes = dtypes
        self.index = index
        self.slice = Slice(0, None)
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
