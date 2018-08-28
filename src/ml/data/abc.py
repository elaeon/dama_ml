"""
Module for create datasets from distinct sources of data.
"""
from abc import ABC, abstractmethod


class AbsDataset(ABC):
    @abstractmethod
    def __enter__(self):
        return NotImplemented

    @abstractmethod
    def __exit__(self, type, value, traceback):
        return NotImplemented

    @abstractmethod
    def __iter__(self):
        return NotImplemented

    @abstractmethod
    def __getitem__(self, key):
        return NotImplemented

    @abstractmethod
    def __setitem__(self, key, value):
        return NotImplemented

    @abstractmethod
    def __next__(self):
        return NotImplemented

    @abstractmethod
    def chunks_writer(self, name, data, init=0):
        return NotImplemented

    @abstractmethod
    def chunks_writer_split(self, data_key, labels_key, data, labels_column, init=0):
        return NotImplemented

    @abstractmethod
    def destroy(self):
        return NotImplemented

    @abstractmethod
    def url(self):
        return NotImplemented

    @abstractmethod
    def exists(self):
        return NotImplemented

    @abstractmethod
    def reader(self, *args, **kwargs):
        return NotImplemented

    @property
    @abstractmethod
    def shape(self):
        return NotImplemented

    @property
    @abstractmethod
    def columns(self):
        return NotImplemented

    @abstractmethod
    def num_features(self):
        return NotImplemented

    @abstractmethod
    def to_df(self):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def concat(datasets, chunksize:int=0, name:str=None):
        return NotImplemented
