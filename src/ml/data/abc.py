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
    def auto_dtype(self, ttype):
        return NotImplemented

    @abstractmethod
    def _set_space_shape(self, name, shape, dtype):
        return NotImplemented

    @abstractmethod
    def _get_data(self, name):
        return NotImplemented

    @abstractmethod
    def _set_space_fmtypes(self, num_features):
        return NotImplemented

    @abstractmethod
    def _set_attr(self, name, value):
        return NotImplemented
            
    @abstractmethod
    def _get_attr(self, name):
        return NotImplemented

    @abstractmethod
    def chunks_writer(self, name, data, init=0):
        return NotImplemented

    @abstractmethod
    def chunks_writer_split(self, data_key, labels_key, data, labels_column, init=0):
        return NotImplemented

    @abstractmethod
    def create_route(self):
        return NotImplemented

    @abstractmethod
    def destroy(self):
        return NotImplemented

    @abstractmethod
    def url(self):
        return NotImplemented

    @abstractmethod
    def exist(self):
        return NotImplemented

    @classmethod
    @abstractmethod
    def url_to_name(self, url):
        return NotImplemented

    @classmethod
    @abstractmethod
    def original_ds(self, name, dataset_path=None):
        return NotImplemented
    
