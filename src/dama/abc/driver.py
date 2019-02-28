from abc import ABC, abstractmethod
from numcodecs.abc import Codec
import numpy as np
import os
from dama.utils.config import get_settings
from dama.utils.logger import log_config
from dama.utils.core import Login, Chunks
from dama.groups.core import DaGroup
from dama.utils.files import build_path


settings = get_settings("paths")
log = log_config(__name__)


class AbsDriver(ABC):
    persistent = None
    ext = None

    def __init__(self, compressor: Codec = None, login: Login = None, mode: str = 'a', path: str = None):
        self.compressor = compressor
        self.conn = None
        if compressor is not None:
            self.compressor_params = {"compression": self.compressor.codec_id,
                                      "compression_opts": self.compressor.level}
        else:
            self.compressor_params = {}

        self.mode = mode
        self.attrs = None
        self.path = path
        self.url = None
        self.login = login
        log.debug("Driver: {}, mode: {}, compressor: {}".format(self.cls_name(),
                                                                self.mode, self.compressor))

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def build_url(self, filename, group_level=None, with_class_name=True, path=None):
        filename = "{}.{}".format(filename, self.ext)
        if path is None:
            path = self.path
        else:
            self.path = path

        if with_class_name is True:
            if group_level is None:
                dir_levels = [path, self.cls_name(), filename]
            else:
                dir_levels = [path, self.cls_name(), group_level, filename]
        else:
            dir_levels = [path, filename]
        self.url = os.path.join(*dir_levels)
        build_path(dir_levels[:-1])

    def data(self, chunks: Chunks) -> DaGroup:
        return DaGroup(self.absgroup, chunks=chunks)

    @property
    def absgroup(self):
        return NotImplemented

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @abstractmethod
    def __contains__(self, item):
        return NotImplemented

    @abstractmethod
    def open(self):
        return NotImplemented

    @abstractmethod
    def close(self):
        return NotImplemented

    @property
    @abstractmethod
    def dtypes(self):
        return NotImplemented

    @abstractmethod
    def set_schema(self, dtypes: np.dtype, idx: list = None, unique_key: list = None):
        return NotImplemented

    @abstractmethod
    def set_data_shape(self, shape):
        return NotImplemented

    @abstractmethod
    def destroy(self):
        return NotImplemented

    @abstractmethod
    def exists(self):
        return NotImplemented

    @abstractmethod
    def spaces(self):
        return NotImplemented
