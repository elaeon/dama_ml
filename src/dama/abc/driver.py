from abc import ABC, abstractmethod
from numcodecs.abc import Codec
import numpy as np
from dama.utils.logger import log_config
from dama.utils.core import Login, Chunks
from dama.data.groups.core import DaGroup

log = log_config(__name__)


class AbsDriver(ABC):
    persistent = None
    ext = None

    def __init__(self, compressor: Codec = None, login: Login = None, mode: str = 'a'):
        self.compressor = compressor
        self.conn = None
        if compressor is not None:
            self.compressor_params = {"compression": self.compressor.codec_id,
                                      "compression_opts": self.compressor.level}
        else:
            self.compressor_params = {}

        self.mode = mode
        self.attrs = None
        self.login = login
        log.debug("Driver: {}, mode: {}, compressor: {}".format(self.cls_name(),
                                                                self.mode, self.compressor))

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
    def set_schema(self, dtypes: np.dtype, idx: list = None, unique_key: str = None):
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
