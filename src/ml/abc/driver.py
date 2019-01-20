from abc import ABC, abstractmethod
from numcodecs.abc import Codec
from ml.utils.logger import log_config


log = log_config(__name__)


class AbsDriver(ABC):
    persistent = None
    ext = None

    def __init__(self, compressor: Codec = None, login=None):
        self.f = None
        self.compressor = compressor
        if compressor is not None:
            self.compressor_params = {"compression": self.compressor.codec_id,
                                      "compression_opts": self.compressor.level}
        else:
            self.compressor_params = {}

        self.mode = 'a'
        self.attrs = None
        self.login = login
        log.debug("Driver: {}, mode: {}, compressor: {}".format(self.__class__.__name__,
                                                                self.mode, self.compressor))

    def __getitem__(self, item):
        return self.f[item]

    def __setitem__(self, key, value):
        self.f[key] = value

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
    def enter(self, url):
        return NotImplemented

    @abstractmethod
    def exit(self):
        return NotImplemented

    @abstractmethod
    def require_group(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def require_dataset(self, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def auto_dtype(self, dtype):
        return NotImplemented

    @abstractmethod
    def destroy(self, scope=None):
        return NotImplemented