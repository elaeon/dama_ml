import logging
from abc import ABC, abstractmethod
from numcodecs.abc import Codec
from ml.utils.config import get_settings

settings = get_settings("ml")

log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


class AbsDriver(ABC):
    persistent = None
    ext = None

    def __init__(self, compressor: Codec=None):
        self.f = None
        self.compressor = compressor
        if compressor is not None:
            self.compressor_params = {"compression": self.compressor.codec_id,
                                      "compression_opts": self.compressor.level}
        else:
            self.compressor_params = {}

        self.mode = 'a'
        self.attrs = None
        log.debug("Driver: {}, mode: {}, compressor: {}".format(self.__class__.__name__,
                                                                    self.mode, self.compressor))

    def __getitem__(self, item):
        return self.f[item]

    def __setitem__(self, key, value):
        self.f[key] = value

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