import logging
from colorlog import ColoredFormatter
from dama.utils.config import get_settings


def log_config(file):
    settings = get_settings("log")
    log = logging.getLogger(file)
    logFormatter = ColoredFormatter("[ %(log_color)s%(levelname)-8s%(reset)s] - [%(name)s] %(message)s")
    # logFormatter = logging.Formatter("[%(levelname)s] - [%(name)s] %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(logFormatter)
    log.addHandler(handler)
    log.setLevel(int(settings["loglevel"]))
    return log
