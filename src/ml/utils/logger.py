import logging
from ml.utils.config import get_settings


def log_config(file):
    settings = get_settings("log")
    log = logging.getLogger(file)
    logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(logFormatter)
    log.addHandler(handler)
    log.setLevel(int(settings["loglevel"]))
    return log