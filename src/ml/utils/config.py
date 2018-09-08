import configparser
import os
from ml.utils.files import check_or_create_path_dir, set_up_cfg

FILENAME = "settings.cfg"
BASE_DIR = ".mlpyp"


def get_settings(key:str) -> {}:
    settings_path = config_filepath()
    config = configparser.ConfigParser()
    config.read(settings_path)
    return {flag: value for flag, value in config.items(key)}


def build_settings_file(rewrite=False) -> None:
    filepath = os.path.expanduser("~")
    path = check_or_create_path_dir(filepath, BASE_DIR)
    settings_path = config_filepath()
    if os.path.exists(settings_path) is False or rewrite is True:
        if set_up_cfg(settings_path):
            print("Config file saved in {}".format(settings_path))
    else:
        print("Config file already exists in {}".format(settings_path))


def config_filepath():
    return os.path.join(os.path.expanduser("~"), BASE_DIR, FILENAME)