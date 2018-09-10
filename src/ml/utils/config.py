import configparser
import os
from ml.utils.files import check_or_create_path_dir, set_up_cfg, path2module


FILENAME = "settings.cfg"
BASE_DIR = ".mlpyp"


def get_settings(key:str) -> {}:
    settings_path = config_filepath()
    config = configparser.ConfigParser()
    config.read(settings_path)
    try:
        return {flag: value for flag, value in config.items(key)}
    except configparser.NoSectionError as e:
        if not build_settings_file():
            raise configparser.NoSectionError(e.section)
        else:
            return get_settings(key)


def build_settings_file(rewrite=False) -> None:
    filepath = os.path.expanduser("~")
    check_or_create_path_dir(filepath, BASE_DIR)
    settings_path = config_filepath()
    if os.path.exists(settings_path) is False or rewrite is True:
        if set_up_cfg(settings_path):
            print("Config file saved in {}".format(settings_path))
            return True
    else:
        print("Config file already exists in {}".format(settings_path))
    return False


def config_filepath():
    return os.path.join(os.path.expanduser("~"), BASE_DIR, FILENAME)


def get_fn_name(fn, section=None):
    if fn.__module__ == "__main__":
        settings = get_settings(section)
        fn_module = path2module(settings["class_path"])
    else:
        fn_module = fn.__module__
    return "{}.{}".format(fn_module, fn.__name__)