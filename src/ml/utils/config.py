import configparser
import os
from ml.utils.files import check_or_create_path_dir, path2module
import pkg_resources


FILENAME = "settings.cfg"
BASE_DIR = ".softstream"


def get_settings(key: str) -> dict:
    settings_path = config_filepath()
    config = configparser.ConfigParser()
    config.read(settings_path)
    try:
        return {flag: value for flag, value in config.items(key)}
    except configparser.NoSectionError as e:
        if not build_settings_file(rewrite=True):
            raise configparser.NoSectionError(e.section)


def build_settings_file(rewrite: bool = False) -> bool:
    settings_path = config_filepath()
    if os.path.exists(settings_path) is False or rewrite is True:
        filepath = os.path.expanduser("~")
        check_or_create_path_dir(filepath, BASE_DIR)
        if set_up_cfg(settings_path):
            print("Config file saved in {}".format(settings_path))
            return True
    else:
        print("Config file already exists in {}".format(settings_path))
        return False


def set_up_cfg(filepath):
    setting_expl = pkg_resources.resource_filename('ml', 'data/settings.cfg.example')
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    home = os.path.expanduser("~")
    with open(setting_expl, 'r') as f:
        cfg = f.read()
        cfg = cfg.format(home=home, base=base)
    with open(filepath, 'w') as f:
        f.write(cfg)
    return True


def config_filepath() -> str:
    return os.path.join(os.path.expanduser("~"), BASE_DIR, FILENAME)


def get_fn_name(fn, section=None) -> str:
    if fn.__module__ == "__main__":
        settings = get_settings(section)
        fn_module = path2module(settings["class_path"])
    else:
        fn_module = fn.__module__
    return "{}.{}".format(fn_module, fn.__name__)
