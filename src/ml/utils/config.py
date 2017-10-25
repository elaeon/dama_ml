import ConfigParser
import os
import sys
from ml.utils.files import check_or_create_path_dir, set_up_cfg


def get_settings(key, filepath='', filename="settings.cfg"):
    if filepath == '' or filepath is None:
        filepath = os.path.expanduser("~")
        path = check_or_create_path_dir(filepath, ".mlpyp")
        settings_path = os.path.join(path, filename)
        if os.path.exists(settings_path) is False:
            if set_up_cfg(settings_path):
                print("Builded settings.cfg")
    else:
        settings_path = os.path.join(filepath, filename)
    config = ConfigParser.ConfigParser()
    config.read(settings_path)
    return {flag: value for flag, value in config.items(key)}
