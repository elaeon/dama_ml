import ConfigParser
import os

def get_settings(key, filepath='', filename="settings.cfg"):
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(filepath, filename))
    return {flag: value for flag, value in config.items(key)}
