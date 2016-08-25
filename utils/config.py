import ConfigParser

def get_settings(key, filename="settings.cfg"):
    config = ConfigParser.ConfigParser()
    config.read('settings.cfg')
    return {flag: value for flag, value in config.items(key)}
