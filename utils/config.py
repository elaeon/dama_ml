import ConfigParser

def get_settings(filename="settings.cfg"):
    flags = {"root_data": None, 
            "examples": "examples/", 
            "pictures": "Pictures/", 
            "dataset": "dataset/",
            "checkpoints": "checkpoints/",
            "detector_name": "detector"}
    config = ConfigParser.ConfigParser()
    config.read('settings.cfg')
    for flag, v in flags.items():
        flags[flag] = config.get("ml", flag)
    return flags
