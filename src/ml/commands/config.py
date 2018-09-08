import os


def run(args):
    if args.edit:
        import subprocess
        from ml.utils.config import config_filepath
        filepath = config_filepath()
        subprocess.check_call(["nano", filepath])
    elif args.init:
        from ml.utils.config import build_settings_file
        build_settings_file()
    elif args.force_init:
        from ml.utils.config import build_settings_file
        build_settings_file(rewrite=True)