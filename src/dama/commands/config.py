import subprocess
from dama.utils.config import config_filepath


def run(args):
    if args.edit:
        filepath = config_filepath()
        subprocess.check_call(["nano", filepath])
