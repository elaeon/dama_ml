from ml.utils.config import get_settings
from ml.utils.files import file_exists
from git import Repo
import os
import subprocess


settings = get_settings("paths")


def run(args):
    if args.run:
        prepare_run_code(args.run, args.name)
        if args.no_add:
            pass
        else:
            repo_add_file(settings["code_path"], args.name, args.add_file)


def prepare_run_code(name, repo_name):
    filepath = os.path.join(settings["code_path"], repo_name, name)
    repo_path = os.path.join(settings["code_path"], repo_name)
    if file_exists(filepath):
        try:
            subprocess.check_call(["python", filepath])
        except subprocess.CalledProcessError:
            pass
    else:
        print("ERROR: {} does not exists in {}".format(args.name, repo_path))


def repo_add_file(repo_path, branch="master"):
    repo = Repo(repo_path)
    print(repo.heads)
    #new_file_path = os.path.join("/tmp/repo/bare-repo", 'my-new-file')
    #open(new_file_path, 'wb').close()
    #repo.index.add([filepath])