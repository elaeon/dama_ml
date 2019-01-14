from ml.utils.config import get_settings
from ml.utils.files import file_exists
from git import Repo
import os
import subprocess
import time


settings = get_settings("paths")


def run_commit(args):
    print("ccc")
    if args.run:
        no_error_code = prepare_run_code(args.run, args.name)
        if args.commit_msg:
            if no_error_code:
                repo_add_commit(settings["code_path"], args.name, args.run, args.commit_msg)
        else:
            print("file not commited")


def run_revert(args):
    if args.checkout:
       repo_checkout_file(settings["code_path"], args.name, args.checkout, args.commit, args.branch)


def prepare_run_code(name, repo_name) -> bool:
    filepath = os.path.join(settings["code_path"], repo_name, name)
    repo_path = os.path.join(settings["code_path"], repo_name)
    if file_exists(filepath):
        try:
            subprocess.check_call(["python", filepath])
        except subprocess.CalledProcessError:
            return False
        else:
            return True
    else:
        print("ERROR: {} does not exists in {}".format(name, repo_path))
        return False


def repo_add_commit(repo_code_path, repo_name, filename, commit_msg, branch="master"):
    repo = Repo(os.path.join(repo_code_path, repo_name))
    repo.index.add([filename])
    repo.index.commit(commit_msg)
    print(time.strftime("%a, %d %b %Y %H:%M", time.gmtime(repo.head.commit.committed_date)))


def repo_checkout_file(repo_code_path, repo_name, filename, commit, branch="master"):
    repo = Repo(os.path.join(repo_code_path, repo_name))
    print(filename, commit, branch)
    if repo.active_branch.name != branch:
        repo.git.checkout(branch)
        print("Changed to branch {}".format(branch))

    repo.git.checkout(commit)#, b="new1")
    #print(repo.working_tree_dir)
