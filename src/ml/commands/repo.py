from ml.utils.config import get_settings
from ml.utils.files import file_exists
from ml.utils.basic import time2str
from git import Repo
import os
import subprocess
import tabulate


settings = get_settings("paths")


def run(args):
    repo = Repo(os.path.join(settings["code_path"], args.name))
    if args.branch is None:
        if repo.head.is_detached:
            branch = None
        else:
            branch = repo.active_branch.name
    else:
        branch = args.branch

    if args.run:
        no_error_code = run_code(args.run, args.name)
        if args.commit_msg:
            if no_error_code:
                repo_add_commit(repo, args.run, args.commit_msg, branch)
        else:
            print("file not commited")
    elif args.checkout:
       repo_checkout_file(repo, args.checkout, args.commit, branch)
    elif args.head:
        if repo.head.is_detached:
            for head in repo.heads:
                for item in repo.iter_commits(head):
                    if item.hexsha == repo.head.commit.hexsha:
                        repo.git.checkout(head)
                        print("Head of {}".format(head))
        else:
            print("Nothing to do.")
    elif args.log:
        print("Branch {}".format(branch))
        repo.git.checkout(branch)
        commits = []
        for commit in repo.iter_commits(repo.head, max_count=20):
            l = [time2str(commit.committed_date), commit.hexsha, commit.message]
            commits.append(l)
        print(tabulate.tabulate(commits, ["date", "hexsha", "msg"]))
    elif branch:
        if branch == "all":
            active_branch = repo.active_branch.name
            for branch in repo.branches:
                if branch.name == active_branch:
                    print(" *", branch.name)
                else:
                    print("  ", branch.name)
        else:
            print("Branch {}".format(branch))
            if repo.active_branch.name != branch:
                repo.git.checkout(branch)


def run_code(name, repo_name) -> bool:
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


def repo_add_commit(repo, filename, commit_msg, branch_name):
    if repo.head.is_detached:
        for branch in repo.branches:
            if branch.name == branch_name:
                print("Branch '{}' already exists, choose another".format(branch_name))
                return
    changed_files = {item.a_path for item in repo.index.diff(None)}
    if filename in changed_files:
        if repo.head.is_detached:
            repo.git.checkout(repo.head.commit, b=branch_name)
        repo.index.add([filename])
        commit = repo.index.commit(commit_msg)
        print("Commit: {}".format(commit))
        print(time2str(repo.head.commit.committed_date))
    else:
        print("Not changes found")


def repo_checkout_file(repo, filename, commit, branch_name):
    print("Checkout to {} on branch {}".format(commit, branch_name))
    try:
        if repo.active_branch.name != branch_name:
            repo.git.checkout(branch_name)
            print("Changed to branch {}".format(branch_name))
        repo.git.checkout(commit)
    except TypeError as e:
        print(e)
        print("Try to return to the head of this branch")
        pass
