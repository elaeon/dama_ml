import subprocess
from dama.utils.config import config_filepath
from dama.utils.config import get_settings
from dama.utils.files import check_or_create_path_dir
from dama.utils.files import dir_exists


def run(args):
    if args.edit:
        filepath = config_filepath()
        subprocess.check_call(["nano", filepath])
    elif args.init_repo:
        settings = get_settings("paths")
        repo_init(settings["code_path"], args.init_repo)


def repo_init(path, repo_name):
    from git import Repo
    check_point = check_or_create_path_dir(path, repo_name)
    if dir_exists(check_point):
        print("The repository '{}' already exists".format(repo_name))
        answer = input('Do you want to reset it? [y/n]> ').strip().lower()
        if len(answer) == 0 or answer.startswith("n"):
            print("No changes made")
            return
    repo = Repo.init(check_point, bare=False)
    print("OK")

