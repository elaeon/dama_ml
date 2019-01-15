import os
import ntpath
import datetime
import sys
from pathlib import Path


def filename_from_path(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def filename_n_ext_from_path(path):
    filename = filename_from_path(path)
    filename_l = filename.split(".")
    filename_l.pop()
    return "".join(filename_l)


def check_or_create_path_dir(path, dirname):
    check_point = os.path.join(path, dirname)
    if not os.path.exists(check_point):
        os.makedirs(check_point)
    return check_point


def rm(path):
    import shutil
    try:
        shutil.rmtree(path)
    except OSError as e:
        if e.errno == 20:
            os.remove(path)
        else:
            print("{} not such file or directory".format(path))


def get_models_path(checkpoints_path):
    classes = {}
    for parent, childs, files in os.walk(checkpoints_path):
        parent = parent.split("/").pop()
        if parent and len(childs) > 0:
            classes[parent] = childs
    return classes


def get_date_from_file(file_path):
    return datetime.datetime.utcfromtimestamp(os.path.getmtime(file_path))


def path2module(class_path):
    try:
        main_path = os.path.abspath(sys.modules['__main__'].__file__)
    except AttributeError:
        main_path = "__main__"
    filepath = main_path.replace(class_path, "", 1).replace(".py", "")
    return ".".join(filepath.split("/"))


def file_exists(filepath):
    my_file = Path(filepath)
    return my_file.is_file()


def dir_exists(filepath):
    file_ = Path(filepath)
    return file_.is_dir()


def build_path(levels):
    levels = [level for level in levels if level is not None]
    path = os.path.join(*levels)
    if not dir_exists(path):
        os.makedirs(path)
    return path


def get_dir_file_size(path) -> int:
    try:
        if dir_exists(path):
            return sum(sum(map(lambda fname: os.path.getsize(os.path.join(directory, fname)), files)) for directory, folders, files in os.walk(path))
        return os.path.getsize(path)
    except FileNotFoundError:
        return 0