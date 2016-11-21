import os
import argparse
#import ml

from ml.utils.config import get_settings
from ml.utils.order import order_table_print

settings = get_settings("ml", filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def rm(path):
    import shutil
    try:
        shutil.rmtree(path)
    except OSError:
        print("{} not such file or directory".format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--rm", action="store_true")

    args = parser.parse_args()

    if args.models:
        classes = {}
        for parent, childs, files in os.walk(settings["checkpoints_path"]):
            parent = parent.split("/").pop()
            if parent and len(childs) > 0:
                classes[parent] = childs
        
        headers = ["classif", "dataset", "version"]
        table = []
        for clf, dataset in classes.items():
            for name_version in dataset:
                try:
                    name, version = name_version.split(".")
                    table.append([clf, name, version])
                except ValueError:
                    pass
        order_table_print(headers, table, "classif", reverse=False)
    elif args.rm:
        if args.dataset:
            rm(settings["dataset_path"])
            rm(settings["checkpoints_path"])
