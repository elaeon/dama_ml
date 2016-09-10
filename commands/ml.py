import sys
sys.path.append("/home/alejandro/Programas/ML")

import argparse
import ml
import os

from utils.config import get_settings
from utils.order import order_table_print

settings = get_settings("ml")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
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
