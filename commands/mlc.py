import os
import argparse
import datetime

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.utils.numeric_functions import humanize_bytesize
from ml.clf.generic import DataDrive

settings = get_settings("ml", filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def rm(path):
    import shutil
    try:
        shutil.rmtree(path)
    except OSError, e:
        if e.errno == 20:
            os.remove(path)
        else:
            print("{} not such file or directory".format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--info", type=str, help="name")
    parser.add_argument("--rm", type=str, help="delete elements")

    args = parser.parse_args()

    if args.models:
        classes = {}
        for parent, childs, files in os.walk(settings["checkpoints_path"]):
            parent = parent.split("/").pop()
            if parent and len(childs) > 0:
                classes[parent] = childs
        if args.info:            
            headers = ["classif", "dataset", "version", "group", "score"]
            table = []
            for clf, dataset in classes.items():
                for name_version in dataset:
                    model_path = os.path.join(
                        settings["checkpoints_path"], clf, name_version, name_version)
                    group_name = DataDrive.read_meta("group_name", model_path)
                    score = DataDrive.read_meta("score", model_path)
                    if args.info == group_name:
                        try:
                            name, version = name_version.split(".")
                            table.append([clf, name, version, group_name, score])
                        except ValueError:
                            pass
            order_table_print(headers, table, "score", reverse=True)
        else:
            headers = ["classif", "dataset", "version", "group"]
            table = []
            for clf, dataset in classes.items():
                for name_version in dataset:
                    group_name = DataDrive.read_meta(
                        "group_name", os.path.join(
                            settings["checkpoints_path"], clf, name_version, name_version))
                    try:
                        name, version = name_version.split(".")
                        table.append([clf, name, version, group_name])
                    except ValueError:
                        pass
            order_table_print(headers, table, "classif", reverse=False)
    elif args.dataset:
        from ml.ds import DataSetBuilder
        if args.info:
            dataset = DataSetBuilder.load_dataset(args.info, 
                dataset_path=settings["dataset_path"], info=False)
            dataset.info(classes=True)
        elif args.rm:
            dataset = DataSetBuilder.load_dataset(args.rm, 
                dataset_path=settings["dataset_path"], info=False)
            rm(dataset.url())
            print("Done.")
        else:
            datasets = {}
            for parent, childs, files in os.walk(settings["dataset_path"]):
                datasets[parent] = files
            
            headers = ["dataset", "size", "date"]
            table = []
            total_size = 0
            for path, files in datasets.items():
                for filename in files:
                    size = os.stat(path+filename).st_size
                    date = datetime.datetime.utcfromtimestamp(os.path.getmtime(path+filename))
                    table.append([filename, humanize_bytesize(size), date])
                    total_size += size
            print("Total size: {}".format(humanize_bytesize(total_size)))
            order_table_print(headers, table, "dataset", reverse=False)
    elif args.rm:
        if args.dataset:
            rm(settings["dataset_path"])
            rm(settings["checkpoints_path"])
