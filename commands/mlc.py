import os
import argparse
import datetime

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.utils.numeric_functions import humanize_bytesize
from ml.clf.generic import DataDrive
from ml.ds import DataSetBuilder

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


def get_models_path():
    classes = {}
    for parent, childs, files in os.walk(settings["checkpoints_path"]):
        parent = parent.split("/").pop()
        if parent and len(childs) > 0:
            classes[parent] = childs
    return classes


def delete_file_model(clf, model_name, version):
    name_version = model_name + "." + version
    path = os.path.join(settings["checkpoints_path"], clf, name_version)
    rm(path)
    return path


def dataset_model_relation():
    datasets = {}
    for parent, childs, files in os.walk(settings["dataset_path"]):
        datasets[parent] = files

    dataset_md5 = {}
    for parent, datasets_name in datasets.items():
        for dataset_name in datasets_name:
            dataset = DataSetBuilder.load_dataset(
                dataset_name, dataset_path=settings["dataset_path"], info=False)
            dataset_md5[dataset.md5()] = dataset_name
    return dataset_md5


def get_model_dataset(classes):
    dataset_md5 = dataset_model_relation()
    for clf, dataset in classes.items():
        for name_version in dataset:
            md5 = DataDrive.read_meta(
                "md5", os.path.join(
                    settings["checkpoints_path"], clf, name_version, name_version))
            yield clf, name_version, md5, dataset_md5.get(md5, None)


def get_models_from_dataset(md5):
    from collections import defaultdict
    models_path = get_models_path()
    models_md5 = defaultdict(list)
    for clf, dataset in models_path.items():
        for name_version in dataset:
            model_path_meta = os.path.join(
                settings["checkpoints_path"], clf, name_version, name_version)
            model_path = os.path.join(
                settings["checkpoints_path"], clf, name_version)
            model_md5 = DataDrive.read_meta("md5", model_path_meta)
            models_md5[model_md5].append((model_path, model_path_meta))
    return models_md5.get(md5, None)


def delete_orphans():
    for clf, model_name_v, md5, dataset_name in get_model_dataset(get_models_path()):
        if dataset_name is None:
            name_fragments = model_name_v.split(".")
            model_name = ".".join(name_fragments[:-1])
            model_version = name_fragments[-1]
            path = delete_file_model(clf, model_name, model_version)
            print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--info", type=str, help="name")
    parser.add_argument("--rm", type=str, help="delete elements")
    args = parser.parse_args()

    if args.models and args.dataset and args.rm:
        if args.rm == "orphans":
            delete_orphans()            
            print("Done.")
    elif args.models and args.dataset and args.info:
        dataset = DataSetBuilder.load_dataset(args.info, 
            dataset_path=settings["dataset_path"], info=False)
        for _, model_path_meta in get_models_from_dataset(dataset.md5()):
            print("Dataset used in model: {}".format(DataDrive.read_meta("model_module", model_path_meta)))
    elif args.models:
        models_path = get_models_path()
        if args.info:            
            headers = ["classif", "dataset", "version", "group", "score"]
            table = []
            for clf, dataset in models_path.items():
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
        elif args.rm:
            clf, model_name, version = args.rm.split(".")
            path = delete_file_model(clf, model_name, version)
            print(path)
            print("Done.")
        else:
            headers = ["classif", "dataset", "version", "group"]
            table = []
            for clf, dataset in models_path.items():
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
        if args.info:
            dataset = DataSetBuilder.load_dataset(args.info, 
                dataset_path=settings["dataset_path"], info=False)
            dataset.info(classes=True)
        elif args.rm:
            dataset = DataSetBuilder.load_dataset(args.rm, 
                dataset_path=settings["dataset_path"], info=False)
            models_path = get_models_from_dataset(dataset.md5())                     
            print("Dataset: {}".format(dataset.url()))
            rm(dataset.url())
            for model_path, _ in models_path:
                print("Model: {}".format(model_path))
                rm(model_path)
            delete_orphans()
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
