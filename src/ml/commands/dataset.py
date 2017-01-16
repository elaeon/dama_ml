import os

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.utils.numeric_functions import humanize_bytesize
from ml.clf.wrappers import DataDrive
from ml.ds import DataSetBuilder
from ml.utils.files import rm, get_models_path, delete_file_model
from ml.utils.files import get_date_from_file, get_models_from_dataset

settings = get_settings("ml")

  
def run(args):
    if args.info:
        dataset = DataSetBuilder(args.info, 
            dataset_path=settings["dataset_path"])
        dataset.info(classes=True)
    elif args.rm:
        try:
            dataset = DataSetBuilder(args.rm)
            models_path = get_models_from_dataset(dataset.md5(), settings["checkpoints_path"])
            print("Dataset: {}".format(dataset.url()))
            rm(dataset.url())
            for model_path, _ in models_path:
                print("Model: {}".format(model_path))
                rm(model_path)
        except IOError:
            pass
        delete_orphans(settings["checkpoints_path"])
        print("Done.")
    elif args.clean:
        delete_orphans(settings["checkpoints_path"])
        print("Done.")
    elif args.used_in:
        dataset = DataSetBuilder(args.used_in)
        for _, model_path_meta in get_models_from_dataset(dataset.md5(), settings["checkpoints_path"]):
            print("Dataset used in model: {}".format(DataDrive.read_meta("model_module", model_path_meta)))
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
                date = get_date_from_file(path+filename)
                table.append([filename, humanize_bytesize(size), date.strftime("%Y-%m-%d %H:%M UTC")])
                total_size += size
        print("Total size: {}".format(humanize_bytesize(total_size)))
        order_table_print(headers, table, "dataset", reverse=False)


def dataset_model_relation():
    datasets = {}
    for parent, childs, files in os.walk(settings["dataset_path"]):
        datasets[parent] = files

    dataset_md5 = {}
    for parent, datasets_name in datasets.items():
        for dataset_name in datasets_name:
            dataset = DataSetBuilder(dataset_name)
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


def delete_orphans(checkpoints_path):
    for clf, model_name_v, md5, dataset_name in get_model_dataset(get_models_path(checkpoints_path)):
        if dataset_name is None:
            name_fragments = model_name_v.split(".")
            model_name = ".".join(name_fragments[:-1])
            model_version = name_fragments[-1]
            path = delete_file_model(clf, model_name, model_version, checkpoints_path)
            print(path)
