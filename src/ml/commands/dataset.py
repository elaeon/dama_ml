import os

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.utils.numeric_functions import humanize_bytesize
from ml.clf.wrappers import DataDrive
from ml.clf.measures import ListMeasure
from ml.ds import DataSetBuilder, DataLabel
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
            print("Dataset: {}".format(dataset.url()))
            rm(dataset.url())
        except IOError:
            pass
        print("Done.")
    #elif args.clean:
    #    print("Done.")
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
                dl = DataLabel(name=filename)
                date = dl._get_attr("timestamp")
                dl.close_reader()
                table.append([filename, humanize_bytesize(size), date])
                total_size += size
        print("Total size: {}".format(humanize_bytesize(total_size)))
        list_measure = ListMeasure(headers=headers, measures=table)
        list_measure.print_scores(order_column="dataset")


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
