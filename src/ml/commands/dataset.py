import os
from ml.utils.config import get_settings
settings = get_settings("paths")

  
def run(args):
    from ml.utils.numeric_functions import humanize_bytesize
    from ml.measures import ListMeasure
    from ml.data.ds import Data
    from ml.utils.files import rm

    if args.group_name:
        dataset_path = os.path.join(settings["data_path"], args.group_name)
    else:
        dataset_path = settings["data_path"]

    if args.info:
        dataset = Data(dataset_path=dataset_path, name=args.info)
        #dataset.info(classes=args.targets)
    elif args.rm:
        try:
            for ds in args.rm:
                dataset = Data(ds)
                print("Dataset: {}".format(dataset.url))
                rm(dataset.url)
        except IOError:
            pass
        print("Done.")
    #elif args.used_in:
    #    dataset = Data.original_ds(args.used_in)
    #    for _, model_path_meta in get_models_from_dataset(dataset, settings["checkpoints_path"]):
    #        print("Dataset used in model: {}".format(DataDrive.read_meta("model_module", model_path_meta)))
    #elif args.sts:
    #    dataset = Data.original_ds(args.sts)
    #    print(dataset.stadistics())
    else:
        datasets = {}
        for parent, childs, files in os.walk(dataset_path):
            datasets[parent] = files
        
        headers = ["dataset", "size", "date"]
        table = []
        total_size = 0
        for path, files in datasets.items():
            for filename in files:
                size = os.stat(os.path.join(path, filename)).st_size
                dl = Data(name=filename)
                if dl is not None:
                    with dl:
                        date = dl._get_attr("timestamp")
                    table.append([filename, humanize_bytesize(size), date])
                    total_size += size
        print("Total size: {}".format(humanize_bytesize(total_size)))
        list_measure = ListMeasure(headers=headers, measures=table)
        print(list_measure.to_tabulate(order_column="dataset"))

def dataset_model_relation():
    datasets = {}
    for parent, childs, files in os.walk(settings["data_path"]):
        datasets[parent] = files

    dataset_md5 = {}
    for parent, datasets_name in datasets.items():
        for dataset_name in datasets_name:
            dataset = Data.original_ds(dataset_name)
            dataset_md5[dataset.md5()] = dataset_name
    return dataset_md5


def get_model_dataset(classes):
    dataset_md5 = dataset_model_relation()
    for clf, dataset in classes.items():
        for name_version in dataset:
            md5 = DataDrive.read_meta(
                "md5", os.path.join(
                    settings["models_path"], clf, name_version, name_version))
            yield clf, name_version, md5, dataset_md5.get(md5, None)
