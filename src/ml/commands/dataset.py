import os
import json
from ml.utils.config import get_settings
from ml.utils.numeric_functions import humanize_bytesize

settings = get_settings("paths")

  
def run(args):
    from ml.measures import ListMeasure
    from ml.data.ds import Data
    from ml.data import drivers

    if args.info:
        path = os.path.join(settings["data_path"])
        driver_class = getattr(drivers, args.driver)
        with Data(dataset_path=path, name=args.name, group_name=args.group_name, driver=driver_class()) as dataset:
            dataset.info()
    elif args.rm:
        path = os.path.join(settings["data_path"])
        driver_class = getattr(drivers, args.driver)
        try:
            with Data(dataset_path=path, name=args.name, group_name=args.group_name, driver=driver_class()) as dataset:
                print("Dataset: {}".format(dataset.url))
                dataset.destroy()
        except IOError:
            pass
        print("Done.")
    # elif args.used_in:
    #    dataset = Data.original_ds(args.used_in)
    #    for _, model_path_meta in get_models_from_dataset(dataset, settings["checkpoints_path"]):
    #        print("Dataset used in model: {}".format(DataDrive.read_meta("model_module", model_path_meta)))
    elif args.sts:
        path = os.path.join(settings["data_path"])
        driver_class = getattr(drivers, args.driver)
        with Data(dataset_path=path, name=args.name, group_name=args.group_name, driver=driver_class()) as dataset:
            print(dataset.stadistics())
    else:
        metadata_path = os.path.join(settings["data_path"], 'metadata')
        files_list = [os.path.join(metadata_path, filename) for filename in os.listdir(metadata_path)]
        table = []
        for filename in files_list:
            with open(filename, "r") as f:
                metadata = json.load(f)
                data = map(str, [metadata["name"], humanize_bytesize(metadata["size"]),
                              metadata["timestamp"], metadata["driver"].split(".")[-1],
                              metadata["hash"], metadata["description"]])
                table.append(list(data))
        headers = ["dataset", "size", "date", "driver", "hash", "description"]
        list_measure = ListMeasure(headers=headers, measures=table)
        print(list_measure.to_tabulate(order_column="dataset", limit=10))

