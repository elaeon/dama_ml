import os
from ml.utils.config import get_settings
from ml.utils.numeric_functions import humanize_bytesize


settings = get_settings("paths")

  
def run(args):
    from ml.measures import ListMeasure
    from ml.data.ds import Data
    from ml.data import drivers

    if args.info:
        path = os.path.join(settings["data_path"])
        #driver_class = getattr(drivers, args.driver)
        print(args.hash)
        #with Data(dataset_path=path, name=args.name, group_name=args.group_name, driver=driver_class()) as dataset:
        #    dataset.info()
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
        from ml.utils.core import Login, Metadata
        import pandas as pd
        login = Login(url=os.path.join(settings["metadata_path"], "metadata.sqlite3"), table="metadata")
        headers = ["hash", "name", "driver", "size", "timestamp"]
        metadata = Metadata()
        data = metadata.query(login, "SELECT {} FROM {} order by timestamp desc LIMIT 10".format(",".join(headers), login.table))
        total = metadata.query(login, "SELECT COUNT(*) FROM {}".format(login.table))
        data_list = []
        for elem in data:
            row = list(elem)
            row[3] = humanize_bytesize(row[3])
            row[4] = pd.to_datetime(row[4])
            data_list.append(row)
        headers[4] = "datetime UTC"
        list_measure = ListMeasure(headers=headers, measures=data_list)
        print("Total {} / {}".format(len(data), total[0][0]))
        print(list_measure.to_tabulate())

