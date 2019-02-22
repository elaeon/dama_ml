import os
from dama.utils.config import get_settings
from dama.utils.numeric_functions import humanize_bytesize
from pydoc import locate

settings = get_settings("paths")

  
def run(args):
    from dama.measures import ListMeasure
    from dama.data.ds import Data
    from dama.data import drivers

    if args.info:
        from dama.utils.core import Login, Metadata
        from dama.data.drivers.core import DataDoesNotFound
        login = Login(url=os.path.join(settings["metadata_path"], "metadata.sqlite3"), table="metadata")
        metadata = Metadata(login=login)
        data = metadata.query("SELECT name, driver, dir_levels, hash FROM {} WHERE hash = '{}'".format(login.table,
                                                                                                       args.hash[0]))
        if len(data) == 0:
            print("Resource does not exists")
        else:
            row = data[0]
            driver = locate(row[1])
            path = row[2].replace(driver.cls_name(), "")
            group_name = None
            name = row[0]
            with Data(dataset_path=path, name=name, group_name=group_name, driver=driver()) as dataset:
                try:
                    dataset.info()
                except DataDoesNotFound:
                    metadata.remove_data(row[3])
                    print("Resource does not exists. The metadata was removed.")

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
        from dama.utils.core import Login, Metadata
        login = Login(url=os.path.join(settings["metadata_path"], "metadata.sqlite3"), table="metadata")
        headers = ["hash", "name", "driver", "size", "timestamp"]
        metadata = Metadata(login)
        if args.items is not None:
            elems = args.items.split(":")
            stop = None
            if len(elems) > 1:
                start = int(elems[0])
                if elems[1] != '':
                    stop = int(elems[1])
            else:
                start = int(elems[0])
            page = slice(start, stop)
        else:
            page = slice(None, None)
        total = metadata.query("SELECT COUNT(*) FROM {}".format(login.table))
        df = metadata.data(headers, page, order_by="timestamp")
        df.rename(columns={"timestamp": "datetime UTC"}, inplace=True)
        df["size"] = df["size"].apply(humanize_bytesize)
        print("Total {} / {}".format(len(df), total[0][0]))
        list_measure = ListMeasure(headers=df.columns, measures=df[df.columns].values)
        print(list_measure.to_tabulate())
