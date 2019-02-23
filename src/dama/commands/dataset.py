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
        with Metadata(login=login) as metadata:
            data = metadata.query("SELECT name, driver, dir_levels, hash FROM {} WHERE hash = ?".format(login.table),
                                  (args.hash[0], ))
            if len(data) == 0:
                print("Resource does not exists")
            else:
                row = data[0]
                driver = locate(row[1])
                path = row[2].replace(driver.cls_name(), "")
                group_name = None
                name = row[0]
                with Data(dataset_path=path, name=name, group_name=group_name, driver=driver(mode="r")) as dataset:
                    try:
                        dataset.info()
                    except DataDoesNotFound:
                        print("Resource does not exists.")
    elif args.rm:
        from dama.utils.core import Login, Metadata
        from dama.data.it import Iterator
        login = Login(url=os.path.join(settings["metadata_path"], "metadata.sqlite3"), table="metadata")
        with Metadata(login=login) as metadata:
            if "all" in args.hash:
                data = metadata.data()[["name", "hash", "is_valid"]]
                data = data[data["is_valid"] == True]
                it = Iterator(data)
                to_delete = []
                for row in it:
                    df = row.to_df()
                    if df.values[0][0] is None:
                        continue
                    print(df.values)
                    to_delete.append(df["hash"].values[0])
                print(len(to_delete))
            else:
                metadata.invalid(args.hash[0])
        print("Done.")
    # elif args.used_in:
    #    dataset = Data.original_ds(args.used_in)
    #    for _, model_path_meta in get_models_from_dataset(dataset, settings["checkpoints_path"]):
    #        print("Dataset used in model: {}".format(DataDrive.read_meta("model_module", model_path_meta)))
    elif args.sts:
        from dama.utils.core import Login, Metadata
        login = Login(url=os.path.join(settings["metadata_path"], "metadata.sqlite3"), table="metadata")
        with Metadata(login=login) as metadata:
            data = metadata.query("SELECT name, driver, dir_levels, hash FROM {} WHERE hash = ?".format(login.table),
                                  (args.hash[0],))
            row = data[0]
            driver = locate(row[1])
            path = row[2].replace(driver.cls_name(), "")
            group_name = None
            name = row[0]
            with Data(dataset_path=path, name=name, group_name=group_name, driver=driver(mode="r")) as dataset:
                print(dataset.stadistics())
    else:
        from dama.utils.core import Login, Metadata
        from dama.utils.miscellaneous import str2slice
        import sqlite3
        login = Login(url=os.path.join(settings["metadata_path"], "metadata.sqlite3"), table="metadata")
        headers = ["hash", "name", "driver", "size", "datetime UTC"]
        page = str2slice(args.items)
        with Metadata(login) as metadata:
            try:
                total = metadata.query("SELECT COUNT(*) FROM %s WHERE is_valid=True" % login.table, ())
            except sqlite3.OperationalError as e:
                print(e)
            else:
                data = metadata.data()
                data = data[data["is_valid"] == True][page]
                df = data.to_df()
                df.rename(columns={"timestamp": "datetime UTC"}, inplace=True)
                df["size"] = df["size"].apply(humanize_bytesize)
                print("Total {} / {}".format(len(df), total[0][0]))
                list_measure = ListMeasure(headers=headers, measures=df[headers].values)
                print(list_measure.to_tabulate())
