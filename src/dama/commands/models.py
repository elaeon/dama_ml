import os

from dama.utils.config import get_settings


settings = get_settings("paths")
settings.update(get_settings("vars"))

  
def run(args):
    from dama.measures import ListMeasure
    from dama.data.drivers.sqlite import Sqlite
    from dama.utils.core import Login, Metadata
    login = Login(table=settings["model_tag"])
    driver = Sqlite(login=login, path=settings["metadata_path"], mode="r")

    if args.info:
        pass
    else:
        from dama.utils.core import Login, Metadata
        from dama.utils.miscellaneous import str2slice
        import sqlite3
        headers = ["hash", "name", "driver", "group name", "size", "num groups", "datetime UTC"]
        page = str2slice(args.items)
        with Metadata(driver) as metadata:
            try:
                total = metadata.query("SELECT COUNT(*) FROM %s WHERE is_valid=True" % login.table, ())
            except sqlite3.OperationalError as e:
                print(e)
            else:
                data = metadata.data()
                data = data[data["is_valid"] == True][page]
                df = data.to_df()
                df.rename(columns={"timestamp": "datetime UTC", "group_name": "group name",
                                   "num_groups": "num groups", "driver_name": "driver"}, inplace=True)
                df["size"] = df["size"].apply(humanize_bytesize)
                print("Total {} / {}".format(len(df), total[0][0]))
                list_measure = ListMeasure(headers=headers, measures=df[headers].values)
                print(list_measure.to_tabulate())
        #headers = ["classif", "model name", "version", "group",
        #    "{}".format(measure)]
        #order = [False, False, False, False, order_m]
        #list_measure = ListMeasure(headers=headers, measures=table, order=order)
        #print(list_measure.to_tabulate(order_column=measure))
    #elif args.rm:
    #    for model_name in args.rm:
    #        clf, model_name, version = model_name.split(".")
    #        name_version = model_name + "." + version
    #        model_path_meta = os.path.join(settings["checkpoints_path"], clf, name_version, name_version)
    #        print(DataDrive.read_meta(None, model_path_meta))
            #print("Delete model: {}".format(model_path))
        #rm(model_path)
    #    print("Done.")
    #else:
    #    headers = ["classif", "model name", "version", "dataset", "group"]
    #    table = []
    #    for clf in os.listdir(settings["checkpoints_path"]):
    #        for model_name in os.listdir(os.path.join(settings["checkpoints_path"], clf)):
    #            meta = load_metadata(os.path.join(settings["checkpoints_path"],
    #                clf, model_name, "meta.xmeta"))
    #            if "versions" in meta:
    #                for version in meta["versions"]:
    #                    path = os.path.join(settings["checkpoints_path"], clf,
    #                        model_name, "version.{}".format(version), "meta.xmeta")
    #                    meta = load_metadata(path)
    #                    table.append([clf, model_name, version, meta.get("dataset_name", None),
    #                            meta.get("group_name", None)])
    #    list_measure = ListMeasure(headers=headers, measures=table)
    #    list_measure.print_scores(order_column="classif")
