from dama.utils.config import get_settings


settings = get_settings("paths")
settings.update(get_settings("vars"))


def run(args):
    from dama.measures import ListMeasure
    from dama.drivers.sqlite import Sqlite
    from dama.utils.core import Login
    from dama.utils.core import Metadata
    login = Login(table=settings["model_tag"])
    driver = Sqlite(login=login, path=settings["metadata_path"], mode="r")

    if args.info:
        pass
    # elif args.rm:
    #    for model_name in args.rm:
    #        clf, model_name, version = model_name.split(".")
    #        name_version = model_name + "." + version
    #        model_path_meta = os.path.join(settings["checkpoints_path"], clf, name_version, name_version)
    #        print(DataDrive.read_meta(None, model_path_meta))
    # print("Delete model: {}".format(model_path))
    # rm(model_path)
    #    print("Done.")
    else:
        from dama.utils.miscellaneous import str2slice
        import sqlite3
        headers = ["from_ds", "name", "group_name", "model", "version", "score_name", "score"]
        page = str2slice(args.items)
        if args.exclude_cols is not None:
            headers = ListMeasure.exclude_columns(headers, args.exclude_cols)
        with Metadata(driver) as metadata:
            try:
                total = metadata.query("SELECT COUNT(*) FROM %s WHERE is_valid=?" % login.table, (True, ))
            except sqlite3.OperationalError as e:
                print(e)
            else:
                data = metadata.data()
                if args.score_name is None and args.group_name is None:
                    data = data[data["is_valid"] == True][page]
                elif args.score_name is not None and args.group_name is None:
                    data = data[(data["is_valid"] == True) & (data["score_name"] == args.score_name)][page]
                elif args.score_name is None and args.group_name is not None:
                    data = data[(data["is_valid"] == True) & (data["group_name"] == args.group_name)][page]
                else:
                    data = data[(data["is_valid"] == True) & (data["score_name"] == args.score_name) &\
                                (data["group_name"] == args.group_name)][page]
                df = data.to_df()
                df.rename(columns={"model_module": "model"}, inplace=True)
                print("Using metadata {}".format(metadata.driver.url))
                print("Total {} / {}".format(len(df), total[0][0]))
                list_measure = ListMeasure(headers=headers, measures=df[headers].values)
                print(list_measure.to_tabulate())
