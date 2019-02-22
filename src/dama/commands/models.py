import os

from dama.utils.config import get_settings
from dama.utils.files import get_models_path


settings = get_settings("paths")

  
def run(args):
    from dama.measures import ListMeasure
    models_path = get_models_path(settings["checkpoints_path"])
    if args.info:
        table = []
        measure = None
        order_m = False
        for clf, models_name in models_path.items():
            for model_name in models_name:
                model_path = os.path.join(
                    settings["checkpoints_path"], clf, model_name, "meta.xmeta")
                meta = load_metadata(model_path)
                if meta is None:
                    continue
                if args.info == meta.get("model_name", None) or\
                    args.info == meta.get("group_name", None):
                    for version in meta.get("versions", ["1"]):
                        model_version_path = os.path.join(
                            settings["checkpoints_path"], clf, model_name, "version."+version, "meta.xmeta")
                        meta_v = load_metadata(model_version_path)
                        measure = "logloss" if not args.measure else args.measure
                        try:
                            scores = meta_v["score"]
                            if scores is not None:
                                selected_score = scores[measure]
                                score = selected_score['values'][0]
                                order_m = order_m or selected_score.get('reverse', True)
                            else:
                                continue
                        except KeyError:
                            continue

                        table.append([clf, model_name, version, 
                                    meta.get("group_name", None), score])

                    if args.meta:
                        print(meta)

        headers = ["classif", "model name", "version", "group", 
            "{}".format(measure)]
        order = [False, False, False, False, order_m]
        list_measure = ListMeasure(headers=headers, measures=table, order=order)
        print(list_measure.to_tabulate(order_column=measure))
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
