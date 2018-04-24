import os

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.models import DataDrive
from ml.clf.measures import ListMeasure
from ml.utils.files import get_models_path, get_models_from_dataset, rm
from ml.ds import Data
from ml.ds import load_metadata

settings = get_settings("ml")

  
def run(args):
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
                if args.info == meta.get("model_name", None) or\
                    args.info == meta.get("group_name", None):
                        for version in meta.get("model", {}).get("versions", ["1"]):
                            model_version_path = os.path.join(
                                settings["checkpoints_path"], clf, model_name, "version."+version, "meta.xmeta")
                            meta_v = load_metadata(model_version_path)
                            measure = "logloss" if not args.measure else args.measure
                            try:
                                scores = meta_v["score"]
                                if scores is not None:
                                    selected_score = scores[measure]
                                    score = selected_score['values'][0]
                                    order_m = order_m or selected_score['reverse']
                                else:
                                    score = None
                            except KeyError:
                                score = None

                            table.append([clf, model_name, version, 
                                        meta.get("group_name", None), score])

                        if args.meta:
                            print(meta)

        headers = ["classif", "model name", "version", "group", 
            "{}".format(measure)]
        order = [False, False, False, False, order_m]
        list_measure = ListMeasure(headers=headers, measures=table, order=order)
        list_measure.print_scores(order_column=measure)
    elif args.rm:
        for model_name in args.rm:
            clf, model_name, version = model_name.split(".")
            name_version = model_name + "." + version
            model_path_meta = os.path.join(settings["checkpoints_path"], clf, name_version, name_version)
            print(DataDrive.read_meta(None, model_path_meta))
            #print("Delete model: {}".format(model_path))
        #rm(model_path)
        print("Done.")
    else:
        headers = ["classif", "model name", "version", "dataset", "group"]
        table = []
        for clf, models_name_v in models_path.items():
            for name_version in models_name_v:
                meta = DataDrive.read_meta(None, os.path.join(
                    settings["checkpoints_path"], clf, name_version, name_version))
                try:
                    if meta is not None:
                        name, version = name_version.split(".")
                        table.append([clf, name, version, meta.get("dataset_name", None),
                                meta.get("group_name", None)])
                except ValueError:
                    pass
        list_measure = ListMeasure(headers=headers, measures=table)
        list_measure.print_scores(order_column="classif")
