import os

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.clf.wrappers import DataDrive
from ml.clf.measures import ListMeasure
from ml.utils.files import get_models_path, delete_file_model

settings = get_settings("ml")

  
def run(args):
    models_path = get_models_path(settings["checkpoints_path"])
    if args.info:
        table = []
        measure = None
        order_m = False
        for clf, dataset in models_path.items():
            for name_version in dataset:
                model_path = os.path.join(
                    settings["checkpoints_path"], clf, name_version, name_version)
                meta = DataDrive.read_meta(None, model_path)
                if args.info == meta.get("group_name", None) or\
                    args.info == meta.get("model_name", None):
                    measure = "logloss" if not args.measure else args.measure
                    scores = DataDrive.read_meta("score", model_path)
                    if scores is not None:
                        score_ = scores.get(measure, None)
                        if isinstance(score_, dict):
                            score = score_['values'][0]
                            order_m = score_['reverse']
                    else:
                        score = None

                    try:
                        name, version = name_version.split(".")
                        table.append([clf, name, version, meta.get("dataset_name", None),
                                    meta.get("group_name", None), score])
                    except ValueError:
                        pass

                    if args.meta:
                        print(meta)        
        headers = ["classif", "model name", "version", "dataset", "group", 
            "{}".format(measure)]
        order = [False, False, False, False, False, order_m]
        list_measure = ListMeasure(headers=headers, measures=table, order=order)
        list_measure.print_scores(order_column=measure)
    elif args.rm:
        #models_path = get_models_from_dataset(dataset, settings["checkpoints_path"])
        #    print("Dataset: {}".format(dataset.url()))
            #rm(dataset.url())
            #for model_path, _ in models_path:
            #    print("Model: {}".format(model_path))
                #rm(model_path)
        clf, model_name, version = args.rm.split(".")
        path = delete_file_model(clf, model_name, version, settings["checkpoints_path"])
        print(path)
        print("Done.")
    else:
        headers = ["classif", "model name", "version", "dataset", "group"]
        table = []
        for clf, dataset in models_path.items():
            for name_version in dataset:
                meta = DataDrive.read_meta(None, os.path.join(
                    settings["checkpoints_path"], clf, name_version, name_version))
                try:
                    name, version = name_version.split(".")
                    table.append([clf, name, version, meta.get("dataset_name", None),
                                meta.get("group_name", None)])
                except ValueError:
                    pass
        list_measure = ListMeasure(headers=headers, measures=table)
        list_measure.print_scores(order_column="classif")
