import os

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.clf.generic import DataDrive
from ml.utils.files import get_models_path, delete_file_model

settings = get_settings("ml", filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

  
def run(args):
    models_path = get_models_path(settings["checkpoints_path"])
    if args.info:            
        headers = ["classif", "model name", "version", "group", "score"]
        table = []
        for clf, dataset in models_path.items():
            for name_version in dataset:
                model_path = os.path.join(
                    settings["checkpoints_path"], clf, name_version, name_version)
                group_name = DataDrive.read_meta("group_name", model_path)
                if args.info == group_name:
                    score = DataDrive.read_meta("score", model_path)
                    try:
                        name, version = name_version.split(".")
                        table.append([clf, name, version, group_name, score])
                    except ValueError:
                        pass
        order_table_print(headers, table, "score", reverse=False)
    elif args.rm:
        clf, model_name, version = args.rm.split(".")
        path = delete_file_model(clf, model_name, version, settings["checkpoints_path"])
        print(path)
        print("Done.")
    else:
        headers = ["classif", "model name", "version", "group"]
        table = []
        for clf, dataset in models_path.items():
            for name_version in dataset:
                group_name = DataDrive.read_meta(
                    "group_name", os.path.join(
                        settings["checkpoints_path"], clf, name_version, name_version))
                try:
                    name, version = name_version.split(".")
                    table.append([clf, name, version, group_name])
                except ValueError:
                    pass
        order_table_print(headers, table, "classif", reverse=False)
