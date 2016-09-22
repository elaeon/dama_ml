import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import ml
import argparse
from utils.config import get_settings
from ml.processing import Preprocessing

settings = get_settings("ml")
settings.update(get_settings("numerai"))


def predict(classif, path, label_column):
    import pandas as pd
    import csv

    df = pd.read_csv(path)
    data = df.drop([label_column], axis=1).as_matrix()

    ids = df[label_column].as_matrix()
    predictions = []
    for value, label in zip(list(classif.predict(data, raw=True)), ids):
        predictions.append([str(label), str(value[1])])
    
    with open(settings["predictions_file_path"], "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["t_id", "probability"])
        for row in predictions:
            csvwriter.writerow(row)


def merge_data_labels():
    df = ml.ds.DataSetBuilderFile.merge_data_labels(settings["numerai_test"], 
        settings["numerai_example"], "t_id")
    df.loc[df.probability >= .5, 'probability'] = 1
    df.loc[df.probability < .5, 'probability'] = 0
    df = df.drop(["t_id"], axis=1)
    df.rename(columns={'probability': 'target'}, inplace=True)
    labels = df["target"].as_matrix()
    df = df.drop(["target"], axis=1)
    data = df.as_matrix()
    return data, labels


def build(dataset_name):
    dataset = ml.ds.DataSetBuilderFile(
        dataset_name, 
        dataset_path=settings["dataset_path"], 
        processing_class=Preprocessing,
        train_folder_path=settings["numerai_train"])
    dataset.build_dataset(label_column="target")
    return dataset


#dataset = dataset.subset(.0085)
#dataset.info()
#import GPy
#classif = ml.clf_e.SVC(dataset, check_point_path=check_point_path, pprint=False)
#classif = ml.clf_e.LSTM(dataset, check_point_path=check_point_path, pprint=False, timesteps=7)
#classif = ml.clf_e.GPC(kernel=GPy.kern.sde_StdPeriodic, k_params={"variance":1., "period": 3}, dataset=dataset, check_point_path=check_point_path, pprint=False)
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--build-dataset", help="crea el dataset", action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--predic", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--model-version", type=str)
    args = parser.parse_args()


    if args.build_dataset:
        dataset = build(args.model_name)

    if args.train == 1:
        dataset = ml.ds.DataSetBuilderFile.load_dataset(
            args.model_name, dataset_path=settings["dataset_path"])
        classif = ml.clf.extended.LSTM(dataset=dataset, check_point_path=settings["checkpoints_path"], 
            timesteps=7, model_version=args.model_version)
        classif.train(batch_size=128, num_steps=args.epoch)
        classif.scores()
    elif args.train == 2:
        dataset = ml.ds.DataSetBuilderFile.load_dataset(
            args.model_name, dataset_path=settings["dataset_path"])
        classif = ml.clf.extended.LSTM(dataset=dataset, check_point_path=settings["checkpoints_path"], 
            timesteps=7, model_version=args.model_version)
        classif.train2steps(dataset, valid_size=.1, batch_size=128, 
            num_steps=args.epoch, test_data_labels=merge_data_labels())
        classif.scores()

    if args.predic:
        classif = ml.clf.extended.LSTM(model_name=args.model_name, 
            check_point_path=settings["checkpoints_path"], 
            timesteps=7, model_version=args.model_version)
        classif.scores()
        predict(classif, settings["numerai_test"], "t_id")
