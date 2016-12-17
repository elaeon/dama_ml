#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse

from ml.ds import DataSetBuilderFile
from ml.utils.config import get_settings
from ml.utils.numeric_functions import le
from ml.processing import Preprocessing, FiTScaler
from ml.clf.extended import w_sklearn
from ml.clf.extended import w_tflearn
from ml.clf import generic as clf_generic
from ml.clf import ensemble as clf_ensemble


settings = get_settings("ml")
settings.update(get_settings("numerai"))


def predict(classif, path, label_column):
    import pandas as pd
    import csv

    df = pd.read_csv(path)
    data = df.drop([label_column], axis=1).as_matrix()

    ids = df[label_column].as_matrix()
    predictions = []
    for value, label in zip(list(classif.predict(data, raw=True, chunk_size=258)), ids):
        predictions.append([str(label), str(value[1])])
    
    with open(settings["predictions_file_path"], "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["t_id", "probability"])
        for row in predictions:
            csvwriter.writerow(row)


def merge_data_labels(file_path=None):
    df = DataSetBuilderFile.merge_data_labels(settings["numerai_test"], 
        settings["numerai_example"], "t_id")
    df.loc[df.probability >= .5, 'probability'] = 1
    df.loc[df.probability < .5, 'probability'] = 0
    df = df.drop(["t_id"], axis=1)
    df.rename(columns={'probability': 'target'}, inplace=True)
    if file_path is not None:
        df.to_csv(file_path, index=False)
    labels = df["target"].as_matrix()
    df = df.drop(["target"], axis=1)
    data = df.as_matrix()
    return data, labels


def build(dataset_name, transforms=None):
    dataset = DataSetBuilderFile(
        dataset_name, 
        train_folder_path=settings["numerai_train"],
        transforms_global=transforms)
    dataset.build_dataset(label_column="target")
    return dataset


def build2(dataset_name, transforms=None):
    test_data, test_labels = merge_data_labels("/home/sc/test_data/t.csv")
    dataset = DataSetBuilderFile(
        dataset_name, 
        processing_class=Preprocessing,
        train_folder_path=settings["numerai_train"],
        test_folder_path="/home/sc/test_data/t.csv",
        validator="adversarial",
        transforms_global=transforms)
    dataset.build_dataset(label_column="target")
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--build-dataset", help="crea el dataset", action="store_true")
    parser.add_argument("--dataset-name", help="crea el dataset", type=str)
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--ensemble", type=str)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--predict", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()


    if args.build_dataset and args.dataset_name:
        transforms = [(FiTScaler.module_cls_name(), None)]
        #transforms = None
        dataset = build(args.dataset_name, transforms=transforms)

    if args.train:
        dataset = DataSetBuilderFile.load_dataset(args.dataset_name)

        if args.ensemble == "boosting":
            classif = clf_ensemble.Boosting({"0": [
                w_sklearn.ExtraTrees,
                w_tflearn.MLP,
                w_sklearn.RandomForest,
                w_sklearn.SGDClassifier,
                w_sklearn.SVC,
                w_sklearn.LogisticRegression,
                w_sklearn.AdaBoost,
                w_sklearn.GradientBoost]},
                dataset=dataset,
                model_name=args.model_name,
                model_version=args.model_version,
                weights=[3, 1],
                election='best-c',
                num_max_clfs=5)
        elif args.ensemble == "stacking":
            classif = clf_ensemble.Stacking({"0": [
                w_sklearn.ExtraTrees,
                w_tflearn.MLP,
                w_sklearn.RandomForest,
                w_sklearn.SGDClassifier,
                w_sklearn.SVC,
                w_sklearn.LogisticRegression,
                w_sklearn.AdaBoost,
                w_sklearn.GradientBoost]},
                n_splits=3,
                dataset=dataset,
                model_name=args.model_name,
                model_version=args.model_version)
        else:
            classif = clf_ensemble.Bagging(w_tflearn.MLP, {"0": [
                w_sklearn.ExtraTrees,
                w_tflearn.MLP,
                w_sklearn.RandomForest,
                w_sklearn.SGDClassifier,
                w_sklearn.SVC,
                w_sklearn.LogisticRegression,
                w_sklearn.AdaBoost,
                w_sklearn.GradientBoost]},
                dataset=dataset,
                model_name=args.model_name,
                model_version=args.model_version)
        classif.train(batch_size=128, num_steps=args.epoch) # only_voting=True
        classif.scores().print_scores(order_column="logloss")

    if args.predict:
        if args.ensemble == "boosting":
            classif = clf_ensemble.Boosting({},
                model_name=args.model_name,
                model_version=args.model_version)
        elif args.ensemble == "stacking":
            classif = clf_generic.Stacking({},
                model_name=args.model_name,
                model_version=args.model_version)
        else:
            classif = clf_ensemble.Bagging(None, {},
                model_name=args.model_name,
                model_version=args.model_version)
        classif.scores().print_scores(order_column="logloss")
        predict(classif, settings["numerai_test"], "t_id")
        print("Predictions writed in {}".format(settings["predictions_file_path"]))

    if args.plot:
        dataset = DataSetBuilderFile.load_dataset(args.model_name)
        print("DENSITY: ", dataset.density())
        dataset.plot()
