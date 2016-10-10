import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import ml
import numpy as np
from utils.config import get_settings

settings = get_settings("ml")

np.random.seed(1)

def build_dataset_hard(validator="cross"):
    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    Z = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.7*np.random.randn(SIZE, 1)])
    dataset = ml.ds.DataSetBuilder(
        "gpc_test_hard", 
        dataset_path=settings["dataset_path"], 
        transforms=[("variance_threshold", 0.0), ('scale', None)],
        validator=validator)
    dataset.build_dataset(X, Y, test_data=X, test_labels=Z)
    return dataset


def build_dataset_easy():
    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    dataset = ml.ds.DataSetBuilder(
        "gpc_test_easy", 
        dataset_path=settings["dataset_path"], 
        transforms=[("variance_threshold", 0.0), ('scale', None)],
        validator="cross")
    dataset.build_dataset(X, Y)
    return dataset


def train(dataset):
    classif = ml.clf.generic.Grid([
        ml.clf.extended.ExtraTrees,
        ml.clf.extended.MLP,
        ml.clf.extended.RandomForest,
        ml.clf.extended.SGDClassifier,
        ml.clf.extended.SVC,
    #    ml.clf.extended.SVGPC,
    #    ml.clf.extended.GPC,
        ml.clf.extended.LogisticRegression,
        ml.clf.extended.AdaBoost,
        ml.clf.extended.GradientBoost,
        ml.clf.extended.Voting],
        dataset=dataset,
        model_version="1",
        check_point_path=settings["checkpoints_path"])
    classif.train(batch_size=128, num_steps=15)
    classif.scores()


def test(model_name):
    classif = ml.clf.generic.Grid([
        ml.clf.extended.RandomForest,
        ml.clf.extended.SGDClassifier,
        ml.clf.extended.SVC,
        #ml.clf.extended.SVGPC,
        #ml.clf.extended.GPC,
        ml.clf.extended.MLP,
        ml.clf.extended.LogisticRegression,
        ml.clf.extended.ExtraTrees,
        ml.clf.extended.AdaBoost,
        ml.clf.extended.GradientBoost,
        ml.clf.extended.Voting],
        model_name=model_name,
        model_version="1", 
        check_point_path=settings["checkpoints_path"])
    #classif.confusion_matrix().print_matrix(classif.base_labels)
    classif.scores()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", 
        help="evalua el predictor en base a los datos de prueba", 
        action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--model-name", type=str)
    #parser.add_argument("--model-version", type=str)
    args = parser.parse_args()
    if args.build_dataset:
        dataset = build_dataset_hard(validator="adversarial")
    elif args.train:
        dataset = ml.ds.DataSetBuilder.load_dataset(args.model_name, dataset_path=settings["dataset_path"])
        train(dataset)
    elif args.test:
        test(args.model_name)
