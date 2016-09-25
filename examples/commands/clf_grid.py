import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import ml
import numpy as np
from utils.config import get_settings

settings = get_settings("ml")

np.random.seed(1)


def train():
    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    dataset = ml.ds.DataSetBuilder("gpc_test", dataset_path=settings["dataset_path"])
    dataset.build_from_data_labels(X, Y)
    classif = ml.clf.generic.Grid([
        ml.clf.extended.MLP,
        ml.clf.extended.RandomForest,
        ml.clf.extended.SGDClassifier,
        ml.clf.extended.SVC,
        ml.clf.extended.SVGPC,
        ml.clf.extended.GPC],
        dataset=dataset,
        model_version="1",
        check_point_path=settings["checkpoints_path"])
    classif.train(batch_size=128, num_steps=15)
    classif.scores()

def test():
    classif = ml.clf.generic.Grid([
        ml.clf.extended.RandomForest,
        ml.clf.extended.SGDClassifier,
        ml.clf.extended.SVC,
        ml.clf.extended.SVGPC,
        ml.clf.extended.GPC,
        ml.clf.extended.MLP],
        model_name="gpc_test",
        model_version="1", 
        check_point_path=settings["checkpoints_path"])
    classif.scores()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", 
        help="evalua el predictor en base a los datos de prueba", 
        action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-version", type=str)
    args = parser.parse_args()
    if args.train:
        train()
    elif args.test:
        test()
