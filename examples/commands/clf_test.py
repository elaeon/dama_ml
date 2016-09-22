import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import ml
import numpy as np
from utils.config import get_settings

settings = get_settings("ml")

np.random.seed(1)


def mll():
    DIM = 21
    SIZE = 1000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    dataset = ml.ds.DataSetBuilder("gpc_test", dataset_path=settings["dataset_path"])
    dataset.build_from_data_labels(X, Y)
    classif = ml.clf.extended.GPC(dataset=dataset, check_point_path=settings["checkpoints_path"])
    #classif = ml.clf.extended.RandomForest(dataset=dataset, check_point_path="/home/sc/ml_data/checkpoints/")
    #classif = ml.clf.extended.SGDClassifier(dataset=dataset, check_point_path="/home/sc/ml_data/checkpoints/")
    #classif = ml.clf.extended.SVC(dataset=dataset, check_point_path="/home/sc/ml_data/checkpoints/")
    #classif = ml.clf.extended.SVGPC(optimizer='Adadelta', dataset=dataset, check_point_path="/home/sc/ml_data/checkpoints/")
    classif.train(batch_size=128, num_steps=2)
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
    mll()
