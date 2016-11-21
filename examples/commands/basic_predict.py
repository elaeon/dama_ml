import os
import argparse
import ml
import numpy as np
from ml.utils.config import get_settings

settings = get_settings("ml", filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

np.random.seed(1)

def build_dataset_hard(dataset_name="gpc_test_hard", validator="cross"):
    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    Z = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.7*np.random.randn(SIZE, 1)])
    dataset = ml.ds.DataSetBuilder(
         dataset_name, 
        dataset_path=settings["dataset_path"], 
        transforms=[('scale', None)],
        validator=validator)
    dataset.build_dataset(X, Y, test_data=X, test_labels=Z)
    return dataset


def build_dataset_easy(dataset_name="gpc_test_easy",):
    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    dataset = ml.ds.DataSetBuilder(
        dataset_name,
        dataset_path=settings["dataset_path"], 
        transforms=[('scale', None)],
        validator="cross")
    dataset.build_dataset(X, Y)
    return dataset


def train(dataset, model_version):
    classif = ml.clf.extended.SVGPC(
        model_name=dataset.name
        dataset=dataset,
        model_version=model_version,
        check_point_path=settings["checkpoints_path"])
    classif.train(batch_size=128, num_steps=15)
    classif.scores().print_scores(order_column="f1")


def test(model_name, model_version):
    classif = ml.clf.extended.SVGPC(
        model_name=model_name,
        model_version=model_version,
        check_point_path=settings["checkpoints_path"])
    classif.scores().print_scores(order_column="f1")

def predict(model_name, chunk_size, model_version):
    np.random.seed(0)
    DIM = 21
    SIZE = 11
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    classif = ml.clf.extended.SVGPC(
        model_name=model_name,
        model_version=model_version,
        check_point_path=settings["checkpoints_path"])
    print("TARGET", Y)
    #print("VALUE", X)
    print("PREDICTED", list(classif.predict(X, chunk_size=chunk_size)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", 
        help="evalua el predictor en base a los datos de prueba", 
        action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--build-dataset", type=str, help="[cross] [adversarial]")
    parser.add_argument("--dataset-name", type=str, help="dataset name")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--model-version", type=str)
    args = parser.parse_args()
    if args.build_dataset:
        dataset = build_dataset_hard(validator=args.build_dataset, dataset_name=args.dataset_name)
    elif args.train:
        dataset = ml.ds.DataSetBuilder.load_dataset(args.model_name, dataset_path=settings["dataset_path"])
        train(dataset, args.model_version)
    elif args.test:
        test(args.model_name, args.model_version)
    elif args.predict:
        predict(args.model_name, args.chunk_size, args.model_version)
