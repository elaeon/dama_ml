import os
import argparse
import numpy as np
from ml.ds import DataSetBuilder
from ml.clf.extended.w_gpy import SVGPC

np.random.seed(1)

def build_dataset_hard(dataset_name="gpc_test_hard", validator="cross",
                    compression_level=0):
    DIM = 21
    SIZE = 100000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    #Z = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.7*np.random.randn(SIZE, 1)])
    dataset = DataSetBuilder(
        dataset_name, 
        validator=validator,
        compression_level=compression_level,
        ltype='int')
    dataset.build_dataset(X, Y)#, test_data=None, test_labels=None)
    return dataset


def train(dataset, model_name, model_version, epoch):
    classif = SVGPC(
        model_name=model_name,
        dataset=dataset,
        model_version=model_version,
        group_name="basic")
    classif.train(batch_size=128, num_steps=epoch)
    classif.scores().print_scores(order_column="f1")


def test(model_name, model_version):
    classif = SVGPC(
        model_name=model_name,
        model_version=model_version)
    classif.scores().print_scores(order_column="f1")


def predict(model_name, chunk_size, model_version):
    from ml.clf.wrappers import Measure

    np.random.seed(0)
    DIM = 21
    SIZE = 10000
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    classif = SVGPC(
        model_name=model_name,
        model_version=model_version)
    predictions = np.asarray(list(classif.predict(X, chunk_size=chunk_size)))
    print("{} elems SCORE".format(SIZE), Measure(predictions, Y, classif.numerical_labels2classes).accuracy())


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
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--compression-level", type=int, default=0)
    args = parser.parse_args()
    if args.build_dataset:
        dataset = build_dataset_hard(validator=args.build_dataset, 
            dataset_name=args.dataset_name, compression_level=args.compression_level)
    elif args.train:
        dataset = DataSetBuilder(args.dataset_name)
        train(dataset, args.model_name, args.model_version, args.epoch)
    elif args.test:
        test(args.model_name, args.model_version)
    elif args.predict:
        predict(args.model_name, args.chunk_size, args.model_version)
