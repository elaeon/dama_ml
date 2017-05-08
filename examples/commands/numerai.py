import argparse

from ml.ds import DataSetBuilderFile
from ml.utils.config import get_settings
from ml.utils.numeric_functions import le
from ml.processing import Transforms
from ml.clf.extended import w_sklearn
from ml.clf.extended import w_tflearn
from ml.clf import ensemble as clf_ensemble


settings = get_settings("ml")
settings.update(get_settings("numerai"))


def predict(classif, path):
    import pandas as pd
    import csv

    df = pd.read_csv(path)
    ids = df["id"].as_matrix()
    data = df.drop(["target", "id", "era", "data_type"], axis=1).as_matrix()
    predictions = []
    for value, label in zip(list(classif.predict(data, raw=True, chunk_size=258)), ids):
        predictions.append([str(label), str(value[1])])
    
    with open("predictions.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["id", "probability"])
        for row in predictions:
            csvwriter.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    dataset = DataSetBuilderFile(args.dataset_name)
    predict(classif, args.path, "id")
    print("ok")
