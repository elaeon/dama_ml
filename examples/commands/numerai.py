import argparse

from ml.ds import DataSetBuilder
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
    
    with open(settings["predictions_file_path"], "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["id", "probability"])
        for row in predictions:
            csvwriter.writerow(row)

def build():
    import pandas as pd
    import numpy as np

    dsb = DataSetBuilder(name="numerai", compression_level=6, validator=None)
    training_data = pd.read_csv(settings["numerai_train"], header=0)
    prediction_data = pd.read_csv(settings["numerai_test"], header=0)

    features = [f for f in list(training_data) if "feature" in f]
    data = training_data[features].as_matrix()
    labels = training_data["target"].as_matrix()
    #test = prediction_data[prediction_data['data_type'] == 'test']
    #test_data = test[features].as_matrix()
    #test_labels = test['target'].as_matrix()
    validation = prediction_data[prediction_data['data_type'] == 'validation']
    validation_data = validation[features].as_matrix()
    validation_labels = validation['target'].as_matrix()

    data = np.vstack((data, validation_data))
    labels = np.append(labels, validation_labels, axis=0)
    #dsb.build_dataset(data, labels, test_data=test_data, test_labels=test_labels,
    #    validation_data=validation_data, validation_labels=validation_labels)

    dsb.build_dataset(data, labels)
    dsb.info()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--build-dataset", action="store_true")
    args = parser.parse_args()

    if args.build_dataset:
        build()
    else:    
        dataset = DataSetBuilderFile(name="numeraiml datasets")
        predict(classif, args.path, "id")
        print("ok")
