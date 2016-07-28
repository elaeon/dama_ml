import ml
from utils.config import get_settings
from tabulate import tabulate
from skimage import io
from skimage import exposure
from sklearn import preprocessing

from ml.processing import Preprocessing

settings = get_settings()
dataset_name = "numeraiv1"
dataset_path = "/home/sc/"
IMAGE_SIZE = int(settings["image_size"])
check_point_path = "/home/sc/ml_data/checkpoints/"

def build():
    ds_builder = ml.ds.DataSetBuilder(dataset_name, 
        image_size=IMAGE_SIZE, 
        dataset_path=dataset_path, 
        train_folder_path=settings["root_data"]+settings["pictures"]+"tickets/train/")
    ds_builder.original_to_images_set(
        [settings["root_data"]+settings["pictures"]+"tickets/numbers/"])
    ds_builder.build_dataset()

def matrix():
    dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, dataset_path=dataset_path)
    labels_counter = dataset.labels_info()
    table = []
    for label_ref, _ in labels_counter.items():
        labels = labels_counter.keys()
        porcentajes = [label_ref]
        for label, prob in ml.ds.proximity_label(label_ref, labels, dataset):
            porcentajes.append(prob*100)
        table.append(porcentajes)

    print(tabulate(table, labels_counter.keys()))

def test():
    dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, dataset_path=dataset_path)
    label_ref = "$"
    #for label, prob in ml.ds.proximity_label(label_ref, ["1"], dataset):
    #    print(label_ref, label, prob*100)
    count = 0
    for data, result in ml.ds.proximity_dataset(label_ref, [label_ref], dataset):
        count += 1
        data_n = exposure.rescale_intensity(data, in_range=(-1, 1))
        #min_max = preprocessing.MinMaxScaler(feature_range=(0, .99))
        #data_n = min_max.fit_transform(data)
        io.imsave("/tmp/img2/{}.jpg".format(count), data_n)
        #break


def predict(classif, path, label_column):
    import pandas as pd
    import csv

    df = pd.read_csv(path)
    data = df.drop([label_column], axis=1).as_matrix()

    ids = df[label_column].as_matrix()
    predictions = []
    for value, label in zip(list(classif.predict(data, raw=True)), ids):
        predictions.append([str(label), str(value[1])])
    
    with open("/home/sc/predictions.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["t_id", "probability"])
        for row in predictions:
            csvwriter.writerow(row)


def graph(dataset):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(color_codes=True)
    classif = ml.clf_e.LSTM(dataset, check_point_path=check_point_path, pprint=False, timesteps=7)
    dataset_e, _, labels_e = classif.erroneous_clf()
    dataset_c, _, labels_c = classif.correct_clf()
    df_e = ml.ds.DataSetBuilder.to_DF(dataset_e, labels_e)
    df_c = ml.ds.DataSetBuilder.to_DF(dataset_c, labels_c)
    #df_G = dataset.to_df()
    #for index in range(0, 10):
    #    sns.distplot(dataset.dataset[index])
    #    sns.kdeplot(dataset.dataset[index], shade=False);
    #sns.pairplot(dataset.dataset);
    #sns.distplot(df[df[21] == 1][0])

    e0 = df_e[df_e["target"] == 1].sample(frac=0.1, replace=True).drop(["target"], axis=1).as_matrix()
    e1 = df_e[df_e["target"] == 0].sample(frac=0.1, replace=True).drop(["target"], axis=1).as_matrix()
    c0 = df_c[df_c["target"] == 0].sample(frac=0.1, replace=True).drop(["target"], axis=1).as_matrix()
    c1 = df_c[df_c["target"] == 1].sample(frac=0.1, replace=True).drop(["target"], axis=1).as_matrix()
    #c_G = df_G[df_G["target"] == 1].sample(frac=0.01, replace=True).drop(["target"], axis=1).as_matrix()
    sns.tsplot(e0, color="g")#, err_style="boot_traces")
    sns.tsplot(c0, color="r")
    sns.tsplot(c1, color="g")#, err_style="boot_traces")
    sns.tsplot(e1, color="r")#, err_style="boot_traces")
    #sns.kdeplot(df_e[df_e["target"] == 1]["c0"], color="r", shade=False)
    #sns.kdeplot(df_c[df_c["target"] == 0]["c0"], color="g", shade=False)
    sns.plt.show()


def f3dplot(x, y, z, x1, y1, z1):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.scatter(x1, y1, z1, c='b', marker='x')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def new_df():
    df = ml.ds.DataSetBuilderFile.merge_data_labels(
        "/home/sc/test_data/numerai_datasets/numerai_tournament_data.csv", 
        "/home/sc/test_data/numerai_datasets/example_predictions.csv", "t_id")
    df.loc[df.probability >= .5, 'probability'] = 1
    df.loc[df.probability < .5, 'probability'] = 0
    df = df.drop(["t_id"], axis=1)
    df.rename(columns={'probability': 'target'}, inplace=True)
    labels = df["target"].as_matrix()
    df = df.drop(["target"], axis=1)
    data = df.as_matrix()
    test_data_labels = [data, labels]
    return data, labels

test_data_labels = new_df()
#test_data_labels = None
#df.to_csv("/home/sc/test_data/numerai_datasets/numerai_tournament_data_c.csv", index=False)


#dataset = ml.ds.DataSetBuilderFile(
#dataset_name, 
#dataset_path=dataset_path, 
#processing_class=Preprocessing,
#train_folder_path="/home/sc/test_data/numerai_datasets/numerai_training_data.csv")
#test_folder_path="/home/sc/test_data/numerai_datasets/numerai_tournament_data_c.csv")
#dataset.build_dataset(label_column="target")
"""
         0        1
--  ------  -------
 0  64.991  39.8907
 1  40.42   64.5937
"""
dataset = ml.ds.DataSetBuilderFile.load_dataset(dataset_name, dataset_path=dataset_path, processing_class=Preprocessing)
classif = ml.clf_e.LSTM(dataset, check_point_path=check_point_path, pprint=False, timesteps=7)
#classif.train(batch_size=128, num_steps=1)
classif.train2steps(dataset, valid_size=.2, batch_size=128, num_steps=5, test_data_labels=test_data_labels)
classif = ml.clf_e.LSTM(dataset, check_point_path=check_point_path, pprint=False, timesteps=7)
classif.print_score()
#predict(classif, "/home/sc/test_data/numerai_datasets/numerai_tournament_data.csv", "t_id")
