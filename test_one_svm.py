import ml
from utils.config import get_settings
from tabulate import tabulate
from skimage import io
from skimage import exposure
from sklearn import preprocessing

from ml.processing import Preprocessing

settings = get_settings()
dataset_name = "numeraiv3"
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


class TestClf(ml.clf.SKLP):
    def __init__(self, avg0, avg1, favg0, favg1, *args, **kwargs):
        self.favg0 = favg0
        self.favg1 = favg1
        self.avg0 = avg0
        self.avg1 = avg1
        super(TestClf, self).__init__(*args, **kwargs)
        
    def prob(self, mavg0, mavg1, dataset_data):
        from collections import Counter
        for row in dataset_data:
            d_0 = (abs(avg0 - data) for avg0, data in zip(mavg0, row)) 
            d_1 = (abs(avg1 - data) for avg1, data in zip(mavg1, row))
            counter = Counter((int(c0 <= c1) for c0, c1 in zip(d_0, d_1)))
            p1 = float(counter[1]) / (counter[0] + counter[1])
            yield p1

    def prepare_model(self):
        pass

    def _predict(self, data, raw=True, transform=True):
        if transform is True:
            data = self.transform_shape(self.dataset.processing(data, 'global'))
        lecture_a = [p1 for p1 in self.prob(self.avg0, self.avg1, data)]
        lecture_b = [p1 for p1 in self.prob(self.favg0, self.favg1, data)]
        for e1, e2 in zip(lecture_a, lecture_b):
            p = .35*e1 + .55*e2
            yield [1 - p, p]

    def train(self, batch_size=0, num_steps=0):
        pass
        

def test_train(classif, train=False, num_steps=10):
    #classif = ml.clf.SVC(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.Binary(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf_e.MLP(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.RandomForest(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf_e.LogisticRegression(dataset, check_point_path=check_point_path, pprint=False)

    if train is True:
        classif.train(batch_size=128, num_steps=num_steps)
    classif.print_score()
    #classifs = {
    #    "SCV": {"name": ml.clf.SVCFace, 
    #        "params": {"check_point_path": check_point_path, "pprint":False}}, 
    #    "TENSOR": {"name": ml.clf.TfLTensor, 
    #        "params": {"check_point_path": check_point_path, "pprint":False}},
    #    "FOREST": {"name": ml.clf.RandomForest,
    #        "params": {"check_point_path": check_point_path, "pprint":False}}}
    #dt.dataset_test(classifs, dataset, "f1")

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

def test2(dataset, classif):
    dataset_e, _, labels_e = classif.erroneous_clf()
    dataset_c, _, labels_c = classif.correct_clf()
    df_e = ml.ds.DataSetBuilder.to_DF(dataset_e, labels_e)
    df_c = ml.ds.DataSetBuilder.to_DF(dataset_c, labels_c)

    e0 = df_e[df_e["target"] == 1].drop(["target"], axis=1).as_matrix().mean(axis=0)
    e1 = df_e[df_e["target"] == 0].drop(["target"], axis=1).as_matrix().mean(axis=0)
    c0 = df_c[df_c["target"] == 0].drop(["target"], axis=1).as_matrix().mean(axis=0)
    c1 = df_c[df_c["target"] == 1].drop(["target"], axis=1).as_matrix().mean(axis=0)

    t_clf = TestClf(c0, c1, e0, e1, dataset, check_point_path=check_point_path, pprint=False)
    t_clf.print_score()
    return t_clf


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
    #e1 = df_e[df_e["target"] == 0].sample(frac=0.03, replace=True).drop(["target"], axis=1).as_matrix()
    c0 = df_c[df_c["target"] == 0].sample(frac=0.1, replace=True).drop(["target"], axis=1).as_matrix()
    print(e0.var(axis=0))
    print(e0.mean(axis=0))
    print(e0.std(axis=0))
    print(c0.var(axis=0))
    print(c0.mean(axis=0))
    print(c0.std(axis=0))
    #c1 = df_c[df_c["target"] == 1].sample(frac=0.1, replace=True).drop(["target"], axis=1).as_matrix()
    #c_G = df_G[df_G["target"] == 1].sample(frac=0.01, replace=True).drop(["target"], axis=1).as_matrix()
    sns.tsplot(e0, color="r")#, err_style="boot_traces")
    #sns.tsplot(e1, color="b", err_style="boot_traces")
    sns.tsplot(c0, color="g")#, err_style="boot_traces")
    #sns.tsplot(c1, color="m", err_style="boot_traces")
    #sns.tsplot(c_G, color="y", err_style="boot_traces")
    #sns.kdeplot(df_e[df_e["target"] == 1]["c0"], color="r", shade=False)
    #sns.kdeplot(df_c[df_c["target"] == 0]["c0"], color="g", shade=False)
    sns.plt.show()

    #cr0 = np.correlate(c0[0], c0[0], mode='full')
    #cr0 = cr0[cr0.size/2:]
    #cr1 = np.correlate(c1[0], c1[0], mode='full')
    #cr1 = cr1[cr1.size/2:]
    #er0 = np.correlate(e0[0], e0[0], mode='full')
    #er0 = er0[er0.size/2:]
    #print(np.dot(cr0,er0))
    #print(np.dot(cr1,er0))


def normal():
    import pandas as pd
    import seaborn as sns
    df = pd.read_csv("/home/sc/test_data/numerai_datasets/numerai_training_data.csv")
    df2 = df.drop(["target"], axis=1)
    df["total"] = df2.sum(axis=1)
    sns.kdeplot(df[df["target"] == 1]["total"], label="1")
    sns.kdeplot(df[df["target"] == 0]["total"], label="0")
    sns.plt.show()


#build()
#matrix()
#dataset = ml.ds.DataSetBuilderFile(dataset_name, dataset_path=dataset_path, processing_class=Preprocessing)
#dataset.build_dataset("/home/sc/test_data/numerai_datasets/numerai_training_data.csv", "target")
"""
         0        1
--  ------  -------
 0  64.991  39.8907
 1  40.42   64.5937
"""
dataset = ml.ds.DataSetBuilderFile.load_dataset(dataset_name, dataset_path=dataset_path, processing_class=Preprocessing)
classif = ml.clf_e.LSTM(dataset, check_point_path=check_point_path, pprint=False, timesteps=7)
#classif = ml.clf_e.MLP(dataset, check_point_path=check_point_path, pprint=False)
#test_train(classif, train=True, num_steps=20)
classif_t = test2(dataset, classif)
#graph(dataset)
#normal()
#import pandas as pd
#v,l = dataset.new_validation_dataset(pd.read_csv("/home/sc/test_data/numerai_datasets/numerai_tournament_data.csv"), pd.read_csv("/home/sc/test_data/numerai_datasets/example_predictions.csv"), pd.read_csv("/home/sc/predictions.csv"), "t_id", valid_size=.2)
#dataset.valid_data = v
#dataset.valid_labels = l
#dataset.save()
#classif = ml.clf_e.LSTM(dataset, check_point_path=check_point_path, pprint=False, timesteps=7)
#test_train(classif, train=True, num_steps=10)
#predict(classif_t, "/home/sc/test_data/numerai_datasets/numerai_tournament_data.csv", "t_id")
