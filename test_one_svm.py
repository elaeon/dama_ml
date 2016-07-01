import ml
from utils.config import get_settings
from tabulate import tabulate
from skimage import io
from skimage import exposure
from sklearn import preprocessing

settings = get_settings()
dataset_name = "numeraiv2"
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

def test_train(dataset, train=False):
    #classif = ml.clf.SVCFace(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.Binary(dataset, check_point_path=check_point_path, pprint=False)
    classif = ml.clf.TfLTensor(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.RandomForest(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.LogisticRegression(dataset, check_point_path=check_point_path, pprint=False)

    if train is True:
        classif.train(batch_size=120, num_steps=50)
    dt = ml.clf.ClassifTest(logloss=True)
    dt.classif_test(classif, "f1")
    #classifs = {
    #    "SCV": {"name": ml.clf.SVCFace, 
    #        "params": {"check_point_path": check_point_path, "pprint":False}}, 
    #    "TENSOR": {"name": ml.clf.TfLTensor, 
    #        "params": {"check_point_path": check_point_path, "pprint":False}},
    #    "FOREST": {"name": ml.clf.RandomForest,
    #        "params": {"check_point_path": check_point_path, "pprint":False}}}
    #dt.dataset_test(classifs, dataset, "f1")

def predict(dataset, path, label_column):
    import pandas as pd
    import csv
    #classif = ml.clf.SVCFace(dataset, check_point_path=check_point_path, pprint=False)
    classif = ml.clf.TfLTensor(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.RandomForest(dataset, check_point_path=check_point_path, pprint=False)
    df = pd.read_csv(path)
    dataset = df.drop([label_column], axis=1).as_matrix()
    ids = df[label_column].as_matrix()
    #for predic, label in zip(list(classif.predict(dataset))[:20], ids[:20]):
    #    print(predic, label)
    predictions = []
    for value, label in zip(list(classif.predict(dataset, raw=True)), ids):
        predictions.append([str(label), str(value[1])])
        #break
    
    with open("/home/sc/predictions.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["t_id", "probability"])
        for row in predictions:
            csvwriter.writerow(row)

def pca(dataset):
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA, RandomizedPCA, KernelPCA
    #reg = LinearRegression()
    #train_dataset, train_labels = dataset.train_dataset, dataset.train_labels
    #values = []
    #for n_components in range(3, 22):
    #    pca = PCA(n_components=n_components)
    #    train_dataset_ = pca.fit_transform(train_dataset)
    #    print(train_dataset_.shape)
    #    reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")#"newton-cg")
    #    reg = reg.fit(train_dataset_, train_labels)
    #    score = reg.score(train_dataset_, train_labels)
    #    print(score)
    #    values.append((score, n_components))
    #p, c = max(values, key=lambda x: x[0])
    #pca = PCA(n_components=c)
    c = 20
    n_dataset = ml.ds.DataSetBuilder(dataset.name)
    n_dataset.train_dataset = PCA(n_components=c).fit_transform(dataset.train_dataset)
    n_dataset.test_dataset = PCA(n_components=c).fit_transform(dataset.test_dataset)
    n_dataset.valid_dataset = PCA(n_components=c).fit_transform(dataset.valid_dataset)
    n_dataset.train_labels = dataset.train_labels
    n_dataset.test_labels = dataset.test_labels
    n_dataset.valid_labels = dataset.valid_labels
    return n_dataset
    #predictions = reg.predict(train_dataset)

    ### your code goes here
    #limit = 10
    #print(train_labels[:limit])
    #print(predictions[:limit])

def graph(dataset):
    import numpy as np
    import seaborn as sns
    sns.set(color_codes=True)
    classif = ml.clf.TfLTensor(dataset, check_point_path=check_point_path, pprint=False)
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
    #c1 = df_c[df_c["target"] == 1].sample(frac=0.1, replace=True).drop(["target"], axis=1).as_matrix()
    #c_G = df_G[df_G["target"] == 1].sample(frac=0.01, replace=True).drop(["target"], axis=1).as_matrix()
    #sns.tsplot(e0, color="r", err_style="boot_traces")
    #sns.tsplot(e1, color="b", err_style="boot_traces")
    #sns.tsplot(c0, color="g", err_style="boot_traces")
    #sns.tsplot(c1, color="m", err_style="boot_traces")
    #sns.tsplot(c_G, color="y", err_style="boot_traces")
    sns.kdeplot(df_e[df_e["target"] == 1]["c0"], color="r", shade=False)
    sns.kdeplot(df_c[df_c["target"] == 0]["c0"], color="g", shade=False)
    sns.plt.show()

#build()
#matrix()
#dataset = ml.ds.DataSetBuilderFile(dataset_name, dataset_path=dataset_path)
#dataset.build_dataset("/home/sc/test_data/numerai_datasets/numerai_training_data.csv", "target")
"""
         0        1
--  ------  -------
 0  64.991  39.8907
 1  40.42   64.5937
"""
dataset = ml.ds.DataSetBuilderFile.load_dataset(dataset_name, dataset_path=dataset_path)
#n_dataset = pca(dataset)
#outlier(dataset)
#test_train(dataset, train=True)
graph(dataset)
#predict(dataset, "/home/sc/test_data/numerai_datasets/numerai_tournament_data.csv", "t_id")

