import ml
from utils.config import get_settings
from tabulate import tabulate
from skimage import io
from skimage import exposure
from sklearn import preprocessing

settings = get_settings()
dataset_name = "test"
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

def test_train(dataset):
    #classif = ml.clf.SVCFace(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.Binary(dataset, check_point_path=check_point_path, pprint=False)
    classif = ml.clf.TfLTensor(dataset, num_features=21, check_point_path=check_point_path, pprint=False)
    classif.train(batch_size=200, num_steps=100)
    dt = ml.clf.ClassifTest()
    dt.classif_test(classif, "f1")
    #classifs = {
    #    "SCV": {"name": ml.clf.SVCFace, 
    #        "params": {"check_point_path": check_point_path, "pprint":False}}, 
    #    "TENSOR": {"name": ml.clf.TfLTensor, 
    #        "params": {"check_point_path": check_point_path, "num_features": 21, "pprint":False}}}
    #dt.dataset_test(classifs, dataset, "f1")

def predict(dataset, path, label_column):
    import pandas as pd
    classif = ml.clf.SVCFace(dataset, check_point_path=check_point_path, pprint=False)
    #classif = ml.clf.TfLTensor(dataset, num_features=21, 
    #    check_point_path=check_point_path, pprint=False)
    df = pd.read_csv(path)
    dataset = df.drop([label_column], axis=1).as_matrix()
    ids = df[label_column].as_matrix()
    #for predic, label in zip(list(classif.predict(dataset))[:20], ids[:20]):
    #    print(predic, label)
    for data, label in zip(dataset, ids)[:20]:
        print(list(classif.predict([data])), label)

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
test_train(dataset)
#predict(dataset, "/home/sc/test_data/numerai_datasets/numerai_tournament_data.csv", "t_id")

