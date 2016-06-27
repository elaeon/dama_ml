import ml
from utils.config import get_settings
from tabulate import tabulate
from skimage import io
from skimage import exposure
from sklearn import preprocessing

settings = get_settings()
dataset_name = "test"
dataset_path = "/home/alejandro/"
IMAGE_SIZE = int(settings["image_size"])

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

#build()
#matrix()
#dataset, labels = ml.ds.DataSetBuilder.csv2dataset("/home/alejandro/Descargas/numerai_datasets/numerai_training_data.csv", "target")
#print(dataset.shape, labels)
#dataset = ml.ds.DataSetBuilderFile(dataset_name, dataset_path=dataset_path)
#dataset.build_dataset("/home/alejandro/Descargas/numerai_datasets/numerai_training_data.csv", "target")
"""
         0        1
--  ------  -------
 0  64.991  39.8907
 1  40.42   64.5937
"""
dataset = ml.ds.DataSetBuilderFile.load_dataset(dataset_name, dataset_path=dataset_path)
#print(dataset.is_binary())
#classif = ml.clf.SVCFace(dataset, check_point_path="/home/alejandro/ml_data/checkpoints/", pprint=False)
#classif = ml.clf.Binary(dataset, check_point_path="/home/alejandro/ml_data/checkpoints/", pprint=False)
classif = ml.clf.TfLTensor(dataset, num_features=21, check_point_path="/home/alejandro/ml_data/checkpoints/", pprint=False)
classif.train(num_steps=10)
dt = ml.clf.ClassifTest()
dt.classif_test(classif, "f1")
