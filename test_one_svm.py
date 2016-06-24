import ml
from utils.config import get_settings
from tabulate import tabulate

settings = get_settings()
dataset_name = "test2"
IMAGE_SIZE = int(settings["image_size"])

def build():
    ds_builder = ml.ds.DataSetBuilder(dataset_name, 
        image_size=IMAGE_SIZE, 
        dataset_path="/home/sc/", 
        train_folder_path=settings["root_data"]+settings["pictures"]+"tickets/train/")
    ds_builder.original_to_images_set(
        [settings["root_data"]+settings["pictures"]+"tickets/numbers/"])
    ds_builder.build_dataset()

def matrix():
    dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, dataset_path="/home/sc/")
    labels_counter = dataset.labels_info()
    table = []
    for label_ref, _ in labels_counter.items():
        labels = labels_counter.keys()
        porcentajes = [label_ref]
        for label, prob in ml.ds.proximity_label(label_ref, labels, dataset):
            porcentajes.append(prob*100)
        table.append(porcentajes)

    print(tabulate(table, labels_counter.keys()))

#dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, dataset_path="/home/sc/")
#label_ref = "2"
#for label, prob in ml.ds.proximity_label(label_ref, ["2"], dataset):
#    print(label_ref, label, prob*100)

matrix()
