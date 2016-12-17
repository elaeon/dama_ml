import os
import shutil
import ntpath
import datetime

from skimage import io
from ml.clf.generic import DataDrive
from ml.processing import PreprocessingImage

def filename_from_path(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def build_tickets_processed(transforms, settings, PICTURES):
    tickets_processed_url = settings["tickets_processed"]
    if not os.path.exists(tickets_processed_url):
        os.makedirs(tickets_processed_url)
    for path in [os.path.join(settings["tickets"], f) for f in PICTURES]:
        name = path.split("/").pop()
        image = io.imread(path)
        image = PreprocessingImage(image, transforms).pipeline()
        d_path = os.path.join(tickets_processed_url, name)
        io.imsave(d_path, image)


def delete_tickets_processed(settings):
    shutil.rmtree(settings["tickets_processed"])


def rm(path):
    import shutil
    try:
        shutil.rmtree(path)
    except OSError, e:
        if e.errno == 20:
            os.remove(path)
        else:
            print("{} not such file or directory".format(path))


def get_models_path(checkpoints_path):
    classes = {}
    for parent, childs, files in os.walk(checkpoints_path):
        parent = parent.split("/").pop()
        if parent and len(childs) > 0:
            classes[parent] = childs
    return classes


def delete_file_model(clf, model_name, version, checkpoints_path):
    name_version = model_name + "." + version
    path = os.path.join(checkpoints_path, clf, name_version)
    rm(path)
    return path


def get_models_from_dataset(md5, checkpoints_path):
    from collections import defaultdict
    models_path = get_models_path(checkpoints_path)
    models_md5 = defaultdict(list)
    for clf, dataset in models_path.items():
        for name_version in dataset:
            model_path_meta = os.path.join(checkpoints_path, clf, name_version, name_version)
            model_path = os.path.join(checkpoints_path, clf, name_version)
            model_md5 = DataDrive.read_meta("md5", model_path_meta)
            models_md5[model_md5].append((model_path, model_path_meta))
    return models_md5.get(md5, [])


def get_date_from_file(file_path):
    return datetime.datetime.utcfromtimestamp(os.path.getmtime(file_path))
