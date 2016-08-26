import os
import shutil
import ml
import ntpath

from skimage import io

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
        image = ml.processing.PreprocessingImage(image, transforms.get_transforms("global")).pipeline()
        d_path = os.path.join(tickets_processed_url, name)
        io.imsave(d_path, image)

def delete_tickets_processed(settings):
    folder = settings["examples"] + settings["pictures"] + "tickets_processed/"
    shutil.rmtree(folder)
