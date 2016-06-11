import os
import shutil
import ml
from skimage import io

def build_tickets_processed(d_filters, settings, PICTURES):
    root = settings["examples"] + settings["pictures"]
    tickets_processed_url = os.path.join(root, "tickets_processed/")
    if not os.path.exists(tickets_processed_url):
        os.makedirs(tickets_processed_url)
    for path in [os.path.join(root + "tickets/", f) for f in PICTURES]:
        name = path.split("/").pop()
        image = io.imread(path)
        image = ml.ds.ProcessImage(image, d_filters.get_filters()).image
        d_path = os.path.join(tickets_processed_url, name)
        io.imsave(d_path, image)
        #print("Saved ", path, d_path)

def delete_tickets_processed(settings):
    folder = settings["examples"] + settings["pictures"] + "tickets_processed/"
    shutil.rmtree(folder)
