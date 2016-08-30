import sys
sys.path.append("/home/alejandro/Programas/ML")

import os
import argparse
import ml

from skimage import io
from utils.config import get_settings
from utils.files import filename_from_path

settings = get_settings("ml")
settings.update(get_settings("tickets"))

PICTURES = ["DSC_0055.jpg", "DSC_0056.jpg",
        "DSC_0058.jpg", "DSC_0059.jpg",
        "DSC_0060.jpg", "DSC_0061.jpg",
        "DSC_0062.jpg", "DSC_0053.jpg",
        "DSC_0054.jpg", "DSC_0057.jpg",
        "DSC_0063.jpg", "DSC_0064.jpg",
        "DSC_0065.jpg"]

def numbers_images_set(url, transforms):
    import xmltodict
    from tqdm import tqdm

    ds_builder = ml.ds.DataSetBuilderImage("", image_size=settings["image_size"])
    labels_images = {}
    for filename in ['tickets.xml', 'tickets_test.xml']:
        with open(os.path.join(settings["xml"], filename)) as fd:
            doc = xmltodict.parse(fd.read())   
            for numbers in doc["dataset"]["images"]["image"]:
                image_file = numbers["@file"]
                filepath = image_file
                filepath = filepath[2:] if filepath.startswith("../") else filepath
                filename = filename_from_path(filepath)
                image = io.imread(settings["tickets"]+filename)
                image = ml.ds.PreprocessingImage(image, transforms.get_transforms("global")).pipeline()
                for box in numbers["box"]:
                    rectangle = (int(box["@top"]), 
                        int(box["@top"])+int(box["@height"]), 
                        int(box["@left"]), 
                        int(box["@left"])+int(box["@width"]))
                    transforms.add_transform("local", "cut", rectangle)
                    thumb_bg = ml.ds.PreprocessingImage(image, transforms.get_transforms("local")).pipeline()
                    labels_images.setdefault(box["label"], [])
                    labels_images[box["label"]].append(thumb_bg)

    pbar = tqdm(labels_images.items())
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ds_builder.save_images(url, label, images)

if __name__ == '__main__':
    IMAGE_SIZE = int(settings["image_size"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", help="crea el detector de numeros", action="store_true")
    parser.add_argument("--transforms", help="crea el detector de numeros", action="store_true")
    args = parser.parse_args()

    if args.build:
        if args.transforms:
            transforms = ml.processing.Transforms([
                ("global", 
                    [("rgb2gray", None)]),
                ("local", 
                    [("cut", None), 
                    ("resize", (IMAGE_SIZE, 'asym')), 
                    ("threshold", 91), 
                    ("merge_offset", (IMAGE_SIZE, 1))])])
        else:
            transforms = ml.processing.Transforms([
                ("global", []),
                ("local", [])])
        numbers_images_set(settings["numbers"], transforms)
