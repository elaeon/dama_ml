import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
import argparse
import ml

from skimage import io
from utils.config import get_settings
from utils.files import filename_from_path

settings = get_settings("ml")
settings.update(get_settings("numbers"))
settings.update(get_settings("tickets"))


def tickets2numbers_from_xml(url, transforms):
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
                    transforms.add_first_transform("local", "cut", rectangle)
                    #print(transforms.get_transforms("local"), rectangle)
                    thumb_bg = ml.ds.PreprocessingImage(image, transforms.get_transforms("local")).pipeline()
                    #ds_builder.save_images(url, box["label"], [thumb_bg])
                    labels_images.setdefault(box["label"], [])
                    labels_images[box["label"]].append(thumb_bg)

    pbar = tqdm(labels_images.items())
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ds_builder.save_images(url, label, images, rewrite=True)


def tickets2numbers_from_detector(url, classif, transforms):
    from tqdm import tqdm
    from skimage import img_as_ubyte
    from ml.detector import HOG
    import glob
    import dlib

    tickets = glob.glob(os.path.join(settings["tickets"], "*.jpg"))
    settings.update(get_settings("transcriptor"))
    numbers = []
    hog = HOG(name="detector3", checkpoints_path=settings["checkpoints_path"])
    detector = hog.detector()
    for path in [os.path.join(settings["tickets"], f) for f in tickets]:
        img = io.imread(path)
        img = img_as_ubyte(
            ml.ds.PreprocessingImage(img, transforms.get_transforms("global")).pipeline()) 
        dets = detector(img)
        print(path)
        print("Numbers detected: {}".format(len(dets)))        
        for r in dets:
            m_rectangle = (r.top(), r.top() + r.height()-2, 
                r.left() - 5, r.left() + r.width())
            transforms.add_transform("local", "cut", m_rectangle)
            thumb_bg = ml.ds.PreprocessingImage(img, transforms.get_transforms("local")).pipeline()
            numbers.append(thumb_bg)
    numbers_predicted = list(classif.predict(numbers))
    labels_numbers = zip(numbers_predicted, numbers)

    numbers_g = {}
    for label, number in labels_numbers:
        numbers_g.setdefault(label, [])
        numbers_g[label].append(number)

    pbar = tqdm(numbers_g.items())
    ds_builder = ml.ds.DataSetBuilderImage("", image_size=settings["image_size"])
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ds_builder.save_images(url, label, images)


if __name__ == '__main__':
    IMAGE_SIZE = int(settings["image_size"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-images", help="[xml] [detector]", type=str)
    parser.add_argument("--transforms", help="crea el detector de numeros", action="store_true")
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()

    if args.transforms:
        transforms = ml.processing.Transforms([
            ("global", []), 
            ("local", [("pixelate", (16, 16))])])
    else:
        transforms = ml.processing.Transforms([
            ("global", []),
            ("local", [])])

    if args.build_images == "xml":
        tickets2numbers_from_xml(settings["numbers"], transforms)
    elif args.build_images == "detector":
        transforms = ml.processing.Transforms([
            ("global", 
                [("rgb2gray", None), ("contrast", None)]),
            ("local", 
                [])])
        classif = ml.clf_e.RandomForest(
            model_name=args.model_name,
            check_point_path=settings["checkpoints_path"],
            model_version=args.model_version)
        tickets2numbers_from_detector(settings["numbers_detector"], classif, transforms)
