import os
import argparse

from skimage import io
from ml.utils.config import get_settings
from ml.utils.files import filename_from_path
from ml.ds import DataSetBuilderImage
from ml.processing import Transforms, cut
from ml.clf.extended.w_sklearn import RandomForest

settings = get_settings("ml")
settings.update(get_settings("tickets"))
settings.update(get_settings("transcriptor"))


def tickets2numbers_from_xml(url, transforms=None):
    import xmltodict
    from tqdm import tqdm

    ds_builder = DataSetBuilderImage("", image_size=settings["image_size"])
    labels_images = {}
    for filename in ['tickets.xml', 'tickets_test.xml']:
        with open(os.path.join(settings["xml"], filename)) as fd:
            doc = xmltodict.parse(fd.read())   
            for numbers in doc["dataset"]["images"]["image"]:
                image_file = numbers["@file"]
                filepath = image_file
                filepath = filepath[2:] if filepath.startswith("../") else filepath
                filename = filename_from_path(filepath)
                image = io.imread(os.path.join(settings["tickets"], filename))
                image = transforms.apply(image)
                for box in numbers["box"]:
                    rectangle = (int(box["@top"]), 
                        int(box["@top"])+int(box["@height"]), 
                        int(box["@left"]), 
                        int(box["@left"])+int(box["@width"]))
                    thumb_bg = cut(image, rectangle=rectangle)
                    ds_builder.save_images(url, box["label"], [thumb_bg])
                    labels_images.setdefault(box["label"], [])
                    labels_images[box["label"]].append(thumb_bg)

    pbar = tqdm(labels_images.items())
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ds_builder.save_images(url, label, images, rewrite=True)
    print("Saved in: {}".format(url))


def tickets2numbers_from_detector(url, classif):
    from tqdm import tqdm
    from skimage import img_as_ubyte
    from ml.detector import HOG
    import glob
    import dlib

    tickets = glob.glob(os.path.join(settings["tickets"], "*.jpg"))
    numbers = []
    hog = HOG(model_name="detector", model_version="1")
    detector = hog.detector()
    for path in [os.path.join(settings["tickets"], f) for f in tickets]:
        if "DSC_0055.jpg" in path:
            continue
        img = io.imread(path)
        img_p = img_as_ubyte(hog.transforms.apply(img)) 
        dets = detector(img_p)
        print(path)
        print("Numbers detected: {}".format(len(dets)))        
        for r in dets:
            m_rectangle = (r.top(), r.top() + r.height()-2, 
                r.left() - 5, r.left() + r.width())
            thumb_bg = cut(img, rectangle=m_rectangle)
            numbers.append(thumb_bg)
    numbers_predicted = list(classif.predict(numbers, chunk_size=258))
    labels_numbers = zip(numbers_predicted, numbers)

    numbers_g = {}
    for label, number in labels_numbers:
        numbers_g.setdefault(label, [])
        numbers_g[label].append(number)

    pbar = tqdm(numbers_g.items())
    ds_builder = DataSetBuilderImage("", image_size=settings["image_size"])
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ds_builder.save_images(url, label, images)


if __name__ == '__main__':
    IMAGE_SIZE = int(settings["image_size"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-images", help="[xml] [detector]", type=str)
    parser.add_argument("--pixelate", help="pixelea los numeros", action="store_true")
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()

    transforms = Transforms()
    if args.pixelate:
        from ml.processing import pixelate        
        transforms.add(pixelate, pixel_width=16, pixel_height=16)

    if args.build_images == "xml":
        tickets2numbers_from_xml(settings["numbers"], transforms=transforms)
    elif args.build_images == "detector":
        classif = RandomForest(
            model_name=args.model_name,
            model_version=args.model_version)
        tickets2numbers_from_detector(settings["numbers_detector"], classif)
