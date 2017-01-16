import argparse
import os
import glob

from skimage import io
from ml.utils.files import build_tickets_processed, delete_tickets_processed
from ml.utils.config import get_settings
from ml.ds import save_metadata, load_metadata
from ml.detector import HOG
from ml.processing import Transforms

settings = get_settings("ml")
settings.update(get_settings("transcriptor"))
settings.update(get_settings("tickets"))


PICTURES = ["DSC_0055.jpg", "DSC_0056.jpg",
        "DSC_0058.jpg", "DSC_0059.jpg",
        "DSC_0060.jpg", "DSC_0061.jpg",
        "DSC_0062.jpg", "DSC_0053.jpg",
        "DSC_0054.jpg", "DSC_0057.jpg",
        "DSC_0063.jpg", "DSC_0064.jpg",
        "DSC_0065.jpg"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="--train-hog [xml_filename]", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    if args.train:
        from ml.processing import rgbgray, contrast
        transforms = Transforms()
        transforms.add(rgb2gray), 
        transforms.add(contrast)
        hog = HOG(model_name="detector", transforms=transforms)
        build_tickets_processed(transforms, settings, PICTURES)
        hog.train(args.train)
        delete_tickets_processed(settings)
        print("Cleaned")
    elif args.test:
        hog = HOG(model_name="detector")
        hog.test_set(settings, PICTURES)
    elif args.draw:
        hog = HOG(model_name="detector")
        pictures = glob.glob(os.path.join(settings["tickets"], "*.jpg"))        
        hog.draw_detections(sorted(pictures)[0:1])
