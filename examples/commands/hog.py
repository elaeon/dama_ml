import sys
sys.path.append("/home/alejandro/Programas/ML")

import argparse
import ml
import os
import glob

from skimage import io
from utils.files import build_tickets_processed, delete_tickets_processed
from utils.config import get_settings

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

    checkpoints_path = settings["checkpoints_path"]
    detector_path = checkpoints_path + "Hog/" + settings["detector_name"] + "/"
    detector_path_meta = detector_path + settings["detector_name"] + "_meta.pkl"
    detector_path_svm = detector_path + settings["detector_name"] + ".svm"

    if args.train:
        from ml.detector import HOG
        hog = HOG()
        transforms = ml.ds.Transforms([("detector", 
            [("rgb2gray", None), ("contrast", None)])])
        build_tickets_processed(transforms.get_transforms("detector"), settings, PICTURES)
        ml.ds.save_metadata(detector_path, detector_path_meta,
            {"d_filters": transforms.get_transforms("detector"), 
            "filename_training": args.train})
        hog.train(args.train, detector_path_svm)
        delete_tickets_processed(settings)
        print("Cleaned")
    elif args.test:
        from ml.detector import HOG
        hog = HOG()
        hog.test_set("f1", PICTURES)
    elif args.draw:
        from ml.detector import HOG
        transforms = ml.ds.Transforms([
            ("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])])
        print("HOG Filters:", transforms.get_transforms("detector"))
        pictures = glob.glob(os.path.join(settings["tickets"], "*.jpg"))
        hog = HOG()
        hog.draw_detections(detector_path_svm, 
            transforms.get_transforms("detector"), sorted(pictures)[0:1])
