import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import ml
import os
import glob

from skimage import io
from ml.utils.files import build_tickets_processed, delete_tickets_processed
from ml.utils.config import get_settings

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
        from ml.detector import HOG
        hog = HOG(name=settings["detector_name"], checkpoints_path=settings["checkpoints_path"])
        transforms = ml.ds.Transforms([("detector", 
            [("rgb2gray", None), ("contrast", None)])])
        build_tickets_processed(transforms.get_transforms("detector"), settings, PICTURES)
        ml.ds.save_metadata(detector_path, detector_path_meta,
            {"d_filters": transforms.get_transforms("detector"), 
            "filename_training": args.train})
        hog.train(args.train)
        delete_tickets_processed(settings)
        print("Cleaned")
    elif args.test:
        from ml.detector import HOG
        hog = HOG(name=settings["detector_name"], checkpoints_path=settings["checkpoints_path"])
        hog.test_set("f1", PICTURES)
    elif args.draw:
        from ml.detector import HOG
        transforms = ml.ds.Transforms([
            ("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])])
        print("HOG Filters:", transforms.get_transforms("detector"))
        pictures = glob.glob(os.path.join(settings["tickets"], "*.jpg"))
        hog = HOG()
        hog.draw_detections(transforms.get_transforms("detector"), sorted(pictures)[0:1])
