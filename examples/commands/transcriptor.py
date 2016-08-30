import sys
sys.path.append("/home/alejandro/Programas/ML")

import argparse
import ml

from skimage import io
from utils.config import get_settings
settings = get_settings("ml")
settings.update(get_settings("numbers"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-img", help="", type=str)
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()

    if args.test_img:
        classif = ml.clf_e.RandomForest(
            model_name=args.model_name,
            check_point_path=settings["checkpoints_path"],
            model_version=args.model_version)
        data = io.imread(args.test_img)
        print(list(classif.predict([data])))
