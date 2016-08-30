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
    args = parser.parse_args()

    if args.test_img:
        dataset = ml.ds.DataSetBuilder.load_dataset(
            settings["dataset_name"],
            processing_class=ml.processing.PreprocessingImage,
            dataset_path=settings["dataset_path"])
        classif = ml.clf_e.RandomForest(dataset, 
            check_point_path=settings["checkpoints_path"],
            pprint=False, model_version=args.model_version)
        data = io.imread(args.test_img)
        print(list(classif.predict([data])))
