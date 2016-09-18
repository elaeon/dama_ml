import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import ml

from utils.config import get_settings

settings = get_settings("ml")
settings.update(get_settings("numbers"))

if __name__ == '__main__':
    IMAGE_SIZE = int(settings["image_size"])
    transforms = [
            ("rgb2gray", None),
            ("resize", (settings["image_size"], 'asym')), 
            ("threshold", 91), 
            ("merge_offset", (IMAGE_SIZE, 1)),
            ("scale", None)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dataset", help="crea el dataset", type=str)
    parser.add_argument("--test", 
        help="evalua el predictor en base a los datos de prueba", 
        action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-version", type=str)
    args = parser.parse_args()

    if args.build_dataset:
        ds_builder = ml.ds.DataSetBuilderImage(
            args.build_dataset, 
            image_size=int(settings["image_size"]), 
            dataset_path=settings["dataset_path"], 
            train_folder_path=[
                settings["train_folder_path"]], 
                #settings["numbers_detector"]],
            transforms=transforms,
            transforms_apply=True,
            processing_class=ml.processing.PreprocessingImage)
        ds_builder.build_dataset()
    elif args.train:
        dataset = ml.ds.DataSetBuilder.load_dataset(
            args.model_name, 
            dataset_path=settings["dataset_path"])
        classif = ml.clf.extended.RandomForest(
            dataset=dataset, 
            check_point_path=settings["checkpoints_path"], 
            model_version=args.model_version)
        classif.batch_size = 100
        classif.train(num_steps=args.epoch)
    elif args.test:
        classif = ml.clf.extended.RandomForest(
            model_name=args.model_name, 
            check_point_path=settings["checkpoints_path"], 
            model_version=args.model_version)
        classif.batch_size = 100
        classif.scores()
