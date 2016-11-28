import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import ml

from ml.utils.config import get_settings

settings = get_settings("ml")
settings.update(get_settings("numbers"))

if __name__ == '__main__':
    IMAGE_SIZE = int(settings["image_size"])
    transforms = [
            ("rgb2gray", None),
            ("resize", {"image_size": IMAGE_SIZE, "type_r": "asym"}), 
            ("threshold", 91), 
            ("merge_offset", {"image_size": IMAGE_SIZE, "bg_color": 1}),
            ("scale", None)]

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dataset", help="crea el dataset", action="store_true")
    parser.add_argument("--dataset-name", help="crea el dataset", type=str)
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
            args.dataset_name, 
            image_size=int(settings["image_size"]), 
            dataset_path=settings["dataset_path"], 
            train_folder_path=[
                settings["train_folder_path"], 
                settings["numbers_detector"]],
            transforms_row=transforms,
            #transforms_global=[(ml.processing.FiTScaler.module_cls_name(), None)],
            transforms_apply=True,
            processing_class=ml.processing.PreprocessingImage)
        ds_builder.build_dataset()
    elif args.train:
        dataset = ml.ds.DataSetBuilder.load_dataset(
            args.dataset_name, 
            dataset_path=settings["dataset_path"])
        classif = ml.clf.extended.RandomForest(
            dataset=dataset,
            model_name=args.model_name, 
            check_point_path=settings["checkpoints_path"], 
            model_version=args.model_version)
        classif.train(num_steps=args.epoch, batch_size=128)
    elif args.test:
        classif = ml.clf.extended.RandomForest(
            model_name=args.model_name, 
            check_point_path=settings["checkpoints_path"], 
            model_version=args.model_version)
        classif.scores().print_scores(order_column="f1")
