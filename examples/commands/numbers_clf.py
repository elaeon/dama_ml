import argparse
from ml.ds import DataSetBuilderImage
from ml.utils.config import get_settings
from ml.processing import PreprocessingImage, FiTScaler
from ml.clf.extended import RandomForest
from skimage import io

settings = get_settings("ml")
settings.update(get_settings("numbers"))
settings.update(get_settings("transcriptor"))

if __name__ == '__main__':
    IMAGE_SIZE = int(settings["image_size"])
    transforms = [
            ("rgb2gray", None),
            ("resize", {"image_size": IMAGE_SIZE, "type_r": "asym"}), 
            ("threshold", 91), 
            ("merge_offset", {"image_size": IMAGE_SIZE, "bg_color": 1})]

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dataset", help="[cross] [adversarial]", type=str)
    parser.add_argument("--dataset-name", help="crea el dataset", type=str)
    parser.add_argument("--test", 
        help="evalua el predictor en base a los datos de prueba", 
        action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--predict", type=str)
    args = parser.parse_args()

    if args.build_dataset:
        ds_builder = DataSetBuilderImage(
            args.dataset_name, 
            image_size=int(settings["image_size"]), 
            train_folder_path=[
                settings["train_folder_path"]],
            transforms_row=transforms,
            transforms_global=[(FiTScaler.module_cls_name(), None)],
            processing_class=PreprocessingImage)
        ds_builder.build_dataset()
        ds_builder.info()
    elif args.train:
        dataset = DataSetBuilderImage.load_dataset(
            args.dataset_name)
        classif = RandomForest(
            dataset=dataset,
            model_name=args.model_name, 
            model_version=args.model_version,
            group_name="numbers")
        classif.train(num_steps=args.epoch, batch_size=128)
    elif args.test:
        classif = RandomForest(
            model_name=args.model_name, 
            model_version=args.model_version)
        classif.scores().print_scores(order_column="f1")
    elif args.predict:
        classif = RandomForest(
            model_name=args.model_name, 
            model_version=args.model_version)
        img = io.imread(args.predict)
        print(list(classif.predict([img])))
