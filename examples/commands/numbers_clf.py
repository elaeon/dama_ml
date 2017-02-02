import argparse
from ml.ds import DataSetBuilderImage
from ml.utils.config import get_settings
from ml.processing import FiTScaler, Transforms
from ml.processing import rgb2gray, resize, threshold, merge_offset
from ml.clf.extended.w_sklearn import RandomForest
from skimage import io

settings = get_settings("ml")
settings.update(get_settings("tickets"))

if __name__ == '__main__':
    IMAGE_SIZE = int(settings["image_size"])
    transforms = Transforms()
    transforms.add(rgb2gray),
    transforms.add(resize, image_size_h=IMAGE_SIZE, image_size_w=IMAGE_SIZE), 
    transforms.add(threshold, block_size=91) 
    transforms.add(merge_offset, image_size=IMAGE_SIZE, bg_color=1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dataset", help="[cross] [adversarial]", type=str)
    parser.add_argument("--from-detector", action="store_true")
    parser.add_argument("--from-xml", action="store_true")
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
        if args.from_detector:
            train_folder_path=[settings["numbers_detector"]]
        elif args.from_xml:
            train_folder_path = [settings["numbers_xml"]]
        else:
            train_folder_path = [settings["numbers_xml"]]

        ds_builder = DataSetBuilderImage(
            args.dataset_name, 
            image_size=int(settings["image_size"]), 
            train_folder_path=train_folder_path,
            transforms=transforms,
            compression_level=9,
            rewrite=True)
        ds_builder.build_dataset()
        ds_builder.info()
    elif args.train:
        dataset = DataSetBuilderImage(name=args.dataset_name)
        classif = RandomForest(
            dataset=dataset,
            model_name=args.model_name, 
            model_version=args.model_version,
            group_name="numbers")
        print("Training")
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
