import argparse
import ml
from utils.camera_client import read_img_stream

FILTERS = [("rgb2gray", None), ("align_face", None), ("resize", 90)]
IP_ADDRESS = '192.168.52.102'
PORT = 8000

def build_images_face(url, number_id):
    ds_builder = ml.ds.DataSetBuilder(90)
    images = (ml.ds.ProcessImage(image, FILTERS).image 
            for img in read_img_stream(IP_ADDRESS, PORT, num_images=20))
    images, _ = ds_builder.build_train_test((number_id, images), sample=False)
    ds_builder.save_images(url, number_id, images.values())


def detect_face(face_classif):
    from collections import Counter
    images = (ml.ds.ProcessImage(image, FILTERS).image 
            for img in read_img_stream(IP_ADDRESS, PORT, num_images=20))
    counter = Counter(face_classif.predict_set(images))
    if len(counter) > 0:
        print(max(counter.items(), key=lambda x: x[1]))


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--empleado", help="numero de empleado", type=int)
    parser.add_argument("--foto", help="numero de empleado", action="store_true")
    parser.add_argument("--dataset", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--test", 
        help="predice los datos test con el dataset como base de conocimiento", 
        action="store_true")
    parser.add_argument("--build", help="crea el dataset", action="store_true")
    parser.add_argument("--rebuild", help="construye el dataset desde las images origen", action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--classif", help="selecciona el clasificador", type=str)
    args = parser.parse_args()
    image_size = 90
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "test_5"

    if args.empleado:
        build_images_face("/home/sc/Pictures/face/", args.empleado)
    elif args.build:
        ds_builder = ml.ds.DataSetBuilder(dataset_name, 90, filters=FILTERS)
        ds_builder.build_dataset("/home/sc/Pictures/face/")
    elif args.rebuild:
        ds_builder = ml.ds.DataSetBuilder(dataset_name, 90, filters=FILTERS)
        ds_builder.original_to_images_set("/home/sc/Pictures/face_o/")
        ds_builder.build_dataset("/home/sc/Pictures/face/")
    else:        
        classifs = {
            "svc": {
                "name": ml.clf.SVCFace,
                "params": {"image_size": image_size}},
            "tensor": {
                "name": ml.clf.TensorFace,
                "params": {"image_size": image_size}},
            "tensor2": {
                "name": ml.clf.TfLTensor,#face_training.Tensor2LFace,
                "params": {"image_size": image_size}},
            "cnn": {
                "name": ml.clf.ConvTensor,#ConvTensorFace
                "params": {"num_channels": 1, "image_size": image_size}},
            "residual": {
                "name": ml.clf.ResidualTensor,
                "params": {"num_channels": 1, "image_size": image_size}}
        }
        class_ = classifs[args.classif]["name"]
        params = classifs[args.classif]["params"]
        dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name)
        face_classif = class_(dataset_name, dataset, **params)
        face_classif.batch_size = 10
        print("#########", face_classif.__class__.__name__)
        if args.foto:                  
            detect_face(face_classif)
        elif args.test:
            d = ml.ds.DataSetBuilder(dataset_name, 90)
            d.detector_test(face_classif)
        elif args.train:
            face_classif.fit()
            face_classif.train(num_steps=50)
