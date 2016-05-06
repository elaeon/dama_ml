from skimage import io
from skimage import color
from skimage import img_as_ubyte

import os
import sys
import glob
import dlib
import argparse
import ml

root_data = "/home/sc/ml_data/"

def numbers_images_set(url):
    import xmltodict

    ds_builder = ml.ds.DataSetBuilder("", 90)
    labels_images = {}
    root = "examples/"
    for filename in ['tickets.xml', 'tickets_test.xml']:
        with open(os.path.join(root, "xml/"+filename)) as fd:
            doc = xmltodict.parse(fd.read())   
            for numbers in doc["dataset"]["images"]["image"]:
                image_file = numbers["@file"]
                filepath = image_file
                print(filepath)
                image = color.rgb2gray(io.imread(root+filepath))
                for box in numbers["box"]:
                    rectangle = (int(box["@top"]), 
                        int(box["@top"])+int(box["@height"]), 
                        int(box["@left"]), 
                        int(box["@left"])+int(box["@width"]))
                    filters = [("cut", rectangle), ("resize", (90, 'asym')), ("merge_offset", 90)]
                    thumb_bg = ml.ds.ProcessImage(image, filters).image
                    labels_images.setdefault(box["label"], [])
                    labels_images[box["label"]].append(thumb_bg)

    for label, images in labels_images.items():
        ds_builder.save_images(url, label, images)


def train():
    options = dlib.simple_object_detector_training_options()

    options.add_left_right_image_flips = False
    options.C = 1
    options.num_threads = 4
    options.be_verbose = True
    #options.epsilon = 0.0005
    #options.detection_window_size #60 pixels wide by 107 tall

    root = "examples/xml/"
    path = os.path.join(root_data, "checkpoints/")
    training_xml_path = os.path.join(root, "tickets.xml")
    testing_xml_path = os.path.join(root, "tickets_test.xml")
    dlib.train_simple_object_detector(training_xml_path, path+"detector.svm", options)

    print("")  # Print blank line to create gap from previous output
    print("Test accuracy: {}".format(
        dlib.test_simple_object_detector(testing_xml_path, path+"detector.svm")))
    #print("Training accuracy: {}".format(
    #    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))


def test():
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    #detector = dlib.fhog_object_detector("detector.svm")
    path = os.path.join(root_data, "checkpoints/")
    detector = dlib.simple_object_detector(path+"detector.svm")

    # We can look at the HOG filter we learned.  It should look like a face.  Neat!

    # Now let's run the detector over the images in the faces folder and display the
    # results.
    print("Showing detections on the images in the faces folder...")
    root = "examples/"
    win = dlib.image_window()
    pictures = ["Pictures/tickets/DSC_0055.jpg", "Pictures/tickets/DSC_0056.jpg",
        "Pictures/tickets/DSC_0058.jpg", "Pictures/tickets/DSC_0059.jpg",
        "Pictures/tickets/DSC_0060.jpg", "Pictures/tickets/DSC_0061.jpg",
        "Pictures/tickets/DSC_0062.jpg"]
    #glob.glob(os.path.join(faces_folder, "*.jpg")):
    for f in pictures[0:1]:
        print(f)
        print("Processing file: {}".format(f))
        #img = img_as_ubyte(color.rgb2gray(io.imread(f)))
        img = io.imread(os.path.join(root, f))
        dets = detector(img)
        print("Numbers detected: {}".format(len(dets)))

        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()


def traductor(face_classif):
    path = os.path.join(root_data, "checkpoints/")
    detector = dlib.simple_object_detector(path+"detector.svm")
    pictures = ["Pictures/tickets/DSC_0055.jpg", "Pictures/tickets/DSC_0056.jpg",
        "Pictures/tickets/DSC_0058.jpg", "Pictures/tickets/DSC_0059.jpg",
        "Pictures/tickets/DSC_0060.jpg", "Pictures/tickets/DSC_0061.jpg",
        "Pictures/tickets/DSC_0062.jpg"]
    win = dlib.image_window()
    root = "examples/"
    for f in pictures[0:1]:
        print("Processing file: {}".format(f))
        img = io.imread(os.path.join(root, f))
        dets = detector(img)
        image = color.rgb2gray(img)
        print("Numbers detected: {}".format(len(dets)))
        for d in dets:
            rectangle = (d.top(),
                    d.top() + d.height(), 
                    d.left()-5, 
                    d.left() + d.width())
            filters = [("cut", rectangle), ("resize", (90, 'asym')), ("merge_offset", 90)]
            thumb_bg = ml.ds.ProcessImage(image, filters).image
            win.set_image(img_as_ubyte(thumb_bg))
            print(list(face_classif.predict([thumb_bg])))
            dlib.hit_enter_to_continue()
                

#test DSC_0055, DSC_0056
#training DSC_0053, DSC_0054, DSC_0057, DSC_0059, DSC_0062, DSC_0058
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--test", 
        help="predice los datos test con el dataset como base de conocimiento", 
        action="store_true")
    parser.add_argument("--build", help="crea el dataset", action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--clf", help="selecciona el clasificador", type=str)
    parser.add_argument("--build_number_set", help="crea el detector de numeros", action="store_true")
    parser.add_argument("--train-hog", action="store_true")
    parser.add_argument("--test-hog", action="store_true")
    parser.add_argument("--traductor", action="store_true")
    args = parser.parse_args()
    
    image_size = 90
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "test"

    checkpoints_path = os.path.join(root_data, "checkpoints/")
    if args.build:
        ds_builder = ml.ds.DataSetBuilder(dataset_name, 90, 
            dataset_path=root_data+"dataset/", test_folder_path=root_data+"Pictures/tickets/test/", 
            train_folder_path=root_data+"Pictures/tickets/train/")
        ds_builder.original_to_images_set(root_data+"Pictures/tickets/numbers/")
        ds_builder.build_dataset(root_data+"Pictures/tickets/train/")
    elif args.build_number_set:
        numbers_images_set(root_data+"Pictures/tickets/numbers/")
    elif args.train_hog:
        train()
        #Test accuracy: precision: 0.973604, recall: 0.996881, average precision: 0.994134
        #Test accuracy: precision: 0.974619, recall: 0.997921, average precision: 0.995007
        #Test accuracy: precision: 0.975585, recall: 0.996881, average precision: 0.994052
    elif args.test_hog:
        test()
    else:        
        classifs = {
            "svc": {
                "name": ml.clf.SVCFace,
                "params": {"image_size": image_size, "check_point_path": checkpoints_path}},
            "tensor": {
                "name": ml.clf.TensorFace,
                "params": {"image_size": image_size, "check_point_path": checkpoints_path}},
            "tensor2": {
                "name": ml.clf.TfLTensor,
                "params": {"image_size": image_size, "check_point_path": checkpoints_path}},
            "cnn": {
                "name": ml.clf.ConvTensor,
                "params": {"num_channels": 1, "image_size": image_size}},
            "residual": {
                "name": ml.clf.ResidualTensor,
                "params": {"num_channels": 1, "image_size": image_size}}
        }
        class_ = classifs[args.clf]["name"]
        params = classifs[args.clf]["params"]
        dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, dataset_path=root_data+"dataset/")
        face_classif = class_(dataset_name, dataset, **params)
        face_classif.batch_size = 10
        print("#########", face_classif.__class__.__name__)
        if args.test:
            ds_builder = ml.ds.DataSetBuilder(dataset_name, 90, 
                dataset_path=root_data+"dataset/", test_folder_path=root_data+"Pictures/tickets/test/", 
                train_folder_path=root_data+"Pictures/tickets/train/")
            ds_builder.detector_test(face_classif)
            print("------ Dataset")
            face_classif.detector_test_dataset()
        elif args.train:
            face_classif.fit()
            face_classif.train(num_steps=10)
        elif args.traductor:
            traductor(face_classif)
