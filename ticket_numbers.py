from skimage import io
from skimage import color
from skimage import filters as sk_filters
from skimage import img_as_ubyte

import os
import sys
import glob
import dlib
import argparse
import ml

from utils.config import get_settings
settings = get_settings()

PICTURES = ["Pictures/tickets/DSC_0055.jpg", "Pictures/tickets/DSC_0056.jpg",
        "Pictures/tickets/DSC_0058.jpg", "Pictures/tickets/DSC_0059.jpg",
        "Pictures/tickets/DSC_0060.jpg", "Pictures/tickets/DSC_0061.jpg",
        "Pictures/tickets/DSC_0062.jpg"]

def numbers_images_set(url):
    import xmltodict
    from tqdm import tqdm

    ds_builder = ml.ds.DataSetBuilder("", 90)
    labels_images = {}
    root = settings["examples"]
    for filename in ['tickets.xml', 'tickets_test.xml']:
        with open(os.path.join(root, "xml/"+filename)) as fd:
            doc = xmltodict.parse(fd.read())   
            for numbers in doc["dataset"]["images"]["image"]:
                image_file = numbers["@file"]
                filepath = image_file
                filepath = filepath[2:] if filepath.startswith("../") else filepath
                print(filepath)
                image = color.rgb2gray(io.imread(root+filepath))
                image = sk_filters.threshold_adaptive(image, 41, offset=0)
                for box in numbers["box"]:
                    rectangle = (int(box["@top"]), 
                        int(box["@top"])+int(box["@height"]), 
                        int(box["@left"]), 
                        int(box["@left"])+int(box["@width"]))
                    filters = [("cut", rectangle), 
                        ("resize", (90, 'asym')), ("merge_offset", (90, 1))]
                    thumb_bg = ml.ds.ProcessImage(image, filters).image
                    labels_images.setdefault(box["label"], [])
                    labels_images[box["label"]].append(thumb_bg)

    pbar = tqdm(labels_images.items())
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ds_builder.save_images(url, label, images)


def train():
    options = dlib.simple_object_detector_training_options()

    options.add_left_right_image_flips = False
    options.C = 1
    options.num_threads = 4
    options.be_verbose = True
    #options.epsilon = 0.0005
    #options.detection_window_size #60 pixels wide by 107 tall

    root = settings["examples"] + "xml/"
    path = os.path.join(settings["root_data"], "checkpoints/")
    training_xml_path = os.path.join(root, "tickets.xml")
    testing_xml_path = os.path.join(root, "tickets_test.xml")
    dlib.train_simple_object_detector(training_xml_path, path+"detector.svm", options)

    print("")  # Print blank line to create gap from previous output
    print("Test accuracy: {}".format(
        dlib.test_simple_object_detector(testing_xml_path, path+"detector.svm")))


def test():
    # Now let's use the detector as you would in a normal application.  First we
    # will load it from disk.
    #detector = dlib.fhog_object_detector("detector.svm")
    path = os.path.join(settings["root_data"], "checkpoints/")
    detector = dlib.simple_object_detector(path+"detector.svm")

    # We can look at the HOG filter we learned.  It should look like a face.  Neat!

    # Now let's run the detector over the images in the faces folder and display the
    # results.
    print("Showing detections on the images in the faces folder...")
    root = settings["examples"]
    win = dlib.image_window()
    #glob.glob(os.path.join(faces_folder, "*.jpg")):
    for f in PICTURES[0:1]:
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


def transcriptor(face_classif, url=None):
    from dlib import rectangle
    from utils.order import order_2d

    path = os.path.join(settings["root_data"], settings["checkpoints"])
    detector = dlib.simple_object_detector(path+"detector.svm")
    root = settings["examples"]
    if url is None:
        pictures = [os.path.join(root, f) for f in PICTURES]
    else:
        pictures = [url]
    win = dlib.image_window()
    for f in pictures[0:1]:
        print("Processing file: {}".format(f))
        img_o = io.imread(f)
        dets = detector(img_o)
        img = color.rgb2gray(img_o)
        img = sk_filters.threshold_adaptive(img, 41, offset=0)
        print("Numbers detected: {}".format(len(dets)))
        data = [(e.left(), e.top(), e.right(), e.bottom()) for e in dets]
        for block, coords in order_2d(data, index=(1, 0), block_size=40).items()[:12]:
            win.clear_overlay()
            #print("####BLOCK:", block)
            win.set_image(img_o)
            numbers = []
            for v in coords:
                r = rectangle(*v)
                m_rectangle = (r.top(), r.top() + r.height()-2, 
                    r.left() - 5, r.left() + r.width())
                filters = [("cut", m_rectangle), 
                    ("resize", (90, 'asym')), ("merge_offset", (90, 1))]
                thumb_bg = ml.ds.ProcessImage(img, filters).image
                #win.set_image(img_as_ubyte(thumb_bg))
                win.add_overlay(r)
                #print(list(face_classif.predict([thumb_bg])))
                numbers.append(thumb_bg)
                #dlib.hit_enter_to_continue()
            numbers_predicted = list(face_classif.predict(numbers))
            num_pred_coords = zip(numbers_predicted, coords)
            num_pred_coords = num_pred_coords + [num_pred_coords[-2]]
            results = []
            tmp_l = []
            for v1, v2 in zip(num_pred_coords, num_pred_coords[1:]):
                if v1[0] == "$":
                    results.append(tmp_l)
                    results.append([v1[0]])
                    tmp_l = []
                elif (abs(v1[1][0] - v2[1][0]) > 150):
                    tmp_l.append(v1[0])
                    results.append(tmp_l)
                    tmp_l = []
                else:
                    tmp_l.append(v1[0])
            results.append(tmp_l)
            #print(results)
            dlib.hit_enter_to_continue()
            yield results

def transcriptor_test(face_classif, url=None):
    predictions = transcriptor(face_classif, url=url)
    def flat_results():
        flat_p = [["".join(prediction) for prediction in row  if len(prediction) > 0] for row in predictions]
        return flat_p

    with open(settings["examples"]+"txt/transcriptor.txt") as f:
        file_ = f.read().split("\n")
        results = [[col.strip(" ") for col in line.split(",")] for line in file_]

    def count(result, prediction):
        total = 0
        total_positive = 0
        for v1, v2 in zip(result, prediction):
            mask = [x == y for x, y in zip(v1, v2)]
            total += len(mask)
            total_positive += mask.count(True)
        return total, total_positive

    def mark_error(result, prediction):
        results = []
        for r_chain, p_chain in zip(result, prediction):
            #print(r_chain, p_chain)
            str_l = []
            for chr1, chr2 in zip(r_chain, p_chain):
                if chr1 != chr2:
                    str_l.append("|")
                    str_l.append(chr2)
                    str_l.append("|")
                else:
                    str_l.append(chr2)
            results.append("".join(str_l))
        return results
                    
    flat_p = flat_results()
    total = 0
    total_positive = 0
    total_false = 0
    for result, prediction in zip(results, flat_p):
        d_false = 0
        if len(result) == len(prediction):
            v1, v2 = count(result, prediction)
            error_p = mark_error(result, prediction)
        else:
            clean_prediction = []
            for elem in prediction:
                if len(elem) > 1 or elem == "$":
                    clean_prediction.append(elem)
                elif len(elem) == 1 and elem != "$":
                    total_false += 1
                    d_false += 1
            v1, v2 = count(result, clean_prediction)
            error_p = mark_error(result, clean_prediction)
        if v1 != v2 or len(result) != len(prediction):
            print(result, error_p, d_false, prediction)
        total += v1
        total_positive += v2
            
    print(total, total_positive, total_false)
    print("Accuracy {}%".format(total_positive*100./total))
    print("Precision {}%".format((total-total_false)*100/total))

def build_dirty_image_set(url, face_classif):
    from tqdm import tqdm
    root = settings["examples"]
    numbers = []
    for f in PICTURES:#[PICTURES[4], PICTURES[6]]:
        url_l = os.path.join(root, f)
        path = os.path.join(settings["root_data"], settings["checkpoints"])
        detector = dlib.simple_object_detector(path+"detector.svm")
        img_o = io.imread(url_l)
        dets = detector(img_o)
        img = color.rgb2gray(img_o)
        img = sk_filters.threshold_adaptive(img, 41, offset=0)
        print("Numbers detected: {}".format(len(dets)))        
        for r in dets:
            m_rectangle = (r.top(), r.top() + r.height()-2, 
                r.left() - 5, r.left() + r.width())
            filters = [("cut", m_rectangle), 
                ("resize", (90, 'asym')), ("merge_offset", (90, 1))]
            thumb_bg = ml.ds.ProcessImage(img, filters).image
            numbers.append(thumb_bg)
    numbers_predicted = list(face_classif.predict(numbers))
    labels_numbers = zip(numbers_predicted, numbers)
    numbers_g = {}
    for label, number in labels_numbers:
        numbers_g.setdefault(label, [])
        numbers_g[label].append(number)
    pbar = tqdm(numbers_g.items())
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ml.ds.DataSetBuilder.save_images(url, label, images)

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
    parser.add_argument("--build_numbers_set", help="crea el detector de numeros", action="store_true")
    parser.add_argument("--train-hog", action="store_true")
    parser.add_argument("--test-hog", action="store_true")
    parser.add_argument("--transcriptor", type=str)
    parser.add_argument("--transcriptor-test", type=str)
    parser.add_argument("--build-dirty", help="", action="store_true")
    args = parser.parse_args()
    
    image_size = 90
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "test"

    checkpoints_path = os.path.join(settings["root_data"], "checkpoints/")
    if args.build:
        ds_builder = ml.ds.DataSetBuilder(dataset_name, 90, 
            dataset_path=settings["root_data"]+settings["dataset"], 
            test_folder_path=settings["root_data"]+settings["pictures"]+"tickets/test/", 
            train_folder_path=settings["root_data"]+settings["pictures"]+"tickets/train/")
        ds_builder.original_to_images_set(settings["root_data"]+settings["pictures"]+"tickets/numbers/")
        ds_builder.build_dataset(settings["root_data"]+settings["pictures"]+"tickets/train/")
    elif args.build_numbers_set:
        numbers_images_set(settings["root_data"]+settings["pictures"]+"tickets/numbers/")
    elif args.train_hog:
        train()
        #Test accuracy: precision: 0.973604, recall: 0.996881, average precision: 0.994134
        #Test accuracy: precision: 0.974619, recall: 0.997921, average precision: 0.995007
        #Test accuracy: precision: 0.975585, recall: 0.996881, average precision: 0.994052
        #Test accuracy: precision: 0.984424, recall: 0.985447, average precision: 0.982171
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
        dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, 
            dataset_path=settings["root_data"]+settings["dataset"])
        face_classif = class_(dataset_name, dataset, **params)
        face_classif.batch_size = 10
        print("#########", face_classif.__class__.__name__)
        if args.test:
            ds_builder = ml.ds.DataSetBuilder(dataset_name, 90, 
                dataset_path=settings["root_data"]+settings["dataset"], 
                test_folder_path=settings["root_data"]+settings["pictures"]+"/tickets/test/", 
                train_folder_path=settings["root_data"]+settings["pictures"]+"/tickets/train/")
            ds_builder.detector_test(face_classif)
            print("------ Dataset")
            face_classif.detector_test_dataset()
        elif args.train:
            face_classif.fit()
            face_classif.train(num_steps=20)
        elif args.transcriptor:
            if args.transcriptor == "d":
                transcriptor(face_classif)
            else:
                transcriptor(face_classif, url=args.transcriptor)
        elif args.transcriptor_test:
            transcriptor_test(face_classif)
        elif args.build_dirty:
            build_dirty_image_set(
                settings["root_data"]+settings["pictures"]+"tickets/dirty_numbers/", 
                face_classif)
