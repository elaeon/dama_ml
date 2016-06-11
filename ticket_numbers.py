from skimage import io
from skimage import img_as_ubyte

import os
#import sys
import glob
import dlib
import argparse
import ml

from utils.config import get_settings
from utils.files import build_tickets_processed, delete_tickets_processed
settings = get_settings()

PICTURES = ["DSC_0055.jpg", "DSC_0056.jpg",
        "DSC_0058.jpg", "DSC_0059.jpg",
        "DSC_0060.jpg", "DSC_0061.jpg",
        "DSC_0062.jpg", "DSC_0053.jpg",
        "DSC_0054.jpg", "DSC_0057.jpg",
        "DSC_0063.jpg", "DSC_0064.jpg",
        "DSC_0065.jpg"]


IMAGE_SIZE = settings["image_size"]
DETECTOR_NAME = settings["detector_name"]

def numbers_images_set(url, g_filters):
    import xmltodict
    from tqdm import tqdm

    ds_builder = ml.ds.DataSetBuilder("", IMAGE_SIZE)
    labels_images = {}
    root = settings["examples"]
    for filename in ['tickets.xml', 'tickets_test.xml']:
        with open(os.path.join(root, "xml/"+filename)) as fd:
            doc = xmltodict.parse(fd.read())   
            for numbers in doc["dataset"]["images"]["image"]:
                image_file = numbers["@file"]
                filepath = image_file
                filepath = filepath[2:] if filepath.startswith("../") else filepath
                image = io.imread(root+filepath)
                image = ml.ds.ProcessImage(image, g_filters.get_filters()).image
                print(filepath)
                for box in numbers["box"]:
                    rectangle = (int(box["@top"]), 
                        int(box["@top"])+int(box["@height"]), 
                        int(box["@left"]), 
                        int(box["@left"])+int(box["@width"]))
                    l_filters.add_value("cut", rectangle)
                    thumb_bg = ml.ds.ProcessImage(image, l_filters.get_filters()).image
                    labels_images.setdefault(box["label"], [])
                    labels_images[box["label"]].append(thumb_bg)

    pbar = tqdm(labels_images.items())
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ds_builder.save_images(url, label, images)


def transcriptor(classif, g_filters, l_filters, d_filters, detector_path, url=None):
    from dlib import rectangle
    from utils.order import order_2d

    path = os.path.join(settings["root_data"], settings["checkpoints"])
    detector = dlib.simple_object_detector(detector_path)
    root = settings["examples"] + settings["pictures"] + "tickets/"
    if url is None:
        pictures = [os.path.join(root, f) for f in PICTURES[0:1]]
    else:
        pictures = [url]
    #win = dlib.image_window()
    for f in pictures:
        print("Processing file: {}".format(f))
        img = io.imread(f)
        dets = detector(ml.ds.ProcessImage(img, d_filters.get_filters()).image)
        #print("Numbers detected: {}".format(len(dets)))
        img = ml.ds.ProcessImage(img, g_filters.get_filters()).image
        data = [(e.left(), e.top(), e.right(), e.bottom()) for e in dets]
        for block, coords in order_2d(data, index=(1, 0), block_size=20).items():
            #win.clear_overlay()
            #print("####BLOCK:", block)
            #win.set_image(img_o)
            numbers = []
            for v in coords:
                r = rectangle(*v)
                m_rectangle = (r.top(), r.top() + r.height()-2, 
                    r.left() - 5, r.left() + r.width())
                l_filters.add_value("cut", m_rectangle)
                thumb_bg = ml.ds.ProcessImage(img, l_filters.get_filters()).image
                #win.set_image(img_as_ubyte(thumb_bg))
                #win.add_overlay(r)
                #print(list(classif.predict([thumb_bg])))
                numbers.append(thumb_bg)
                #dlib.hit_enter_to_continue()
            numbers_predicted = list(classif.predict(numbers))
            num_pred_coords = zip(numbers_predicted, coords)
            if len(num_pred_coords) >= 2:
                num_pred_coords = num_pred_coords + [num_pred_coords[-2]]
            elif len(num_pred_coords) == 1:
                #num_pred_coords = num_pred_coords + [num_pred_coords[-1]]
                continue
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
            #print(results, numbers_predicted, len(num_pred_coords))
            #print(results)
            #dlib.hit_enter_to_continue()
            yield results

def transcriptor_test(classif, g_filters, l_filters, d_filters, detector_path, url=None):
    predictions = transcriptor(classif, g_filters, l_filters, d_filters, detector_path, url=url)
    def flat_results():
        flat_p = [["".join(prediction) for prediction in row if len(prediction) > 0] 
            for row in predictions]
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

def build_dirty_image_set(url, classif, d_filters):
    from tqdm import tqdm
    root = settings["examples"] + settings["pictures"] + "tickets/"
    numbers = []
<<<<<<< HEAD
    #for f in PICTURES:
    for url_l in glob.glob(os.path.join(root, "Pictures/tickets/*.jpg")):
        #url_l = os.path.join(root, f)
        path = os.path.join(settings["root_data"], settings["checkpoints"])
        detector = dlib.simple_object_detector(path+DETECTOR_NAME+".svm")
        img_o = io.imread(url_l)
        dets = detector(img_o)
        img = ml.ds.ProcessImage(img_o, g_filters.get_filters()).image
=======
    for path in [os.path.join(root, f) for f in PICTURES]:
        checkpoint_path = os.path.join(settings["root_data"], settings["checkpoints"])
        detector = dlib.simple_object_detector(checkpoint_path+DETECTOR_NAME+".svm")
        img = io.imread(path)
        dets = detector(ml.ds.ProcessImage(img, d_filters.get_filters()).image)        
>>>>>>> 17d021ea711d64fac663a4adebe144bda9dfdd46
        print("Numbers detected: {}".format(len(dets)))        
        for r in dets:
            m_rectangle = (r.top(), r.top() + r.height()-2, 
                r.left() - 5, r.left() + r.width())
            l_filters.add_value("cut", m_rectangle)
            thumb_bg = ml.ds.ProcessImage(img, l_filters.get_filters()).image
            numbers.append(thumb_bg)
    numbers_predicted = list(classif.predict(numbers))
    labels_numbers = zip(numbers_predicted, numbers)
    numbers_g = {}
    for label, number in labels_numbers:
        numbers_g.setdefault(label, [])
        numbers_g[label].append(number)
    pbar = tqdm(numbers_g.items())
    for label, images in pbar:
        pbar.set_description("Processing {}".format(label))
        ml.ds.DataSetBuilder.save_images(url, label, images)

def transcriptor_product_price_writer(classif, g_filters, l_filters, d_filters, detector_path, url=None):
    import heapq
    predictions = transcriptor(classif, g_filters, l_filters, d_filters, detector_path, url=url)
    def flat_results():
        flat_p = [["".join(prediction) for prediction in row  if len(prediction) > 0] 
            for row in predictions]
        return flat_p

    product_price = []
    product = None
    prices = []
    for row in flat_results():
        #print(row)
        try:
            index = row.index("$")
            price = int(row[index+1]) / 100.
            prices.append(price)
            if index > 0 and len(row[index-1]) > 10:
                product_price.append((row[index-1], price))
                product = None
            elif product is not None:
                product_price.append((product, price))
                product = None
        except ValueError:
            for elem in row:
                if elem.rfind('000') > 0:
                    product = elem
        except IndexError:
            print("Error Index", row)
    return product_price, sum([price for _, price in product_price]), heapq.nlargest(2, prices)


def calc_avg_price_tickets(filename, g_filters, l_filters, d_filters, detector_path):
    base_path = settings["base_dir"] + settings["examples"] + settings["pictures"]
    prices = 0
    counter = 0
    for path in glob.glob(os.path.join(base_path, "tickets/*.jpg")):
        _, sum_, v = transcriptor_product_price_writer(
            filename, g_filters, l_filters, d_filters, detector_path, url=path)
        prices += v[1]
        print("COST:", v, sum_)
        counter += 1
    print("TOTAL:", prices / counter)


#test DSC_0055, DSC_0056
#training DSC_0053, DSC_0054, DSC_0057, DSC_0059, DSC_0062, DSC_0058
if __name__ == '__main__':
    g_filters = ml.ds.Filters("global", [("rgb2gray", None), ("threshold", 91)])
    l_filters = ml.ds.Filters("local", 
        [("cut", None), ("resize", (IMAGE_SIZE, 'asym')), ("merge_offset", (IMAGE_SIZE, 1))])

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="nombre del dataset a utilizar", type=str)
    parser.add_argument("--test", 
        help="predice los datos test con el dataset como base de conocimiento", 
        action="store_true")
    parser.add_argument("--build", help="crea el dataset", action="store_true")
    parser.add_argument("--train", help="inicia el entrenamiento", action="store_true")
    parser.add_argument("--clf", help="selecciona el clasificador", type=str)
    parser.add_argument("--build_numbers_set", help="crea el detector de numeros", action="store_true")
    parser.add_argument("--train-hog", help="--train-hog [xml_filename]", type=str)
    parser.add_argument("--test-hog", type=str)
    parser.add_argument("--transcriptor", type=str)
    parser.add_argument("--transcriptor-test", type=str)
    parser.add_argument("--build-dirty", help="", action="store_true")
    parser.add_argument("--build-tickets", action="store_true")
    parser.add_argument("--test-clf", type=str)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = None

    checkpoints_path = os.path.join(settings["root_data"], "checkpoints/")
    detector_path = checkpoints_path + "Hog/" + DETECTOR_NAME + "/"
    detector_path_meta = detector_path + DETECTOR_NAME + "_meta.pkl"
    detector_path_svm = detector_path + DETECTOR_NAME + ".svm"
    if args.build_tickets:
        d_filters = ml.ds.Filters("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])
        build_tickets_processed(d_filters, settings, PICTURES)
    elif args.build:
        ds_builder = ml.ds.DataSetBuilder(dataset_name, image_size=IMAGE_SIZE, 
            dataset_path=settings["root_data"]+settings["dataset"], 
            train_folder_path=settings["root_data"]+settings["pictures"]+"tickets/train/",
            filters={"local": l_filters, "global": g_filters})
        ds_builder.original_to_images_set(
            [settings["root_data"]+settings["pictures"]+"tickets/dirty_numbers/",
            settings["root_data"]+settings["pictures"]+"tickets/numbers/"],
            filter_data=False,
            test_data=False)
        ds_builder.build_dataset()
    elif args.build_numbers_set:
        build_tickets_processed(g_filters, settings, PICTURES)
        numbers_images_set(settings["root_data"]+settings["pictures"]+"tickets/numbers/", g_filters)
        delete_tickets_processed(settings)
    elif args.train_hog:
        from ml.detector import HOG
        hog = HOG()
        d_filters = ml.ds.Filters("detector", 
                [("rgb2gray", None), ("blur", .25), 
                    ("contrast", None), ("as_ubyte", None)])
                #[("rgb2gray", None), ("threshold", 91), ("as_ubyte", None)])
        #d_filters = ml.ds.Filters("global", [])
        build_tickets_processed(d_filters, settings, PICTURES)
        ml.ds.save_metadata(detector_path, detector_path_meta,
            {"d_filters": d_filters.get_filters(), 
            "filename_training": args.train_hog})
        hog.train(args.train_hog, detector_path_svm)
        delete_tickets_processed(settings)
        print("Cleaned")
    elif args.test_hog:
        from ml.detector import HOG
        hog = HOG()
        hog.test_set(args.test_hog, PICTURES)
    elif args.draw:
        from ml.detector import HOG
        d_filters = ml.ds.Filters("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])
        print("Detector Filters:", d_filters.get_filters())
        pictures = glob.glob(os.path.join(
            settings["base_dir"]+settings["examples"]+settings["pictures"]+"tickets", "*.jpg"))
        hog = HOG()
        hog.draw_detections(detector_path_svm, d_filters, sorted(pictures)[0:1])
    else:
        classifs = {
            "svc": {
                "name": ml.clf.SVCFace,
                "params": {"check_point_path": checkpoints_path, "pprint": False}},
            #"tensor": {
            #    "name": ml.clf.TensorFace,
            #    "params": {"check_point_path": checkpoints_path}},
            "tensor2": {
                "name": ml.clf.TfLTensor,
                "params": {"check_point_path": checkpoints_path, "pprint": False}},
            #"cnn": {
            #    "name": ml.clf.ConvTensor,
            #    "params": {"num_channels": 1, "check_point_path": checkpoints_path, "pprint": False}},
            #"residual": {
            #    "name": ml.clf.ResidualTensor,
            #    "params": {"num_channels": 1}}
        }
        if args.test_clf:
            dt = ml.clf.ClassifTest()
            dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, 
                dataset_path=settings["root_data"]+settings["dataset"], pprint=False)
            dt.dataset_test(classifs, dataset_name, dataset, args.test_clf)
        else:
            class_ = classifs[args.clf]["name"]
            dataset = ml.ds.DataSetBuilder.load_dataset(dataset_name, 
                dataset_path=settings["root_data"]+settings["dataset"], validation_dataset=False)
            params = classifs[args.clf]["params"]
            classif = class_(dataset_name, dataset, **params)
            classif.batch_size = 100
            print("#########", classif.__class__.__name__)
            if args.test:
                ds_builder = ml.ds.DataSetBuilder(dataset_name, 
                    dataset_path=settings["root_data"]+settings["dataset"], 
                    train_folder_path=settings["root_data"]+settings["pictures"]+"/tickets/train/")
                print("------ TEST FROM TEST-DATASET")
                classif.detector_test_dataset()
                dt = ml.clf.ClassifTest()
                dt.classif_test(classif, "f1")
            elif args.train:
                classif.train(num_steps=args.epoch)
            elif args.transcriptor:
                d_filters = ml.ds.Filters("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])
                print("Detector Filters:", d_filters.get_filters())
                g_filters = ml.ds.Filters("global", dataset["global_filters"])
                l_filters = ml.ds.Filters("local", dataset["local_filters"])
                if args.transcriptor == "avg":
                    calc_avg_price_tickets(classif, g_filters, l_filters, d_filters, detector_path_svm)
                else:
                    l, s, v = transcriptor_product_price_writer(
                        classif, g_filters, l_filters, d_filters, detector_path_svm, url=args.transcriptor)
                    print(l)
                    print(s)
                    print(v)
            elif args.transcriptor_test:
                d_filters = ml.ds.Filters("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])
                print("Detector Filters:", d_filters.get_filters())
                g_filters = ml.ds.Filters("global", dataset["global_filters"])
                l_filters = ml.ds.Filters("local", dataset["local_filters"])
                transcriptor_test(classif, g_filters, l_filters, d_filters, detector_path_svm)
            elif args.build_dirty:
                d_filters = ml.ds.Filters("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])
                build_dirty_image_set(
                    settings["root_data"]+settings["pictures"]+"tickets/dirty_numbers2/", 
                    classif,
                    d_filters)

#for directory in os.listdir("/home/sc/git/ML/examples/Pictures/tickets/"):
#     im = io.imread(os.path.join("/home/sc/git/ML/examples/Pictures/tickets/", directory))
#     print(exposure.is_low_contrast(im))
