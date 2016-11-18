import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse
import ml
import os

from skimage import io
from ml.utils.config import get_settings

settings = get_settings("ml")
settings.update(get_settings("transcriptor"))
settings.update(get_settings("tickets"))
settings.update(get_settings("numbers"))


PICTURES = ["DSC_0055.jpg", "DSC_0056.jpg",
        "DSC_0058.jpg", "DSC_0059.jpg",
        "DSC_0060.jpg", "DSC_0061.jpg",
        "DSC_0062.jpg", "DSC_0053.jpg",
        "DSC_0054.jpg", "DSC_0057.jpg",
        "DSC_0063.jpg", "DSC_0064.jpg",
        "DSC_0065.jpg"]


def transcriptor(classif, transforms, detector_path, url=None):
    import dlib
    from dlib import rectangle
    from ml.utils.order import order_2d
    from skimage import img_as_ubyte

    detector = dlib.simple_object_detector(detector_path)
    if url is None:
        pictures = [os.path.join(settings["tickets"], f) for f in PICTURES[0:1]]
    else:
        pictures = [url]
    win = dlib.image_window()
    for f in pictures:
        print("Processing file: {}".format(f))
        img = io.imread(f)
        img = img_as_ubyte(
            ml.ds.PreprocessingImage(img, 
            transforms.get_transforms("detector")).pipeline())
        dets = detector(img)
        data = [(e.left(), e.top(), e.right(), e.bottom()) for e in dets]
        for block, coords in order_2d(data, index=(1, 0), block_size=20).items():
            win.clear_overlay()
            print("####ROW:", block)
            win.set_image(img)
            numbers = []
            for v in coords:
                r = rectangle(*v)
                m_rectangle = (r.top(), r.top() + r.height()-2, 
                    r.left() - 5, r.left() + r.width())
                thumb_bg = ml.ds.PreprocessingImage(img, [("cut", m_rectangle)]).pipeline()
                win.add_overlay(r)
                numbers.append(thumb_bg)
            numbers_predicted = list(classif.predict(numbers))
            print("PREDICTED", numbers_predicted)
            num_pred_coords = zip(numbers_predicted, coords)
            if len(num_pred_coords) >= 2:
                num_pred_coords = num_pred_coords + [num_pred_coords[-2]]
            elif len(num_pred_coords) == 1:
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
            dlib.hit_enter_to_continue()
            #break
            yield results

def transcriptor_test(classif, transforms, detector_path, url=None):
    predictions = transcriptor(classif, transforms, detector_path, url=url)
    def flat_results():
        flat_p = [["".join(prediction) for prediction in row if len(prediction) > 0] 
            for row in predictions]
        return flat_p

    transcriptor_example = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'txt/transcriptor.txt'))
    with open(transcriptor_example) as f:
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
                    
    flat_p = flat_results()
    total = 0
    total_positive = 0
    total_false = 0
    for result, prediction in zip(results, flat_p):
        if len(result) == len(prediction):
            v1, v2 = count(result, prediction)
        else:
            clean_prediction = []
            for elem in prediction:
                if len(elem) > 1 or elem == "$":
                    clean_prediction.append(elem)
                elif len(elem) == 1 and elem != "$":
                    total_false += 1
            v1, v2 = count(result, clean_prediction)
        if v1 != v2 or len(result) != len(prediction):
            print("ORIGINAL: {}, PREDICTION {}". format(result, prediction))
        total += v1
        total_positive += v2
          
    print("TOTAL NUMBERS: [{}] POSITIVE: [{}] NEGATIVE: [{}]".format(
        total, total_positive, total_false))
    print("Accuracy {}%".format(total_positive*100./total))
    print("Precision {}%".format((total-total_false)*100/total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcriptor-number-test", help="", type=str)
    parser.add_argument("--transcriptor-ticket-test", help="", action="store_true")
    parser.add_argument("--model-version", type=str)
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()

    checkpoints_path = settings["checkpoints_path"]
    detector_path = checkpoints_path + "HOG/" + settings["detector_name"] + "/"
    detector_path_meta = detector_path + settings["detector_name"] + "_meta.pkl"
    detector_path_svm = detector_path + settings["detector_name"] + ".svm"

    classif = ml.clf.extended.RandomForest(
        model_name=args.model_name,
        check_point_path=settings["checkpoints_path"],
        model_version=args.model_version)

    if args.transcriptor_number_test:
        data = io.imread(args.transcriptor_number_test)
        print(list(classif.predict([data])))
    elif args.transcriptor_ticket_test:
        transforms = ml.ds.Transforms([
            ("detector", ml.ds.load_metadata(detector_path_meta)["d_filters"])])
        print("HOG Filters:", transforms.get_transforms("detector"))
        transcriptor_test(classif, transforms, detector_path_svm)
