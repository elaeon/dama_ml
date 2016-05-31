import dlib
import os
import ml

from utils.config import get_settings
from utils.files import build_tickets_processed, delete_tickets_processed
settings = get_settings()

class HOG(object):
    def __init__(self):
        self.options = dlib.simple_object_detector_training_options()

        self.options.add_left_right_image_flips = False
        self.options.C = .5
        self.options.num_threads = 4
        self.options.be_verbose = True
        #self.options.epsilon = 0.0005
        #self.options.detection_window_size #60 pixels wide by 107 tall

    def train(self, xml_filename, detector_path_svm):
        root = settings["examples"] + "xml/"
        path = os.path.join(settings["root_data"], "checkpoints/")
        training_xml_path = os.path.join(root, xml_filename)
        testing_xml_path = os.path.join(root, "tickets_test.xml")
        dlib.train_simple_object_detector(training_xml_path, detector_path_svm, self.options)

        print("")
        print("Test accuracy: {}".format(
            dlib.test_simple_object_detector(testing_xml_path, path+DETECTOR_NAME+".svm")))


    def test(self, detector_path):
        # Now let's use the detector as you would in a normal application.  First we
        # will load it from disk.
        #detector = dlib.fhog_object_detector("detector.svm")
        #path = os.path.join(settings["root_data"], "checkpoints/")
        #detector = dlib.simple_object_detector(path+DETECTOR_NAME+".svm")

        # We can look at the HOG filter we learned.  It should look like a face.  Neat!

        # Now let's run the detector over the images in the faces folder and display the
        # results.
        #print("Showing detections on the images in the faces folder...")
        #root = settings["examples"] + settings["pictures"] + "tickets/"
        #win = dlib.image_window()
        #glob.glob(os.path.join(faces_folder, "*.jpg")):
        #for path in [os.path.join(root, f) for f in PICTURES[0:1]]:
        #    print(path)
        #    print("Processing file: {}".format(path))
        #    img = io.imread(path)
        #    img = ml.ds.ProcessImage(img, d_filters.get_filters()).image
        #    dets = detector(img)
        #    print("Numbers detected: {}".format(len(dets)))

        #    win.clear_overlay()
        #    win.set_image(img)
        #    win.add_overlay(dets)
        #    dlib.hit_enter_to_continue()
        
        root = settings["examples"] + "xml/"
        path = os.path.join(settings["root_data"], "checkpoints/")
        testing_xml_path = os.path.join(root, "tickets_test.xml")
        return dlib.test_simple_object_detector(testing_xml_path, detector_path)

    def images_from_directories(self, folder_base):
        images = []
        for directory in os.listdir(folder_base):
            files = os.path.join(folder_base, directory)
            if os.path.isdir(files):
                number_id = directory
                for image_file in os.listdir(files):
                    images.append((number_id, os.path.join(files, image_file)))
        return images

    def test_set(self, order_column, PICTURES):
        from utils.order import order_table_print
        headers = ["Detector", "Precision", "Recall", "F1"]
        files = {}
        for k, v in self.images_from_directories(os.path.join(settings["root_data"], "checkpoints/Hog/")):
            files.setdefault(k, {})
            if v.endswith(".svm"):
                files[k]["svm"] = v
            else:
                files[k]["meta"] = v

        table = []
        for name, type_ in files.items():
            try:
                meta = ml.ds.load_metadata(type_["meta"])
            except KeyError:
                print("The file '{}' has not metadata".format(name))
                continue
            build_tickets_processed(ml.ds.Filters("detector", meta["d_filters"]), settings, PICTURES)
            measure = self.test(type_["svm"])
            table.append((name, measure.precision, measure.recall, measure.average_precision))
            delete_tickets_processed(settings)

        order_table_print(headers, table, order_column)
