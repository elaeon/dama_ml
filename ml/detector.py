import dlib
import os
import ml

from utils.config import get_settings
from utils.files import build_tickets_processed, delete_tickets_processed
settings = get_settings("ml")
settings.update(get_settings("tickets"))


class HOG(object):
    def __init__(self, name=None, checkpoints_path=None, detector_version=None):
        self.options = dlib.simple_object_detector_training_options()

        self.options.add_left_right_image_flips = False
        self.options.C = .5
        self.options.num_threads = 4
        self.options.be_verbose = True
        #self.options.epsilon = 0.0005
        #self.options.detection_window_size #60 pixels wide by 107 tall
        self.name = name
        self.checkpoints_path = checkpoints_path
        self.detector_version = detector_version
        self.detector_path_meta = os.path.join(
            self.checkpoints_path, self.__class__.__name__, self.name, self.name+"_meta.pkl")
        self.detector_path_svm = os.path.join(
            self.checkpoints_path, self.__class__.__name__, self.name, self.name+".svm")

    def train(self, xml_filename):
        examples = os.path.join(os.path.dirname(__file__), '../examples/xml')
        training_xml_path = os.path.join(examples, xml_filename)
        testing_xml_path = os.path.join(examples, "tickets_test.xml")
        dlib.train_simple_object_detector(training_xml_path, self.detector_path_svm, self.options)

        print("")
        print("Test accuracy: {}".format(
            dlib.test_simple_object_detector(testing_xml_path, self.detector_path_svm)))

    def test(self, detector_path):
        examples = os.path.join(os.path.dirname(__file__), '../examples/xml')
        testing_xml_path = os.path.join(examples, "tickets_test.xml")
        return dlib.test_simple_object_detector(testing_xml_path, self.detector_path_svm)

    def draw_detections(self, transforms, pictures):
        from skimage import io
        from skimage import img_as_ubyte

        detector = self.detector()
        win = dlib.image_window()
        for path in pictures:
            print("Processing file: {}".format(path))
            img = io.imread(path)
            img = img_as_ubyte(ml.ds.PreprocessingImage(img, transforms).pipeline())
            dets = detector(img)
            print("Numbers detected: {}".format(len(dets)))

            win.clear_overlay()
            win.set_image(img)
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()

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
        base_dir = os.path.join(settings["checkpoints_path"], self.__class__.__name__)
        for k, v in self.images_from_directories(base_dir):
            files.setdefault(k, {})
            if v.endswith(".svm"):
                files[k]["svm"] = v
            else:
                files[k]["meta"] = v

        table = []
        for name, type_ in files.items():
            transforms = ml.ds.Transforms([
                ("detector", ml.ds.load_metadata(type_["meta"])["d_filters"])])
            build_tickets_processed(transforms.get_transforms("detector"), settings, PICTURES)
            measure = self.test(type_["svm"])
            table.append((name, measure.precision, measure.recall, measure.average_precision))
            delete_tickets_processed(settings)

        order_table_print(headers, table, order_column)

    def detector(self):
        return dlib.simple_object_detector(self.detector_path_svm)
