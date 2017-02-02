import dlib
import os
import ml

from ml.utils.config import get_settings
from ml.utils.files import build_tickets_processed, delete_tickets_processed
from ml.clf.wrappers import DataDrive, ListMeasure
from ml.processing import Transforms

settings = get_settings("ml")
settings.update(get_settings("tickets"))


class HOG(DataDrive):
    """
    Create a histrogram oriented gradient.
    You need the dlib's library and his python bindings to use this class.

    :type model_name: string
    :param model_name: Name of the model

    :type check_point_path: string
    :param check_point_path: path where the model will be saved, this param is taken from settings

    :type model_version: string
    :param model_version: a string number for identify the differents models

    :type transforms: Transforms
    :param transforms: the transforms to apply to the data

    """
    def __init__(self, model_name=None, check_point_path=None, 
            model_version=None, transforms=None):
        if check_point_path is None:
            check_point_path = settings["checkpoints_path"]
        super(HOG, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name)
        #self.options.epsilon = 0.0005
        #self.options.detection_window_size #60 pixels wide by 107 tall
        self.options = dlib.simple_object_detector_training_options()
        self.options.add_left_right_image_flips = False
        self.options.C = .5
        self.options.num_threads = 4
        self.options.be_verbose = True
        self.transforms = transforms
        self.load()

    def load(self):
        """
        Loadd the metadata saved after the training.
        """
        try:
            meta = self.load_meta()
            self.transforms = Transforms.from_json(meta["transforms"])
            self.data_training_path = meta["data_training_path"]
        except IOError:
            pass
        except KeyError:
            pass

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def _metadata(self, score=None):
        list_measure = self.scores()
        return {"transforms": self.transforms.to_json(),
                "model_module": self.module_cls_name(),
                "data_training_path": self.data_training_path,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "score": list_measure.measures_to_dict()}

    def train(self, xml_filename):
        """
        :type xml_filename: string
        :param xml_filename: name of the filename where are defined the bounding boxes
        """
        examples = os.path.join(os.path.dirname(__file__), '../../examples/xml')
        self.data_training_path = os.path.join(examples, xml_filename)
        detector_path_svm = self.make_model_file()
        dlib.train_simple_object_detector(self.data_training_path, detector_path_svm, self.options)        
        self.save_meta()

    def scores(self, measures=None):
        """
        :type measures: list
        :param measures: list of measures names to show in the score's table.
        """
        if measures is None:
            measures = ["presicion", "recall", "f1"]
        elif isinstance(measures, str):
            measures = measures.split(",")

        list_measure = ListMeasure()
        score = self.test()

        list_measure.add_measure("CLF", self.__class__.__name__)
        list_measure.add_measure("precision", score.precision)
        list_measure.add_measure("recall", score.recall)
        list_measure.add_measure("f1", score.average_precision)

        return list_measure

    def test(self):
        """
        test the training model.
        """
        detector_path_svm = self.make_model_file()
        examples = os.path.join(os.path.dirname(__file__), '../../examples/xml')
        testing_xml_path = os.path.join(examples, "tickets_test.xml")
        return dlib.test_simple_object_detector(testing_xml_path, detector_path_svm)

    def draw_detections(self, pictures):
        """
        :type pictures: list
        :param pictures: list of paths of pictures to search the boinding boxes.

        draw the bounding boxes from the training model.
        """
        from skimage import io
        from skimage import img_as_ubyte

        detector = self.detector()
        win = dlib.image_window()
        for path in pictures:
            print("Processing file: {}".format(path))
            img = io.imread(path)
            img = img_as_ubyte(self.transforms.apply(img))
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

    def test_set(self, settings, PICTURES):
            build_tickets_processed(self.transforms, settings, PICTURES)
            score = self.test()
            delete_tickets_processed(settings)
            print(score)

    def detector(self):
        """
        return dlib.simple_object_detector
        """
        return dlib.simple_object_detector(self.make_model_file())
