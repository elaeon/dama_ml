from skimage import io
from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing
from skimage import img_as_ubyte

import os
import numpy as np
import cPickle as pickle

FACE_FOLDER_PATH = "/home/sc/Pictures/face/"
FACE_ORIGINAL_PATH = "/home/sc/Pictures/face_o/"
FACE_TEST_FOLDER_PATH = "/home/sc/Pictures/test/"
DATASET_PATH = "/home/sc/data/dataset/"

class ProcessImage(object):
    def __init__(self, image, filters):
        self.image = image
        self.pipeline(filters)

    def resize(self, image_size):
        if isinstance(image_size, int):
            type_ = "sym"
        elif isinstance(image_size, tuple):
            image_size, type_ = image_size

        if type_ == "asym":
            dim = []
            for v in self.image.shape:
                if v > image_size:
                    dim.append(image_size)
                else:
                    dim.append(v)
        else:
            dim = (image_size, image_size)
        if dim < self.image.shape or self.image.shape <= dim:
            self.image = transform.resize(self.image, dim)
        
    def rgb2gray(self):
        self.image = img_as_ubyte(color.rgb2gray(self.image))

    def blur(self, level):
        self.image = filters.gaussian(self.image, level)

    def align_face(self):
        from ml.face_detector import FaceAlign
        dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
        align = FaceAlign(dlibFacePredictor)
        self.image = align.process_img(self.image)

    def detector(self):
        from ml.face_detector import DetectorDlib
        dlibFacePredictor = "/home/sc/dlib-18.18/python_examples/shape_predictor_68_face_landmarks.dat"
        align = DetectorDlib(dlibFacePredictor)
        self.image = align.process_img(self.image)

    def cut(self, rectangle):
        top, bottom, left, right = rectangle
        self.image = self.image[top:bottom, left:right]

    def merge_offset(self, image_size):
        import random
        bg = np.ones((image_size, image_size))
        offset = (int(round(abs(bg.shape[0] - self.image.shape[0]) / 2)), 
                int(round(abs(bg.shape[1] - self.image.shape[1]) / 2)))
        pos_v, pos_h = offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + self.image.shape[0], bg.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + self.image.shape[1], bg.shape[1]), 0))
        v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], self.image.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], self.image.shape[1]))
        bg2 = bg - 1 + np.average(self.image) + random.uniform(-np.var(self.image), np.var(self.image))
        #print(np.std(image))
        #print(np.var(image))
        bg2[v_range1, h_range1] = bg[v_range1, h_range1] - 1
        bg2[v_range1, h_range1] = bg2[v_range1, h_range1] + self.image[v_range2, h_range2]
        self.image = bg2

    def pipeline(self, filters):
        if filters is not None:
            for filter_, value in filters:
                if value is not None:
                    getattr(self, filter_)(value)
                else:
                    getattr(self, filter_)()

class DataSetBuilder(object):
    def __init__(self, name, image_size, channels=None, dataset_path=DATASET_PATH, 
                test_folder_path=FACE_TEST_FOLDER_PATH, 
                train_folder_path=FACE_FOLDER_PATH,
                filters=None):
        self.image_size = image_size
        self.images = []
        self.dataset = None
        self.labels = None
        self.channels = channels
        self.test_folder_path = test_folder_path
        self.train_folder_path = train_folder_path
        self.dataset_path = dataset_path
        self.name = name
        self.filters = filters

    def add_img(self, img):
        self.images.append(img)
        #self.dataset.append(img)

    def images_from_directories(self, folder_base):
        images = []
        for directory in os.listdir(folder_base):
            files = os.path.join(folder_base, directory)
            if os.path.isdir(files):
                number_id = directory
                for image_file in os.listdir(files):
                    images.append((number_id, os.path.join(files, image_file)))
        return images

    def images_to_dataset(self, folder_base):
        """The loaded images must have been processed"""
        images = self.images_from_directories(folder_base)
        max_num_images = len(images)
        if self.channels is None:
            self.dataset = np.ndarray(
                shape=(max_num_images, self.image_size, self.image_size), dtype=np.float32)
            dim = (self.image_size, self.image_size)
        else:
            self.dataset = np.ndarray(
                shape=(max_num_images, self.image_size, self.image_size, self.channels), dtype=np.float32)
            dim = (self.image_size, self.image_size, self.channels)
        self.labels = []
        for image_index, (number_id, image_file) in enumerate(images):
            image_data = io.imread(image_file)
            if image_data.shape != dim:
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            image_data = image_data.astype(float)
            self.dataset[image_index] = preprocessing.scale(image_data)#image_data
            self.labels.append(number_id)
        print 'Full dataset tensor:', self.dataset.shape
        print 'Mean:', np.mean(self.dataset)
        print 'Standard deviation:', np.std(self.dataset)
        print 'Labels:', len(self.labels)

    def randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def cross_validators(self, dataset, labels, train_size=0.7, valid_size=0.1):
        from sklearn import cross_validation
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            dataset, labels, train_size=train_size, random_state=0)

        valid_size_index = int(round(X_train.shape[0] * valid_size))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def save_dataset(self, valid_size=.1, train_size=.7):
        train_dataset, valid_dataset, test_dataset, train_labels, valid_labels, test_labels = self.cross_validators(self.dataset, self.labels, train_size=train_size, valid_size=valid_size)
        try:
            f = open(self.dataset_path+self.name, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
                }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to: ', self.dataset_path+self.name, e)
            raise

        print("Test set: {}, Valid set: {}, Training set: {}".format(
            len(test_labels), len(valid_labels), len(train_labels)))

    @classmethod
    def load_dataset(self, name, dataset_path=DATASET_PATH):
        with open(dataset_path+name, 'rb') as f:
            save = pickle.load(f)
            print('Training set DS[{}], labels[{}]'.format(save['train_dataset'].shape, len(save['train_labels'])))
            print('Validation set DS[{}], labels[{}]'.format(save['valid_dataset'].shape, len(save['valid_labels'])))
            print('Test set DS[{}], labels[{}]'.format(save['test_dataset'].shape, len(save['test_labels'])))
            return save

    def merge_offset(self, image):
        import random
        bg = np.ones((self.image_size, self.image_size))
        offset = (int(round(abs(bg.shape[0] - image.shape[0]) / 2)), 
                int(round(abs(bg.shape[1] - image.shape[1]) / 2)))
        pos_v, pos_h = offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + image.shape[0], bg.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + image.shape[1], bg.shape[1]), 0))
        v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], image.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], image.shape[1]))
        bg2 = bg - 1 + np.average(image) + random.uniform(-np.var(image), np.var(image))
        #print(np.std(image))
        #print(np.var(image))
        bg2[v_range1, h_range1] = bg[v_range1, h_range1] - 1
        bg2[v_range1, h_range1] = bg2[v_range1, h_range1] + image[v_range2, h_range2]
        return bg2

    @classmethod
    def save_images(self, url, number_id, images):
        if not os.path.exists(url):
            os.makedirs(url)
        n_url = "{}{}/".format(url, number_id)
        if not os.path.exists(n_url):
             os.makedirs(n_url)
        for i, image in enumerate(images):
            io.imsave("{}face-{}-{}.png".format(n_url, number_id, i), image)

    def labels_images(self, url):
        images_data = []
        labels = []
        for number_id, path in self.images_from_directories(url):
            images_data.append(io.imread(path))
            labels.append(number_id)

        return images_data, labels

    def detector_test(self, face_classif):
        images_data = []
        labels = []
        for number_id, path in self.images_from_directories(self.test_folder_path):
            images_data.append(io.imread(path))
            labels.append(number_id)
        
        predictions = face_classif.predict_set(images_data)
        face_classif.accuracy(list(predictions), np.asarray(labels))

    def build_dataset(self, from_directory):
        self.images_to_dataset(from_directory)
        self.save_dataset()

    def original_to_images_set(self, url):
        images_data, labels = self.labels_images(url)
        images = (ProcessImage(img, self.filters).image for img in images_data)
        image_train, image_test = self.build_train_test(zip(labels, images))

        for number_id, images in image_train.items():
            self.save_images(self.train_folder_path, number_id, images)

        for number_id, images in image_test.items():
            self.save_images(self.test_folder_path, number_id, images)

    def build_train_test(self, process_images, sample=True):
        import random
        images = {}
        images_index = {}
        
        try:
            for number_id, image_array in process_images:
                images.setdefault(number_id, [])
                images[number_id].append(image_array)
        except TypeError:
            #if no faces are detected
            return {}, {}

        if sample is True:
            sample_data = {}
            images_good = {}
            for number_id in images:
                base_indexes = set(range(len(images[number_id])))
                if len(base_indexes) > 3:
                    sample_indexes = set(random.sample(base_indexes, 3))
                else:
                    sample_indexes = set([])
                sample_data[number_id] = [images[number_id][index] for index in sample_indexes]
                images_index[number_id] = base_indexes.difference(sample_indexes)

            for number_id, indexes in images_index.items():
                images_good.setdefault(number_id, [])
                for index in indexes:
                    images_good[number_id].append(images[number_id][index])
            
            return images_good, sample_data
        else:
            return images, {}
