from skimage import io
from skimage import color
from skimage import filters
from skimage import transform
from sklearn import preprocessing
from skimage import img_as_ubyte
from skimage import exposure

import os
import numpy as np
import cPickle as pickle
import random

FACE_FOLDER_PATH = "/home/sc/Pictures/face/"
FACE_ORIGINAL_PATH = "/home/sc/Pictures/face_o/"
FACE_TEST_FOLDER_PATH = "/home/sc/Pictures/test/"
DATASET_PATH = "/home/sc/data/dataset/"


def save_metadata(path, file_path, data):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_metadata(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def proximity_label(label_ref, labels, dataset):
    from sklearn import svm
    dataset_ref, _ = dataset.only_labels([label_ref])
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(dataset_ref.reshape(-1, 90))
    for label in labels:
        if label == label_ref:
            yield label, 0
        else:
            dataset_other, _ = dataset.only_labels([label])
            y_pred_train = clf.predict(dataset_other.reshape(-1, 90))
            n_error_train = y_pred_train[y_pred_train == -1].size 
            yield label, (1 - (n_error_train / float(dataset_other.reshape(-1, 90).shape[0])))

class Filters(object):
    def __init__(self, name, filters):
        from collections import OrderedDict
        self.filters = OrderedDict(filters)
        self.placeholders = [k for k, v in self.filters.items() if v is None]
        self.name = name

    def get_placeholders(self):
        return self.placeholders

    def add_value(self, filter_name, value):
        self.filters[filter_name] = value

    def get_filters(self):
        return self.filters.items()

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
            try:
                self.image = transform.resize(self.image, dim)
            except ZeroDivisionError:
                pass
        
    def contrast(self):
        #contrast stretching
        p2, p98 = np.percentile(self.image, (2, 98))
        self.image = exposure.rescale_intensity(self.image, in_range=(p2, p98))

    def upsample(self):
        self.image = transform.pyramid_expand(
            self.image, upscale=2, sigma=None, order=1, mode='reflect', cval=0)

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

    def as_ubyte(self):
        self.image = img_as_ubyte(self.image)

    def merge_offset(self, image_size):
        if isinstance(image_size, int):
            bg_color = 1
        elif isinstance(image_size, tuple):
            image_size, bg_color = image_size

        bg = np.ones((image_size, image_size))
        offset = (int(round(abs(bg.shape[0] - self.image.shape[0]) / 2)), 
                int(round(abs(bg.shape[1] - self.image.shape[1]) / 2)))
        pos_v, pos_h = offset
        v_range1 = slice(max(0, pos_v), max(min(pos_v + self.image.shape[0], bg.shape[0]), 0))
        h_range1 = slice(max(0, pos_h), max(min(pos_h + self.image.shape[1], bg.shape[1]), 0))
        v_range2 = slice(max(0, -pos_v), min(-pos_v + bg.shape[0], self.image.shape[0]))
        h_range2 = slice(max(0, -pos_h), min(-pos_h + bg.shape[1], self.image.shape[1]))
        if bg_color is None:
            #print(np.std(image))
            #print(np.var(image))
            bg2 = bg - 1 + np.average(self.image) + random.uniform(-np.var(self.image), np.var(self.image))
        elif bg_color == 1:
            bg2 = bg
        else:
            bg2 = bg - 1
        
        bg2[v_range1, h_range1] = bg[v_range1, h_range1] - 1
        bg2[v_range1, h_range1] = bg2[v_range1, h_range1] + self.image[v_range2, h_range2]
        self.image = bg2

    def threshold(self, block_size=41):
        self.image = filters.threshold_adaptive(self.image, block_size, offset=0)

    def pipeline(self, filters):
        if filters is not None:
            for filter_, value in filters:
                if value is not None:
                    getattr(self, filter_)(value)
                else:
                    getattr(self, filter_)()

class DataSetBuilder(object):
    def __init__(self, name, image_size=None, channels=None, 
                dataset_path=DATASET_PATH, 
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
        self.train_dataset = None
        self.train_labels = None
        self.valid_dataset = None
        self.valid_labels = None
        self.test_dataset = None
        self.test_labels = None

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
        labels = []
        for image_index, (number_id, image_file) in enumerate(images):
            image_data = io.imread(image_file)
            if image_data.shape != dim:
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            image_data = image_data.astype(float)
            self.dataset[image_index] = preprocessing.scale(image_data)#image_data
            labels.append(number_id)
        self.labels = np.asarray(labels)

    def desfragment(self):
        if self.dataset is None:
            self.dataset = np.concatenate((
                self.train_dataset, self.valid_dataset, self.test_dataset), axis=0)
        if self.labels is None:
            self.labels = np.concatenate((
                self.train_labels, self.valid_labels, self.test_labels), axis=0)

    def labels_info(self):
        from collections import Counter
        self.desfragment()
        return Counter(self.labels)

    def only_labels(self, labels):
        s_labels = set(labels)
        dataset, n_labels = zip(*filter(lambda x: x[1] in s_labels, zip(self.dataset, self.labels)))
        return np.asarray(dataset), np.asarray(n_labels)

    def info(self):
        self.desfragment()
        print('Full dataset tensor:', self.dataset.shape)
        print('Mean:', np.mean(self.dataset))
        print('Standard deviation:', np.std(self.dataset))
        print('Labels:', self.labels.shape)
        print('Array length {}'.format(self.image_size))

        if self.filters is not None:
            print('Global filters: {}'.format(self.get_filters("global")))
            print('Local filters: {}'.format(self.get_filters("local")))

        print('Training set DS[{}], labels[{}]'.format(
            self.train_dataset.shape, self.train_labels.shape))

        if self.valid_dataset is not None:
            print('Validation set DS[{}], labels[{}]'.format(
                self.valid_dataset.shape, self.valid_labels.shape))

        print('Test set DS[{}], labels[{}]'.format(
            self.test_dataset.shape, self.test_labels.shape))

    def randomize(self, dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def cross_validators(self, train_size=0.7, valid_size=0.1):
        from sklearn import cross_validation
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self.dataset, self.labels, train_size=train_size, random_state=0)

        valid_size_index = int(round(X_train.shape[0] * valid_size))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def get_filters(self, name):
        if self.filters is not None and name in self.filters:
            v_filters = self.filters[name].get_filters()
        else:
            v_filters = None
        return v_filters

    def to_raw(self):
        l_filters = self.get_filters("local")
        g_filters = self.get_filters("global")

        return {
                'local_filters': l_filters,
                'global_filters': g_filters,
                'array_length': self.image_size,
                'train_dataset': self.train_dataset,
                'train_labels': self.train_labels,
                'valid_dataset': self.valid_dataset,
                'valid_labels': self.valid_labels,
                'test_dataset': self.test_dataset,
                'test_labels': self.test_labels}

    def from_raw(self, raw_data):
        self.filters = {}
        if raw_data["global_filters"] is not None:
            self.filters["global"] = Filters("global", raw_data["global_filters"])
        if raw_data["local_filters"] is not None:
            self.filters["local"] = Filters("local", raw_data["local_filters"])
        self.image_size = raw_data["array_length"]
        self.train_dataset = raw_data['train_dataset']
        self.train_labels = raw_data['train_labels']
        self.valid_dataset = raw_data['valid_dataset']
        self.valid_labels = raw_data['valid_labels']
        self.test_dataset = raw_data['test_dataset']
        self.test_labels = raw_data['test_labels']

    def save_dataset(self, valid_size=.1, train_size=.7):
        train_dataset, valid_dataset, test_dataset, train_labels, valid_labels, test_labels = self.cross_validators(train_size=train_size, valid_size=valid_size)
        try:
            with open(self.dataset_path+self.name, 'wb') as f:
                self.train_dataset = train_dataset
                self.train_labels = train_labels
                self.valid_dataset = valid_dataset
                self.valid_labels = valid_labels
                self.test_dataset = test_dataset
                self.test_labels = test_labels
                pickle.dump(self.to_raw(), f, pickle.HIGHEST_PROTOCOL)
            self.info()
        except Exception as e:
            print('Unable to save data to: ', self.dataset_path+self.name, e)
            raise

    @classmethod
    def load_dataset_raw(self, name, dataset_path=DATASET_PATH, validation_dataset=True):
        with open(dataset_path+name, 'rb') as f:
            save = pickle.load(f)
            if validation_dataset is False:
                save['train_dataset'] = np.concatenate((
                    save['train_dataset'], save['valid_dataset']), axis=0)
                save['train_labels'] = save['train_labels'] + save['valid_labels']
                save['valid_dataset'] = np.empty(0)
                save['valid_labels'] = []
            return save

    @classmethod
    def load_dataset(self, name, dataset_path=DATASET_PATH, validation_dataset=True, pprint=True):
        data = self.load_dataset_raw(name, dataset_path=dataset_path, 
                validation_dataset=validation_dataset)
        dataset = DataSetBuilder(name, dataset_path=dataset_path)
        dataset.from_raw(data)
        if pprint:
            dataset.info()
        return dataset        
        
    @classmethod
    def save_images(self, url, number_id, images):
        if not os.path.exists(url):
            os.makedirs(url)
        n_url = "{}{}/".format(url, number_id)
        if not os.path.exists(n_url):
             os.makedirs(n_url)
        for i, image in enumerate(images):
            io.imsave("{}face-{}-{}.png".format(n_url, number_id, i), image)

    def labels_images(self, urls):
        images_data = []
        labels = []
        if not isinstance(urls, list):
            urls = [urls]

        for url in urls:
            for number_id, path in self.images_from_directories(url):
                images_data.append(io.imread(path))
                labels.append(number_id)

        return images_data, labels

    def build_dataset(self):
        self.images_to_dataset(self.train_folder_path)
        self.save_dataset()
        self.clean_directory(self.train_folder_path)

    def clean_directory(self, path):
        import shutil
        shutil.rmtree(path)

    def original_to_images_set(self, url, test_folder_data=False):
        images_data, labels = self.labels_images(url)
        images = (ProcessImage(img, self.get_filters("global")).image for img in images_data)
        image_train, image_test = self.build_train_test(zip(labels, images), sample=test_folder_data)
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

