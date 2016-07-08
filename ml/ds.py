from skimage import io

import os
import numpy as np
import pandas as pd
import cPickle as pickle
import random

from processing import PreprocessingImage, Preprocessing, Transforms

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
    clf = svm.OneClassSVM(nu=.2, kernel="rbf", gamma=0.5)
    clf.fit(dataset_ref.reshape(dataset_ref.shape[0], -1))
    for label in labels:
        dataset_other, _ = dataset.only_labels([label])
        y_pred_train = clf.predict(dataset_other.reshape(dataset_other.shape[0], -1))
        n_error_train = y_pred_train[y_pred_train == -1].size
        yield label, (1 - (n_error_train / float(y_pred_train.size)))

def proximity_dataset(label_ref, labels, dataset):
    from sklearn import svm
    dataset_ref, _ = dataset.only_labels([label_ref])
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(dataset_ref.reshape(dataset_ref.shape[0], -1))
    for label in labels:
        dataset_other_, _ = dataset.only_labels([label])
        y_pred_train = clf.predict(dataset_other_.reshape(dataset_other_.shape[0], -1))
        return filter(lambda x: x[1] == -1, zip(dataset_other_, y_pred_train))


class DataSetBuilder(object):
    def __init__(self, name, 
                dataset_path=DATASET_PATH, 
                test_folder_path=FACE_TEST_FOLDER_PATH, 
                train_folder_path=FACE_FOLDER_PATH):
        self.dataset = None
        self.labels = None
        self.test_folder_path = test_folder_path
        self.train_folder_path = train_folder_path
        self.dataset_path = dataset_path
        self.name = name
        self.train_dataset = None
        self.train_labels = None
        self.valid_dataset = None
        self.valid_labels = None
        self.test_dataset = None
        self.test_labels = None
        self.transforms = Transforms("global", [("scale", None)])

    def dim(self):
        return self.dataset.shape

    def desfragment(self):
        if self.dataset is None:
            self.dataset = np.concatenate((
                self.train_dataset, self.valid_dataset, self.test_dataset), axis=0)
        if self.labels is None:
            self.labels = np.concatenate((
                self.train_labels, self.valid_labels, self.test_labels), axis=0)

    def labels_info(self):
        from collections import Counter
        return Counter(self.labels)

    def only_labels(self, labels):
        s_labels = set(labels)
        dataset, n_labels = zip(*filter(lambda x: x[1] in s_labels, zip(self.dataset, self.labels)))
        return np.asarray(dataset), np.asarray(n_labels)

    def info(self):
        print('Full dataset tensor:', self.dataset.shape)
        print('Mean:', np.mean(self.dataset))
        print('Standard deviation:', np.std(self.dataset))
        print('Labels:', self.labels.shape)
        print('Global filters: {}'.format(self.transforms.get_transforms("global")))
        #print('Num features {}'.format(self.image_size))

        #print('Training set DS[{}], labels[{}]'.format(
        #    self.train_dataset.shape, self.train_labels.shape))

        #if self.valid_dataset is not None:
        #    print('Validation set DS[{}], labels[{}]'.format(
        #        self.valid_dataset.shape, self.valid_labels.shape))

        #print('Test set DS[{}], labels[{}]'.format(
        #    self.test_dataset.shape, self.test_labels.shape))

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

    def to_raw(self):
        return {
            'train_dataset': self.train_dataset,
            'train_labels': self.train_labels,
            'valid_dataset': self.valid_dataset,
            'valid_labels': self.valid_labels,
            'test_dataset': self.test_dataset,
            'test_labels': self.test_labels,
            'global_filters': self.transforms.get_transforms("global")}

    def from_raw(self, raw_data):
        self.train_dataset = raw_data['train_dataset']
        self.train_labels = raw_data['train_labels']
        self.valid_dataset = raw_data['valid_dataset']
        self.valid_labels = raw_data['valid_labels']
        self.test_dataset = raw_data['test_dataset']
        self.test_labels = raw_data['test_labels']        
        self.desfragment()
        self.transforms = Transforms("global", raw_data["global_filters"])

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

    def is_binary(self):
        return len(self.labels_info()) == 2

    @classmethod
    def to_DF(self, dataset, labels):
        columns_name = map(lambda x: "c"+str(x), range(dataset.shape[-1])) + ["target"]
        return pd.DataFrame(data=np.column_stack((dataset, labels)), columns=columns_name)

    def to_df(self):
        self.desfragment()
        return self.to_DF(self.dataset, self.labels)

    def transform(self, fn):
        data = np.ndarray(
            shape=self.dataset.shape, dtype=np.float32)
        for i, row in enumerate(fn(self.dataset)):
            data[i] = row

        dataset = DataSetBuilder(self.name+"T", dataset_path=self.dataset_path)
        dataset.build_from_data_labels(self.dataset, self.labels)
        return dataset

    def build_from_data_labels(self, data, labels):
        self.dataset = data
        self.labels = labels
        self.save_dataset()

    def get_transforms(self, name):
        if self.transforms is not None and name in self.transforms:
            v_transforms = self.transforms[name].get_transforms()
        else:
            v_transforms = None
        return v_transforms

    def copy(self):
        dataset = DataSetBuilder(self.name)
        dataset.dataset = self.dataset
        dataset.labels = self.labels
        dataset.test_folder_path = self.test_folder_path
        dataset.train_folder_path = self.train_folder_path
        dataset.dataset_path = self.dataset_path
        dataset.train_dataset = self.train_dataset
        dataset.train_labels = self.train_labels
        dataset.valid_dataset = self.valid_dataset
        dataset.valid_labels = self.valid_labels
        dataset.test_dataset = self.test_dataset
        dataset.test_labels = self.test_labels
        return dataset

    def processing(self, data, processing_class):
        if not self.transforms.empty():
            preprocessing = processing_class(data, self.transforms.get_transforms("global"))
            return preprocessing.pipeline()
        else:
            return data


class DataSetBuilderImage(DataSetBuilder):
    def __init__(self, name, image_size=None, channels=None, 
                dataset_path=DATASET_PATH, 
                test_folder_path=FACE_TEST_FOLDER_PATH, 
                train_folder_path=FACE_FOLDER_PATH,
                filters=None):
        super(DataSetBuilderImage, self).__init__(name, dataset_path=DATASET_PATH, 
                test_folder_path=FACE_TEST_FOLDER_PATH, 
                train_folder_path=FACE_FOLDER_PATH)
        self.image_size = image_size
        self.channels = channels
        self.images = []

    def add_img(self, img):
        self.images.append(img)

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
        self.labels = np.ndarray(shape=(max_num_images,), dtype='|S1')
        for image_index, (number_id, image_file) in enumerate(images):
            image_data = io.imread(image_file)
            if image_data.shape != dim:
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            image_data = image_data.astype(float)
            self.dataset[image_index] = self.processing(image_data, PreprocessingImage)
            self.labels[image_index] = number_id

    @classmethod
    def save_images(self, url, number_id, images):
        if not os.path.exists(url):
            os.makedirs(url)
        n_url = "{}{}/".format(url, number_id)
        if not os.path.exists(n_url):
             os.makedirs(n_url)
        for i, image in enumerate(images):
            io.imsave("{}face-{}-{}.png".format(n_url, number_id, i), image)

    def clean_directory(self, path):
        import shutil
        shutil.rmtree(path)

    def original_to_images_set(self, url, test_folder_data=False):
        images_data, labels = self.labels_images(url)
        images = (PreprocessingImage(img, self.get_transforms("global")).image for img in images_data)
        image_train, image_test = self.build_train_test(zip(labels, images), sample=test_folder_data)
        for number_id, images in image_train.items():
            self.save_images(self.train_folder_path, number_id, images)

        for number_id, images in image_test.items():
            self.save_images(self.test_folder_path, number_id, images)

    def build_dataset(self):
        self.images_to_dataset(self.train_folder_path)
        self.save_dataset()
        self.clean_directory(self.train_folder_path)

    def build_train_test(self, process_images, sample=True):
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

    def copy(self):
        dataset = super(DataSetBuilderImage, self).copy()
        dataset.image_size = self.image_size
        dataset.images = []
        dataset.transforms = self.transforms
        dataset.channels = self.channels
        return dataset

    def to_raw(self):
        raw = super(DataSetBuilderImage, self).to_raw()
        new = {
            'local_filters': self.transforms.get_transforms("local"),
            'array_length': self.image_size}
        raw.update(new)
        return raw

    def from_raw(self, raw_data):
        super(DataSetBuilderImage, self).from_raw(raw_data)
        self.transforms.add_transforms("local", raw_data["local_filters"])
        self.image_size = raw_data["array_length"]
        self.desfragment()

    def info(self):
        super(DataSetBuilderImage, self).info()
        print('Image Size {}x{}'.format(self.image_size, self.image_size))

        if not self.transforms.empty():
            print('Local filters: {}'.format(self.transforms.get_transforms("local")))


class DataSetBuilderFile(DataSetBuilder):
    def from_csv(self, path, label_column, processing_class):
        self.dataset, self.labels = self.csv2dataset(path, label_column)
        self.dataset = self.processing(self.dataset, processing_class)

    @classmethod
    def csv2dataset(self, path, label_column):
        df = pd.read_csv(path)
        dataset = df.drop([label_column], axis=1).as_matrix()
        labels = df[label_column].as_matrix()
        return dataset, labels

    def build_dataset(self, path, label_column, processing_class=Preprocessing):
        self.from_csv(path, label_column, processing_class)
        self.save_dataset()
