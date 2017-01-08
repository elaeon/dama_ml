"""
Module for create datasets from distinct sources of data.
"""
from skimage import io

import os
import numpy as np
import pandas as pd
import cPickle as pickle
import random
import h5py
import logging

from ml.processing import PreprocessingImage, Preprocessing, Transforms
from ml.utils.config import get_settings

settings = get_settings("ml")
logging.basicConfig()
log = logging.getLogger(__name__)

def save_metadata(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_metadata(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def dtype_c(dtype):
    types = set(["float64", "float32", "int64", "int32"])
    if hasattr(np, dtype) and dtype in types:
        return getattr(np, dtype)


class ReadWriteData(object):
    
    def _set_space_shape(self, f, name, shape):
        f['data'].create_dataset(name, shape, dtype=dtype_c(self.dtype), 
                                 chunks=True)

    def _set_space_data(self, f, name, data):
        f['data'].create_dataset(name, data.shape, dtype=dtype_c(self.dtype), 
                                data=self.dtype_t(data), chunks=True)

    def _set_data(self, f, name, data):
        key = '/data/' + name
        f[key] = data

    def _get_data(self, name):
        with h5py.File(self.url(), 'r') as f:
            key = '/data/' + name
            return f[key]

    def _set_attr(self, f, name, value):
        f.attrs[name] = value

    def _get_attr(self, name):
        with h5py.File(self.url(), 'r') as f:
            return f.attrs[name]

    def chunks_writer(self, f, name, data, chunks=128, init=0):
        from ml.utils.seq import grouper_chunk
        for i, row in enumerate(grouper_chunk(128, data), init+1):
            seq = np.asarray(list(row))
            end = i * seq.shape[0]
            f[name][init:end] = seq
            init = end


class DataLabel(ReadWriteData):
    """
    Base class for dataset build. Get data from memory.
    create the initial values for the dataset.

    :type name: string
    :param name: dataset's name

    :type dataset_path: string
    :param dataset_path: path where the datased is saved. This param is automaticly set by the settings.cfg file.

    :type data_folder_path: string
    :param data_folder_path: path to the data what you want to add to the dataset, automaticly split the data in train, test and validation. If you want manualy split the data in train and test, check test_folder_path.

    :type transforms: transform instance
    :param transforms: list of transforms

    :type transforms_apply: bool
    :param transforms_apply: apply transformations to the data

    :type dtype: string
    :param dtype: the type of the data to save

    :type description: string
    :param description: an bref description of the dataset

    :type author: string
    :param author: Dataset Author's name
    """
    def __init__(self, name=name, 
                dataset_path=None, 
                data_folder_path=None, 
                transforms=None,
                transforms_apply=True,
                dtype='float64',
                description='',
                author='',
                compression_level=0):
        self.data_folder_path = data_folder_path
        self.dtype = dtype

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path
        self.name = name

        if not self._preload_attrs():
            self.transforms_apply = transforms_apply
            self.autor = author
            self.description = description
            self.compression_level = compression_level
            self.transforms = transforms#Transforms([transforms_global, transforms_row])

    @property
    def data(self):
        return self._get_data('data')

    @property
    def labels(self):
        return self._get_data('labels')

    def url(self):
        """
        return the path where is saved the dataset
        """
        return os.path.join(self.dataset_path, self.name)

    def num_features(self):
        """
        return the number of features of the dataset
        """
        return self.data.shape[1]

    @property
    def shape(self):
        "return the shape of the dataset"
        return self.data.shape

    def labels_info(self):
        """
        return a counter of labels
        """
        from collections import Counter
        dl = self.desfragment()
        return Counter(dl.labels)

    def only_labels(self, labels):
        """
        :type labels: list
        :param labels: list of labels

        return a tuple of arrays with data and labels, the returned data only have the labels selected.
        """
        dl = self.desfragment()
        s_labels = set(dl.labels)
        try:
            dataset, n_labels = zip(*filter(lambda x: x[1] in s_labels, zip(dl.data, dl.labels)))
        except ValueError:
            label = labels[0] if len(labels) > 0 else None
            log.warning("label {} is not found in the labels set".format(label))
            return np.asarray([]), np.asarray([])
        return np.asarray(dataset), np.asarray(n_labels)

    def desfragment(self):
        """
        Concatenate the train, valid and test data in a data array.
        Concatenate the train, valid, and test labels in another array.
        return DataLabel
        """
        return self

    def dtype_t(self, data):
        """
        :type data: narray
        :param data: narray to cast

        cast the data to the predefined dataset dtype
        """
        dtype = dtype_c(self.dtype)
        if data.dtype is not dtype and data.dtype != np.object:
            return data.astype(dtype_c(self.dtype))
        else:
            return data

    def _open_attrs(self):
        self.create_route()
        f = h5py.File(self.url(), 'w')
        f.attrs['path'] = self.url()
        f.attrs['timestamp'] = datetime.datetime.utcnow().strftime("%Y-%M-%dT%H:%m UTC")
        f.attrs['autor'] = self.author
        f.attrs['transforms'] = self.transforms.to_json()
        f.attrs['description'] = self.description
        f.attrs['applied_transforms'] = self.transforms_apply

        if 0 < self.compression_level <= 9:
            params = {"compression": "gzip", "compression_opts": self.compression_level}
        else:
            params = {}

        f.create_group("data", **params)
        return f

    def _preload_attrs(self):
        try:
            with h5py.File(self.url(), 'r') as f:
                self.author = f.attrs['autor']
                self.transforms = f.attrs['transforms']
                self.description = f.attrs['description']
                self.apply_transforms = f.attrs['applied_transforms']
        except:
            return False
        else:
            return True

    def info(self, classes=False):
        """
        :type classes: bool
        :param classes: if true, print the detail of the labels

        This function print the details of the dataset.
        """
        from ml.utils.order import order_table_print
        print('       ')
        print('DATASET NAME: {}'.format(self.name))
        print('Author: {}'.format(self.author))
        print('Transforms: {}'.format(self.transforms.to_json()))
        print('Applied transforms: {}'.format(self.transforms_apply))
        print('Preprocessing Class: {}'.format(self.get_processing_class_name()))
        print('MD5: {}'.format(self.md5()))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Dataset", "Mean", "Std", "Shape", "dType", "Labels"]
        table = []
        table.append(["dataset", self.data[:].mean(), self.data[:].std(), 
            self.data.shape, self.data.dtype, self.labels.size])

    def is_binary(self):
        """
        return true if the labels only has two classes
        """
        return len(self.labels_info()) == 2

    def calc_md5(self):
        import hashlib
        dl = self.desfragment()
        h = hashlib.md5(dl.data)
        return h.hexdigest()

    def md5(self):
        """
        return the signature of the dataset in hex md5
        """
        return self._get_attr("md5")

    def distinct_data(self):
        """
        return the radio of distincts elements in the training data.
        i.e 
        [1,2,3,4,5] return 5/5
        [2,2,2,2,2] return 1/5        
        
        """
        if not isinstance(self.data.dtype, object):
            data = self.data[:].reshape(self.data.shape[0], -1)
        else:
            data = np.asarray([row.reshape(1, -1)[0] for row in self.data])
        y = set((elem for row in data for elem in row))
        return float(len(y)) / data.size

    def sparcity(self):
        """
        return a value between [0, 1] of the sparcity of the dataset.
        0 no zeros exists, 1 all data is zero.
        """
        if not isinstance(self.data.dtype, object):
            data = self.data[:].reshape(self.data.shape[0], -1)
        else:
            data = np.asarray([row.reshape(1, -1)[0] for row in self.data])

        zero_counter = 0
        total = 0
        for row in data:
            for elem in row:
                if elem == 0:
                    zero_counter += 1
                total += 1
    
        return float(zero_counter) / total

    def build_dataset_from_dsb(self, dsb):
        f = self._open_attrs()
        labels_shape = (dsb.shape[0], dsb.train_labels[1]) 
        self._set_space_data(f, "data", dsb.shape)
        self._set_space_data(f, "labels", labels_shape)

        self.chunks_writer(f, "data", dsb.train_data, chunks=100)
        self.chunks_writer(f, "data", dsb.test_data, chunks=100, init=dsb.train_data.shape[0])
        self.chunks_writer(f, "data", dsb.validation_data, chunks=100, init=dsb.test_data.shape[0])

        self.chunks_writer(f, "labels", dsb.train_labels, chunks=100)
        self.chunks_writer(f, "labels", dsb.test_labels, chunks=100, init=dsb.train_labels.shape[0])
        self.chunks_writer(f, "labels", dsb.validation_labels, chunks=100, init=dsb.test_labels[0])

        f.close()

    def build_dataset(self, data, labels):
        f = self._open_attrs()
        data = self.processing(data, initial=True)
        self._set_space_data(f, 'data', data)
        self._set_space_data(f, 'labels', labels)
        self._set_attr(f, "md5", self.calc_md5())
        f.close()


class DataSetBuilder(DataLabel):
    """
    Base class for dataset build. Get data from memory.
    create the initial values for the dataset.

    :type name: string
    :param name: dataset's name

    :type dataset_path: string
    :param dataset_path: path where the datased is saved. This param is automaticly set by the settings.cfg file.

    :type train_folder_path: string
    :param train_folder_path: path to the data what you want to add to the dataset, automaticly split the data in train, test and validation. If you want manualy split the data in train and test, check test_folder_path.

    :type test_folder_path: string
    :param test_folder_path: path to the test data. If None the test data is get from train_folder_path.

    :type transforms_global: list
    :param transforms_global: list of transforms for all row x column

    :type transforms_row: list
    :param transforms_row: list of transforms for each row

    :type transforms_apply: bool
    :param transforms_apply: apply transformations to the data

    :type processing_class: class
    :param processing_class: class where are defined the functions for preprocessing data.

    :type train_size: float
    :param train_size: value between [0, 1] who determine the size of the train data

    :type valid_size: float
    :param valid_size: value between [0, 1] who determine the size of the validation data

    :type validator: string
    :param validator: name of the method for extract from the data, the train data, test data and valid data

    :type dtype: string
    :param dtype: the type of the data to save

    :type description: string
    :param description: an bref description of the dataset

    :type author: string
    :param author: Dataset Author's name
    """
    def __init__(self, name=name, 
                dataset_path=None, 
                test_folder_path=None, 
                train_folder_path=None,
                transforms_row=None,
                transforms_global=None,
                transforms_apply=True,
                processing_class=None,
                train_size=.7,
                valid_size=.1,
                validator='cross',
                dtype='float64',
                description='',
                author='',
                compression_level=0):
        self.test_folder_path = test_folder_path
        self.train_folder_path = train_folder_path
        self.dtype = dtype

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path
        self.name = name
        self.processing_class = processing_class
        self.valid_size = valid_size
        self.train_size = train_size
        self.test_size = round(1 - (train_size + valid_size), 2)
        self.transforms_apply = transforms_apply
        self.validator = validator
        self.autor = author
        self.description = description
        self.compression_level = compression_level

        if transforms_row is None:
            transforms_row = ('row', [])
        else:
            transforms_row = ("row", transforms_row)

        if transforms_global is None:
            transforms_global = ("global", [])
        else:
            transforms_global = ("global", transforms_global)

        self.transforms = Transforms([transforms_global, transforms_row])

    @property
    def train_data(self):
        return self._get_data('train_data')

    @property
    def train_labels(self):
        return self._get_data('train_labels')

    @property
    def test_data(self):
        return self._get_data('test_data')

    @property
    def test_labels(self):
        return self._get_data('test_labels')

    @property
    def validation_data(self):
        return self._get_data('validation_data')

    @property
    def validation_labels(self):
        return self._get_data('validation_labels')

    @property
    def data(self):
        return self.train_data

    def labels(self):
        return self.train_labels

    @property
    def shape(self):
        "return the shape of the dataset"
        rows = self.train_data.shape[0] + self.test_data.shape[0] +\
            self.validation_data.shape[0]
        if self.train_data.dtype != np.object:
            return tuple([rows] + list(self.train_data.shape[1:]))
        else:
            return (rows,)

    def desfragment(self):
        """
        Concatenate the train, valid and test data in a data array.
        Concatenate the train, valid, and test labels in another array.
        return data, labels
        """
        dl = DataLabel(
            name=self.name+"dl",
            dataset_path=self.dataset_path,
            data_folder_path=self.data_folder_path,
            transforms=self.transforms,
            transforms_apply=self.transforms_apply,
            dtype=self.dtype,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level)
        dl.build_dataset_from_dsb(self)
        return dl

    def info(self, classes=False):
        """
        :type classes: bool
        :param classes: if true, print the detail of the labels

        This function print the details of the dataset.
        """
        from ml.utils.order import order_table_print
        print('       ')
        print('DATASET NAME: {}'.format(self.name))
        print('Author: {}'.format(self.author))
        print('Transforms: {}'.format(self.transforms.to_json()))
        print('Applied transforms: {}'.format(self.transforms_apply))
        print('Preprocessing Class: {}'.format(self.get_processing_class_name()))
        print('MD5: {}'.format(self.md5()))
        print('Description: {}'.format(self.description))
        print('       ')
        if self.train_data.dtype != np.object:
            headers = ["Dataset", "Mean", "Std", "Shape", "dType", "Labels"]
            table = []
            table.append(["train set", self.train_data.mean(), self.train_data.std(), 
                self.train_data.shape, self.train_data.dtype, self.train_labels.size])

            if self.valid_data is not None:
                table.append(["valid set", self.valid_data.mean(), self.valid_data.std(), 
                self.valid_data.shape, self.valid_data.dtype, self.valid_labels.size])

            table.append(["test set", self.test_data.mean(), self.test_data.std(), 
                self.test_data.shape, self.test_data.dtype, self.test_labels.size])
            order_table_print(headers, table, "shape")
        else:
            headers = ["Dataset", "Shape", "dType", "Labels"]
            table = []
            table.append(["train set", self.train_data.shape, self.train_data.dtype, 
                self.train_labels.size])

            if self.valid_data is not None:
                table.append(["valid set", self.valid_data.shape, self.valid_data.dtype, 
                self.valid_labels.size])

            table.append(["test set", self.test_data.shape, self.test_data.dtype, 
                self.test_labels.size])
            order_table_print(headers, table, "shape")

        if classes is True:
            headers = ["class", "# items"]
            order_table_print(headers, self.labels_info().items(), "# items")

    def cross_validators(self, data, labels):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, train_size=round(self.train_size+self.valid_size, 2), random_state=0)

        valid_size_index = int(round(data.shape[0] * self.valid_size))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def get_processing_class_name(self):
        """
        return the name of the processing class
        """
        if self.processing_class is None:
            return None
        else:
            return self.processing_class.module_cls_name()

    #def to_raw(self):
    #    return {
    #        'train_dataset': self.dtype_t(self.train_data),
    #        'train_labels': self.train_labels,
    #        'valid_dataset': self.dtype_t(self.valid_data),
    #        'valid_labels': self.valid_labels,
    #        'test_dataset': self.dtype_t(self.test_data),
    #        'test_labels': self.test_labels,
    #        'transforms': self.transforms.get_all_transforms(),
    #        'preprocessing_class': self.get_processing_class_name(),
    #        'applied_transforms': self.transforms_apply,
    #        'md5': self.md5()}

    #@classmethod
    #def from_raw_to_ds(self, name, dataset_path, data, save=True):
    #    ds = DataSetBuilder(name, 
    #            dataset_path=dataset_path)
    #    ds.from_raw(data)
    #    if save is True:
    #        ds.save()
    #    return ds

    #def from_raw(self, raw_data):
    #    from pydoc import locate
        #if self.processing_class is None:
    #    if raw_data["preprocessing_class"] is not None:
    #        self.processing_class = locate(raw_data["preprocessing_class"])

    #    self.transforms = Transforms(raw_data["transforms"])
    #    self.train_data = self.dtype_t(raw_data['train_dataset'])
    #    self.train_labels = raw_data['train_labels']
    #    self.valid_data = self.dtype_t(raw_data['valid_dataset'])
    #    self.valid_labels = raw_data['valid_labels']
    #    self.test_data = self.dtype_t(raw_data['test_dataset'])
    #    self.test_labels = raw_data['test_labels']        
    #    self._cached_md5 = raw_data["md5"]

    #def shuffle_and_save(self, data, labels):
    #    self.train_data, self.valid_data, self.test_data, self.train_labels, self.valid_labels, self.test_labels = 
        #self.save()

    def adversarial_validator(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.valid_data = train_data[0:1]
        self.valid_labels = train_labels[0:1]

        # train class is labeled as 1 
        train_test_data_clf = self._clf()
        train_test_data_clf.train()
        class_train = np.argmax(train_test_data_clf.model.classes_)
        predictions = [p[class_train] 
            for p in train_test_data_clf.predict(self.train_data, raw=True, transform=False)]
        predictions = sorted(enumerate(predictions), key=lambda x: x[1], reverse=False)
        #print([(pred, index) for index, pred in predictions if pred < .5])
        #because all targets are ones (train data) is not necessary compare it
        false_test = [index for index, pred in predictions if pred < .5] # is a false test 
        ok_train = [index for index, pred in predictions if pred >= .5]
 
        valid_data = self.train_data[false_test]
        valid_labels = self.train_labels[false_test]
        valid_size = int(round(self.train_data.shape[0] * self.valid_size))
        self.valid_data = valid_data[:int(round(valid_size))]
        self.valid_labels = valid_labels[:int(round(valid_size))]
        self.train_data = np.concatenate((
            valid_data[int(round(valid_size)):], 
            self.train_data[ok_train]),
            axis=0)
        self.train_labels = np.concatenate((
            valid_labels[int(round(valid_size)):],
            self.train_labels[ok_train]),
            axis=0)
        self.save()

    def create_route(self):
        """
        create directories if the dataset_path does not exist
        """
        if self.dataset_path is not None:
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)

    #@classmethod
    #def load_dataset_raw(self, name, dataset_path=None):
    #    with open(dataset_path+name, 'rb') as f:
    #        save = pickle.load(f)
    #        return save

    @classmethod
    def load_dataset(self, name, dataset_path=None, info=True, dtype='float64',
                    transforms_apply=False):
        """
        :type name: string
        :param name: name of the dataset to load

        :type dataset_path: string
        :param dataset_path: path where the dataset was saved

        :type dtype: string
        :param dtype: cast the data to the defined type

        dataset_path is not necesary to especify, this info is obtained from settings.cfg
        """
        if dataset_path is None:
             dataset_path = settings["dataset_path"]
        #data = self.load_dataset_raw(name, dataset_path=dataset_path)
        dataset = DataSetBuilder(name, dataset_path=dataset_path, dtype=dtype, 
                                transforms_apply=transforms_apply)
        #dataset.from_raw(data)
        if info:
            dataset.info()
        return dataset

    def convert(dtype='float64', transforms_apply=False):
        """
        :type dtype: string
        :param dtype: cast the data to the defined type

        dataset_path is not necesary to especify, this info is obtained from settings.cfg
        """
        dataset = DataSetBuilder(name=self.name, dataset_path=self.dataset_path, 
                                dtype=dtype, transforms_apply=transforms_apply)
        dataset.info()
        return dataset

    @classmethod
    def to_DF(self, dataset, labels):
        if len(dataset.shape) > 2:
            dataset = dataset.reshape(dataset.shape[0], -1)
        columns_name = map(lambda x: "c"+str(x), range(dataset.shape[-1])) + ["target"]
        return pd.DataFrame(data=np.column_stack((dataset, labels)), columns=columns_name)

    def to_df(self):
        """
        convert the dataset to a dataframe
        """
        dl = self.desfragment()
        return self.to_DF(dl.data[:], dl.labels[:])

    def processing_rows(self, data):
        if not self.transforms.empty('row') and self.transforms_apply and data is not None:
            pdata = []
            for row in data:
                preprocessing = self.processing_class(row, self.transforms.get_transforms('row'))
                pdata.append(preprocessing.pipeline())
            return np.asarray(pdata)
        else:
            return data if isinstance(data, np.ndarray) else np.asarray(data)

    #def processing_global(self, data, base_data=None):
    #    if not self.transforms.empty('global') and self.transforms_apply and data is not None:
    #        from pydoc import locate
    #        fiter, params = self.transforms.get_transforms('global')[0]
    #        fiter = locate(fiter)
    #        if isinstance(params, dict):
    #            self.fit = fiter(**params)
    #        else:
    #            self.fit = fiter()
    #        print(base_data)
    #        self.fit.fit(base_data)
    #        return self.fit.transform(data)
            #else:
            #    return self.fit.transform(data)
    #    else:
    #        return data

    def build_dataset(self, data, labels, test_data=None, test_labels=None, 
                        valid_data=None, valid_labels=None):
        """
        :type data: ndarray
        :param data: array of values to save in the dataset

        :type labels: ndarray
        :param labels: array of labels to save in the dataset
        """
        f = self._open_attrs()

        if self.validator == 'cross':
            if test_data is not None and test_labels is not None:
                data = np.concatenate((data, test_data), axis=0)
                labels = np.concatenate((labels, test_labels), axis=0)
            data_labels = self.cross_validators(data, labels)
        elif self.validator == 'adversarial':
            data_labels = self.adversarial_validator(data, labels, test_data, test_labels)

        train_data = self.processing(data_labels[0], initial=True)        
        validation_data = self.processing(data_labels[1])
        test_data = self.processing(data_labels[2])
        
        self._set_space_data(f, 'train_data', train_data)
        self._set_space_data(f, 'test_data', test_data)
        self._set_space_data(f, 'validation_data', validation_data)

        self._set_space_data(f, 'train_labels', data_labels[3])
        self._set_space_data(f,  'test_labels', data_labels[5])
        self._set_space_data(f, 'validation_labels', data_labels[4])

        self._set_attr(f, "md5", self.calc_md5())
        f.close()

    #def copy(self, limit=None):
    #    dataset = DataSetBuilder(self.name)
    #    def calc_nshape(data, value):
    #        if value is None or not (0 < value <= 1) or data is None:
    #            value = 1

    #        limit = int(round(data.shape[0] * value, 0))
    #        return data[:limit]

    #    dataset.test_folder_path = self.test_folder_path
    #    dataset.train_folder_path = self.train_folder_path
    #    dataset.dataset_path = self.dataset_path
    #    dataset.train_data = calc_nshape(self.train_data, limit)
    #    dataset.train_labels = calc_nshape(self.train_labels, limit)
    #    dataset.valid_data = calc_nshape(self.valid_data, limit)
    #    dataset.valid_labels = calc_nshape(self.valid_labels, limit)
    #    dataset.test_data = calc_nshape(self.test_data, limit)
    #    dataset.test_labels = calc_nshape(self.test_labels, limit)
    #    dataset.transforms = self.transforms
    #    dataset.processing_class = self.processing_class
    #    dataset.md5()
    #    return dataset

    def processing(self, data, initial=True):
        data = self.processing_rows(data)
        #if init is True:
        #    return self.processing_global(data, base_data=data)
        #elif init is False and not self.transforms.empty('global'):
        #    base_data, _ = self.desfragment()
        #    return self.processing_global(data, base_data=base_data)
        #else:
        #    return data
        return data

    def subset(self, percentaje):
        """
        :type percentaje: float
        :param percentaje: value between [0, 1], this value represent the size of the dataset to copy.

        i.e 1 copy all dataset, .5 copy half dataset
        """
        return self.copy(limit=percentaje)

    def _clf(self):
        from ml.clf.extended.w_sklearn import RandomForest
        train_labels = np.ones(self.train_labels.shape[0], dtype=int)
        test_labels = np.zeros(self.test_labels.shape[0], dtype=int)
        data = np.concatenate((self.train_data, self.test_data), axis=0)
        labels = np.concatenate((train_labels, test_labels), axis=0)
        dataset = DataSetBuilder("test_train_separability", transforms_apply=False)
        dataset.build_dataset(data, labels)
        return RandomForest(dataset=dataset)

    def score_train_test(self):
        """
        return the score of separability between the train data and the test data.
        """
        classif = self._clf()
        classif.train()
        measure = "auc"
        return classif.load_meta().get("score", {measure, None}).get(measure, None) 

    def plot(self):
        import matplotlib.pyplot as plt
        last_transform = self.transforms.get_transforms("row")[-1]
        data, labels = self.desfragment()
        if last_transform[0] == "tsne":
            if last_transform[1]["action"] == "concatenate":                
                dim = 2
                features_tsne = data[:,-dim:]
            else:
                features_tsne = data
        else:
            features_tsne = ml.processing.Preprocessing(data, [("tsne", 
                {"perplexity": 50, "action": "replace"})])

        classes = self.labels_info().keys()
        colors = ['b', 'r', 'y', 'm', 'c']
        classes_colors = dict(zip(classes, colors))
        fig, ax = plt.subplots(1, 1, figsize=(17.5, 17.5))

        r_indexes = {}        
        for index, target in enumerate(labels):
            r_indexes.setdefault(target, [])
            r_indexes[target].append(index)

        for target, indexes in r_indexes.items():
            features_index = features_tsne[indexes]
            ax.scatter(
                features_index[:,0], 
                features_index[:,1], 
                color=classes_colors[target], 
                marker='o',
                alpha=.4,
                label=target)
         
        ax.set(xlabel='X',
               ylabel='Y',
               title=self.name)
        ax.legend(loc=2)
        plt.show()

    def add_transforms(self, transforms):
        """
        :type transforms_global: list
        :param transforms_global: list of global transforms to apply

        :type transforms_row: list
        :param transforms_row: list of row transforms to apply

        :type processing_class: class
        :param processing_class: class of the row transforms
        """
        dataset = self.copy()
        if old_transforms.empty("global") and transforms_global is not None:
            dataset.transforms = Transforms([transforms_global, transforms_row])
        else:
            dataset.transforms = transforms
            dataset.train_data = dataset.processing(dataset.train_data)
            dataset.test_data = dataset.processing(dataset.test_data)
            dataset.valid_data = dataset.processing(dataset.valid_data)
            dataset.transforms = self.transforms + dataset.transforms

        return dataset


class DataSetBuilderImage(DataSetBuilder):
    """
    Class for images dataset build. Get the data from a directory where each directory's name is the label.
    
    :type image_size: int
    :param image_size: define the image size to save in the dataset

    kwargs are the same that DataSetBuilder's options
    """
    def __init__(self, name, image_size=None, **kwargs):
        super(DataSetBuilderImage, self).__init__(name, **kwargs)
        self.image_size = image_size
        self.images = []

    def add_img(self, img):
        self.images.append(img)

    def images_from_directories(self, directories):
        if isinstance(directories, str):
            directories = [directories]
        elif isinstance(directories, list):
            pass
        else:
            raise Exception

        images = []
        for root_directory in directories:
            for directory in os.listdir(root_directory):
                files = os.path.join(root_directory, directory)
                if os.path.isdir(files):
                    number_id = directory
                    for image_file in os.listdir(files):
                        images.append((number_id, os.path.join(files, image_file)))
        return images

    def images_to_dataset(self, folder_base):
        images = self.images_from_directories(folder_base)
        labels = np.ndarray(shape=(len(images),), dtype='|S1')
        data = []
        for image_index, (number_id, image_file) in enumerate(images):
            img = io.imread(image_file)
            data.append(img) 
            labels[image_index] = number_id
        data = self.processing(data)
        return data, labels

    @classmethod
    def save_images(self, url, number_id, images, rewrite=False):
        if not os.path.exists(url):
            os.makedirs(url)
        n_url = os.path.join(url, number_id)
        if not os.path.exists(n_url):
             os.makedirs(n_url)

        initial = 0 if rewrite else len(os.listdir(n_url)) 
        for i, image in enumerate(images, initial):
            try:
                image_path = "img-{}-{}.png".format(number_id, i)
                io.imsave(os.path.join(n_url, image_path), image)
            except IndexError:
                print("Index error", n_url, number_id)

    def clean_directory(self, path):
        import shutil
        shutil.rmtree(path)

    def original_to_images_set(self, url, test_folder_data=False):
        images_data, labels = self.labels_images(url)
        images = (self.processing(img, 'image') for img in images_data)
        image_train, image_test = self.build_train_test(zip(labels, images), sample=test_folder_data)
        for number_id, images in image_train.items():
            self.save_images(self.train_folder_path, number_id, images)

        for number_id, images in image_test.items():
            self.save_images(self.test_folder_path, number_id, images)

    def build_dataset(self):
        """
        the data is extracted from the train_folder_path, and then saved.
        """
        data, labels = self.images_to_dataset(self.train_folder_path)
        self.shuffle_and_save(data, labels)

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
        return dataset

    def to_raw(self):
        raw = super(DataSetBuilderImage, self).to_raw()
        new = {'array_length': self.image_size}
        raw.update(new)
        return raw

    def from_raw(self, raw_data):
        super(DataSetBuilderImage, self).from_raw(raw_data)
        #self.transforms.add_transforms("local", raw_data["local_filters"])
        self.image_size = raw_data["array_length"]
        self.desfragment()

    def info(self):
        super(DataSetBuilderImage, self).info()
        print('Image Size {}x{}'.format(self.image_size, self.image_size))


class DataSetBuilderFile(DataSetBuilder):
    """
    Class for csv dataset build. Get the data from a csv's file.
    """
    def from_csv(self, folder_path, label_column):
        """
        :type folder_path: string
        :param folder_path: path to the csv.

        :type label_column: string
        :param label_column: column's name where are the labels
        """
        data, labels = self.csv2dataset(folder_path, label_column)
        return data, labels

    @classmethod
    def csv2dataset(self, path, label_column):
        df = pd.read_csv(path)
        dataset = df.drop([label_column], axis=1).as_matrix()
        labels = df[label_column].as_matrix()
        return dataset, labels

    def build_dataset(self, label_column=None):
        """
         :type label_column: string
         :param label_column: column's name where are the labels
        """
        data, labels = self.from_csv(self.train_folder_path, label_column)
        data = self.processing(data)
        if self.test_folder_path is not None:
            raise #NotImplemented
            #test_data, test_labels = self.from_csv(self.test_folder_path, label_column)
            #if self.validator == 'cross':
            #    data = np.concatenate((data, test_data), axis=0)
            #    labels = np.concatenate((labels, test_labels), axis=0)

        if self.validator == 'cross':
            self.shuffle_and_save(data, labels)
        else:
            self.adversarial_validator_and_save(data, labels, test_data, test_labels)

    @classmethod
    def merge_data_labels(self, data_path, labels_path, column_id):
        import pandas as pd
        data_df = pd.read_csv(data_path)
        labels_df = pd.read_csv(labels_path)
        return pd.merge(data_df, labels_df, on=column_id)


