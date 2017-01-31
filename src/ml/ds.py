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
import datetime
import uuid

from ml.processing import Transforms
from ml.utils.config import get_settings

settings = get_settings("ml")

logging.basicConfig()
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(console)


def save_metadata(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_metadata(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    except IOError:
        return {}


def calc_nshape(data, value):
    if value is None or not (0 < value <= 1) or data is None:
        value = 1

    limit = int(round(data.shape[0] * value, 0))
    return data[:limit]


class ReadWriteData(object):

    def auto_dtype(self, data, ttype):
        if ttype == "auto" and data is not None:
            return data.dtype
        elif ttype == "auto" and isinstance(data, type(None)):
            return "float64"
        else:
            return np.dtype(ttype)

    def _set_space_shape(self, f, name, shape, label=False):
        dtype = self.auto_dtype(None, self.dtype) if label is False else self.auto_dtype(None, self.ltype)
        f['data'].create_dataset(name, shape, dtype=dtype, chunks=True, **self.zip_params)

    def _set_space_data(self, f, name, data, label=False):
        dtype = self.auto_dtype(data, self.dtype) if label is False else self.auto_dtype(data, self.ltype)
        f['data'].create_dataset(name, data.shape, dtype=dtype, data=data, chunks=True, **self.zip_params)

    def _set_data(self, f, name, data):
        key = '/data/' + name
        f[key] = data

    def _get_data(self, name):
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.url(), 'r')
        key = '/data/' + name
        return self.f[key]

    def _set_attr(self, name, value):
        while True:
            try:
                with h5py.File(self.url(), 'r+') as f:
                    f.attrs[name] = value
                break
            except IOError:
                self.f.close()
                del self.f
            
    def _get_attr(self, name):
        try:
            with h5py.File(self.url(), 'r') as f:
                return f.attrs[name]
        except KeyError:
            return None
        except IOError:
            log.debug("Error found in file {}".format(self.url()))
            return None

    def chunks_writer(self, f, name, data, chunks=128, init=0):
        from ml.utils.seq import grouper_chunk
        end = init
        for row in grouper_chunk(chunks, data):
            seq = np.asarray(list(row))
            end += seq.shape[0]
            #print("init:{}, end:{}, shape:{}, chunks:{}".format(init, end, seq.shape, chunks))
            f[name][init:end] = seq
            init = end
        return end


class DataLabel(ReadWriteData):
    """
    Base class for dataset build. Get data from memory.
    create the initial values for the dataset.

    :type name: string
    :param name: dataset's name

    :type dataset_path: string
    :param dataset_path: path where the datased is saved. This param is automaticly set by the settings.cfg file.

    :type transforms: transform instance
    :param transforms: list of transforms

    :type apply_transforms: bool
    :param apply_transforms: apply transformations to the data

    :type dtype: string
    :param dtype: the type of the data to save

    :type description: string
    :param description: an bref description of the dataset

    :type author: string
    :param author: Dataset Author's name

    :type compression_level: int
    :param compression_level: number in 0-9 range. If 0 is passed no compression is executed

    :type rewrite: bool
    :param rewrite: if true, you can clean the saved data and add a new dataset.
    """
    def __init__(self, name=None, 
                dataset_path=None,
                transforms=None,
                apply_transforms=True,
                dtype='float64',
                ltype='|S1',
                description='',
                author='',
                compression_level=0,
                chunks=100,
                rewrite=True):
        self.name = name
        self._applied_transforms = False
        self.chunks = chunks
        self.rewrite = rewrite

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path
        
        if transforms is None:
            transforms = Transforms()

        if not self._preload_attrs() or self.rewrite is True:
            self.apply_transforms = apply_transforms
            self.author = author
            self.description = description
            self.compression_level = compression_level
            self.dtype = dtype
            self.ltype = ltype
            self.transforms = transforms
            self.mode = "w"
        else:
            self.mode = "r"

    @property
    def data(self):
        """
        eturn the data in the dataset
        """
        return self._get_data('data')

    @property
    def labels(self):
        """
        return the labels in the dataset
        """
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
        counter = Counter(self.labels)
        return counter

    def only_labels(self, labels):
        """
        :type labels: list
        :param labels: list of labels

        return a tuple of arrays with data and labels, the returned data only have the labels selected.
        """
        try:
            dl = self.desfragment()
            s_labels = set(labels)
            dataset, n_labels = zip(*filter(lambda x: x[1] in s_labels, zip(dl.data, dl.labels)))
            dl.destroy()
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
        return self.copy()

    def type_t(self, ttype, data):
        """
        :type ttype: string
        :param ttype: name of the type to convert the data. If ttype is 'auto' 
        the data is returned without be converted.

        :type data: array
        :param data: data to be converted

        convert the data to the especified ttype.
        """
        if ttype == 'auto':
            return data

        ttype = np.dtype(ttype)
        if data.dtype is not ttype and data.dtype != np.object:
            return data.astype(ttype)
        else:
            return data

    def dtype_t(self, data):
        """
        :type data: narray
        :param data: narray to cast

        cast the data to the predefined dataset dtype
        """
        return self.type_t(self.dtype, data)

    def ltype_t(self, labels):
        """
        :type labels: narray
        :param labels: narray to cast

        cast the labels to the predefined dataset ltype
        """
        return self.type_t(self.ltype, labels)

    def _open_attrs(self):
        self.create_route()
        f = h5py.File(self.url(), 'w')
        f.attrs['path'] = self.url()
        f.attrs['timestamp'] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M UTC")
        f.attrs['author'] = self.author
        f.attrs['transforms'] = self.transforms.to_json()
        f.attrs['description'] = self.description
        f.attrs['applied_transforms'] = self.apply_transforms
        f.attrs['dtype'] = self.dtype
        f.attrs['ltype'] = self.ltype
        f.attrs['compression_level'] = self.compression_level

        if 0 < self.compression_level <= 9:
            self.zip_params = {"compression": "gzip", "compression_opts": self.compression_level}
        else:
            self.zip_params = {}

        f.create_group("data")
        return f

    def _preload_attrs(self):
        try:
            with h5py.File(self.url(), 'r') as f:
                self.author = f.attrs['author']
                self.transforms = Transforms.from_json(f.attrs['transforms'])
                self.description = f.attrs['description']
                self.apply_transforms = f.attrs['applied_transforms']
                self.dtype = f.attrs['dtype']
                self.ltype = f.attrs['ltype']
                self.compression_level = f.attrs['compression_level']
            if self.md5() is None:
                return False
        except KeyError:
            return False
        except IOError:
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
        print('Applied transforms: {}'.format(self.apply_transforms))
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
        """
        calculate the md5 from the data.
        """
        import hashlib
        dl = self.desfragment()
        h = hashlib.md5(dl.data[:])
        dl.destroy()
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
        """
        Transform a dataset with train, test and validation dataset into a datalabel dataset
        """
        if self.mode == "r":
            return

        f = self._open_attrs()
        labels_shape = tuple(dsb.shape[0:1] + dsb.train_labels.shape[1:])
        self._set_space_shape(f, "data", dsb.shape)
        self._set_space_shape(f, "labels", labels_shape, label=True)

        end = self.chunks_writer(f, "/data/data", dsb.train_data, chunks=self.chunks)
        end = self.chunks_writer(f, "/data/data", dsb.test_data, chunks=self.chunks, 
                                init=end)
        self.chunks_writer(f, "/data/data", dsb.validation_data, chunks=self.chunks, 
                            init=end)

        end = self.chunks_writer(f, "/data/labels", dsb.train_labels, chunks=self.chunks)
        end = self.chunks_writer(f, "/data/labels", dsb.test_labels, chunks=self.chunks, 
                                init=end)
        self.chunks_writer(f, "/data/labels", dsb.validation_labels, chunks=self.chunks, 
                            init=end)
        f.close()

    def build_dataset(self, data, labels):
        """
        build a datalabel dataset from data and labels
        """
        f = self._open_attrs()
        data = self.processing(data, initial=True)
        self._set_space_data(f, 'data', self.dtype_t(data))
        self._set_space_data(f, 'labels', self.ltype_t(labels), label=True)
        f.close()
        self._set_attr("md5", self.calc_md5())

    def create_route(self):
        """
        create directories if the dataset_path does not exist
        """
        if self.dataset_path is not None:
            if not os.path.exists(self.dataset_path):
                os.makedirs(self.dataset_path)

    def destroy(self):
        """
        delete the correspondly hdf5 file
        """
        from ml.utils.files import rm
        self.close_reader()
        rm(self.url())
        print("rm {}".format(self.url()))

    def convert(self, name, dtype='float64', ltype='|S1', apply_transforms=False, 
                percentaje=1):
        """
        :type dtype: string
        :param dtype: cast the data to the defined type

        dataset_path is not necesary to especify, this info is obtained from settings.cfg
        """
        dl = DataLabel(name=name, 
            dataset_path=self.dataset_path,
            transforms=self.transforms,
            apply_transforms=apply_transforms,
            dtype=dtype,
            ltype=ltype,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            chunks=self.chunks,
            rewrite=self.rewrite)
        dl._applied_transforms = self.apply_transforms
        dl.build_dataset(calc_nshape(self.data, percentaje), calc_nshape(self.labels, percentaje))
        dl.close_reader()
        return dl

    def copy(self, percentaje=1):
        """
        :type percentaje: float
        :param percentaje: value between [0, 1], this value represent the size of the dataset to copy.
        
        copy the dataset, a percentaje is permited for the size of the copy
        """
        name = self.name + "_copy_" + str(percentaje)
        dl = self.convert(name, dtype=self.dtype, ltype=self.ltype, 
                        apply_transforms=self.apply_transforms, 
                        percentaje=percentaje)
        return dl

    def processing(self, data, initial=True):
        """
        :type data: array
        :param data: data to transform

        :type initial: bool
        :param initial: if multirow transforms are added, then this parameter
        indicates the initial data fit

        execute the transformations to the data.

        """
        data = self.processing_rows(data)
        #if init is True:
        #    return self.processing_global(data, base_data=data)
        #elif init is False and not self.transforms.empty('global'):
        #    base_data, _ = self.desfragment()
        #    return self.processing_global(data, base_data=base_data)
        #else:
        #    return data
        return data

    def close_reader(self):
        """
        close the hdf5 file. If is closed, no more data retrive will be perform.
        """
        if hasattr(self, 'f'):
            self.f.close()
            del self.f

    def processing_rows(self, data):
        """
        :type data: array
        :param data: data to be transformed

        each row is transformed with the transformations defined.
        """
        if not self.transforms.empty() and self.transforms_to_apply and data is not None:
            log.debug("Apply transforms")
            return np.asarray([self.transforms.apply(row) for row in data])
        else:
            log.debug("No transforms applied")
            return data if isinstance(data, np.ndarray) else np.asarray(data)

    @property
    def transforms_to_apply(self):
        return self.apply_transforms and self._applied_transforms is False

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
        df = self.to_DF(dl.data[:], dl.labels[:])
        dl.destroy()
        return df

    def add_transforms(self, name, transforms):
        """
        :type name: string
        :param name: result dataset's name

        :type transforms: Transform
        :param transforms: transforms to apply in the new dataset
        """
        if self.apply_transforms is True:
            dsb_c = self.copy()
            dsb_c.apply_transforms = False
            dsb_c.transforms = transforms
            dsb = dsb_c.convert(name, dtype=self.dtype, ltype=self.ltype, 
                apply_transforms=True, percentaje=1)
            dsb_c.destroy()
            dsb.transforms = self.transforms + transforms
        else:
            dsb = self.copy()
            dsb.transforms += transforms
        return dsb


class DataSetBuilder(DataLabel):
    """
    Base class for dataset build. Get data from memory.
    create the initial values for the dataset.

    :type name: string
    :param name: dataset's name

    :type dataset_path: string
    :param dataset_path: path where the datased is saved. This param is automaticly set by the settings.cfg file.

    :type apply_transforms: bool
    :param apply_transforms: apply transformations to the data

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

    :type compression_level: int
    :param compression_level: number in 0-9 range. If 0 is passed no compression is executed

    :type rewrite: bool
    :param rewrite: if true, you can clean the saved data and add a new dataset.

    :type chunks: int
    :param chunks: number of chunks to use when the dataset is copy or desfragmented.
    """
    def __init__(self, name=None, 
                dataset_path=None,
                apply_transforms=True,
                transforms=None,
                train_size=.7,
                valid_size=.1,
                validator='cross',
                dtype='float64',
                ltype='|S1',
                description='',
                author='',
                compression_level=0,
                chunks=100,
                rewrite=False):
        self.name = name
        self._applied_transforms = False
        self.chunks = chunks
        self.rewrite = rewrite

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path

        if transforms is None:
            transforms = Transforms()

        if not self._preload_attrs() or self.rewrite is True:
            self.dtype = dtype
            self.ltype = ltype
            self.transforms = transforms
            self.valid_size = valid_size
            self.train_size = train_size
            self.test_size = round(1 - (train_size + valid_size), 2)
            self.apply_transforms = apply_transforms
            self.validator = validator
            self.author = author
            self.description = description
            self.compression_level = compression_level
            self.mode = "w"
        else:
            self.mode = "r"

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

    @property
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

    def _open_attrs(self):
        f = super(DataSetBuilder, self)._open_attrs()
        f.attrs["validator"] = self.validator
        f.attrs["train_size"] = self.train_size 
        f.attrs["valid_size"] = self.valid_size 
        return f

    def _preload_attrs(self):
        try:
            with h5py.File(self.url(), 'r') as f:
                self.author = f.attrs['author']
                self.transforms = Transforms.from_json(f.attrs['transforms'])
                self.description = f.attrs['description']
                self.apply_transforms = f.attrs['applied_transforms']
                self.dtype = f.attrs['dtype']
                self.ltype = f.attrs['ltype']
                self.compression_level = f.attrs['compression_level']
                self.validator = f.attrs["validator"]
                self.train_size = f.attrs["train_size"]
                self.valid_size = f.attrs["valid_size"]
            if self.md5() is None:
                return False
        except KeyError:
            return False
        except IOError:
            return False
        else:
            return True

    def desfragment(self):
        """
        Concatenate the train, valid and test data in a data array.
        Concatenate the train, valid, and test labels in another array.
        return data, labels
        """
        id_ = uuid.uuid4().hex
        dl = DataLabel(
            name=self.name+id_,
            dataset_path=self.dataset_path,
            transforms=self.transforms,
            apply_transforms=self.apply_transforms,
            dtype=self.dtype,
            ltype=self.ltype,
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
        print('Applied transforms: {}'.format(self.apply_transforms))
        print('MD5: {}'.format(self.md5()))
        print('Description: {}'.format(self.description))
        print('       ')
        if self.train_data.dtype != np.object:
            headers = ["Dataset", "Mean", "Std", "Shape", "dType", "Labels"]
            table = []
            table.append(["train set", self.train_data[:].mean(), self.train_data[:].std(), 
                self.train_data.shape, self.train_data.dtype, self.train_labels.size])

            if self.validation_data is not None:
                table.append(["valid set", self.validation_data[:].mean(), self.validation_data[:].std(), 
                self.validation_data.shape, self.validation_data.dtype, self.validation_labels.size])

            table.append(["test set", self.test_data[:].mean(), self.test_data[:].std(), 
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

        if isinstance(data, list):
            size = len(data)
        else:
            size = data.shape[0]

        valid_size_index = int(round(size * self.valid_size))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

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

    #def processing_global(self, data, base_data=None):
    #    if not self.transforms.empty('global') and self.apply_transforms and data is not None:
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
                        validation_data=None, validation_labels=None):
        """
        :type data: ndarray
        :param data: array of values to save in the dataset

        :type labels: ndarray
        :param labels: array of labels to save in the dataset
        """
        if self.mode == "r":
            return

        f = self._open_attrs()

        if self.validator == '' and test_data is not None and test_labels is not None \
            and validation_data is not None and validation_labels is not None:
                data_labels = [
                    data, validation_data, test_data,
                    labels, validation_labels, test_labels]
        else:
            if self.validator == 'cross':
                data_labels = self.cross_validators(data, labels)
            elif self.validator == 'adversarial':
                data_labels = self.adversarial_validator(data, labels, test_data, test_labels)

        train_data = self.processing(data_labels[0], initial=True)        
        validation_data = self.processing(data_labels[1])
        test_data = self.processing(data_labels[2])
        
        #print("-------", train_data.dtype)
        self._set_space_data(f, 'train_data', self.dtype_t(train_data))
        self._set_space_data(f, 'test_data', self.dtype_t(test_data))
        self._set_space_data(f, 'validation_data', self.dtype_t(validation_data))

        self._set_space_data(f, 'train_labels', self.ltype_t(data_labels[3]), label=True)
        self._set_space_data(f, 'test_labels', self.ltype_t(data_labels[5]), label=True)
        self._set_space_data(f, 'validation_labels', self.ltype_t(data_labels[4]), label=True)

        f.close()
        self._set_attr("md5", self.calc_md5())

    def _clf(self):
        from ml.clf.extended.w_sklearn import RandomForest
        train_labels = np.ones(self.train_labels.shape[0], dtype=int)
        test_labels = np.zeros(self.test_labels.shape[0], dtype=int)
        data = np.concatenate((self.train_data, self.test_data), axis=0)
        labels = np.concatenate((train_labels, test_labels), axis=0)
        dataset = DataSetBuilder("test_train_separability", apply_transforms=False)
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

    def convert(self, name, dtype='float64', ltype='|S1', apply_transforms=False, 
                percentaje=1):
        """
        :type name: string
        :param name: converted dataset's name
 
        :type dtype: string
        :param dtype: cast the data to the defined type

        :type ltype: string
        :param ltype: cast the labels to the defined type

        :type apply_transforms: bool
        :param apply_transforms: apply the transforms to the data

        :type percentaje: float
        :param percentaje: values between 0 and 1, this value specify the percentaje of the data to apply transforms and cast function, then return a subset

        """
        dsb = DataSetBuilder(name=name, 
            dataset_path=self.dataset_path,
            transforms=self.transforms,
            apply_transforms=apply_transforms,
            train_size=self.train_size,
            valid_size=self.valid_size,
            validator=self.validator,
            dtype=dtype,
            ltype=ltype,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            rewrite=self.rewrite)
        dsb._applied_transforms = self.apply_transforms
        dsb.build_dataset(
            calc_nshape(self.train_data, percentaje), 
            calc_nshape(self.train_labels, percentaje),
            test_data=calc_nshape(self.test_data, percentaje),
            test_labels=calc_nshape(self.test_labels, percentaje),
            validation_data=calc_nshape(self.validation_data, percentaje),
            validation_labels=calc_nshape(self.validation_labels, percentaje))
        dsb.close_reader()
        return dsb


class DataSetBuilderImage(DataSetBuilder):
    """
    Class for images dataset build. Get the data from a directory where each directory's name is the label.
    
    :type image_size: int
    :param image_size: define the image size to save in the dataset

    kwargs are the same that DataSetBuilder's options

    :type data_folder_path: string
    :param data_folder_path: path to the data what you want to add to the dataset, split the data in train, test and validation. If you want manualy split the data in train and test, check test_folder_path.
    """
    def __init__(self, name=None, image_size=None, train_folder_path=None, **kwargs):
        super(DataSetBuilderImage, self).__init__(name, **kwargs)
        self.image_size = image_size
        self.train_folder_path = train_folder_path

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
        """
        :type folder_base: string path
        :param folder_base: path where live the images to convert

        extract the images from folder_base, where folder_base has the structure folder_base/label/
        """
        images = self.images_from_directories(folder_base)
        labels = np.ndarray(shape=(len(images),), dtype='|S1')
        data = []
        for image_index, (number_id, image_file) in enumerate(images):
            img = io.imread(image_file)
            data.append(img)
            labels[image_index] = number_id
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

    def build_dataset(self):
        """
        the data is extracted from the train_folder_path, and then saved.
        """
        data, labels = self.images_to_dataset(self.train_folder_path)
        super(DataSetBuilderImage, self).build_dataset(data, labels)

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
        return dataset

    def info(self):
        super(DataSetBuilderImage, self).info()
        print('Image Size {}x{}'.format(self.image_size, self.image_size))


class DataSetBuilderFile(DataSetBuilder):
    """
    Class for csv dataset build. Get the data from a csv's file.
    """

    def __init__(self, name=None, train_folder_path=None, **kwargs):
        super(DataSetBuilderFile, self).__init__(name, **kwargs)
        self.train_folder_path = train_folder_path

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
        super(DataSetBuilderFile, self).build_dataset(data, labels)

    @classmethod
    def merge_data_labels(self, data_path, labels_path, column_id):
        import pandas as pd
        data_df = pd.read_csv(data_path)
        labels_df = pd.read_csv(labels_path)
        return pd.merge(data_df, labels_df, on=column_id)


class DataSetBuilderFold(object):
    """
    Class for create datasets folds from datasets.
    
    :type n_splits: int
    :param n_plists: numbers of splits for apply to the dataset
    """
    def __init__(self, n_splits=2):
        self.name = uuid.uuid4().hex
        self.splits = []
        self.n_splits = n_splits
    
    def create_folds(self, dl):
        """
        :type dl: DataLabel
        :param dl: datalabel to split

        return an iterator of splited datalabel in n_splits DataSetBuilder datasets
        """
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_splits)
        for i, (train, test) in enumerate(skf.split(dl.data, dl.labels)):
            validation_index = int(train.shape[0] * .1)
            validation = train[:validation_index]
            train = train[validation_index:]
            dsb = DataSetBuilder(name=self.name+"_"+str(i), 
                dataset_path=settings["dataset_folds_path"],
                transforms=None,
                apply_transforms=False,
                dtype=dl.dtype,
                ltype=dl.ltype,
                description="",
                author="",
                compression_level=9,
                rewrite=True)
            data = dl.data[:]
            labels = dl.labels[:]
            dsb.build_dataset(data[train], labels[train], test_data=data[test], 
                test_labels=labels[test], validation_data=data[validation], 
                validation_labels=labels[validation])
            dsb.close_reader()
            yield dsb

    def build_dataset(self, dataset=None):
        """
        :type dataset: DataLabel
        :param dataset: dataset to fold

        construct the dataset fold from an DataSet class
        """
        dl = dataset.desfragment()
        for dsb in self.create_folds(dl):
            self.splits.append(dsb.name)
        dl.destroy()

    def get_splits(self):
        """
        return an iterator of datasets with the splits of original data
        """
        for split in self.splits:
            yield DataSetBuilder(name=split, dataset_path=settings["dataset_folds_path"])

    def destroy(self):
        for split in self.get_splits():
            split.destroy()
