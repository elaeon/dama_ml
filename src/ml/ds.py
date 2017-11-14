"""
Module for create datasets from distinct sources of data.
"""
from skimage import io

import os
import numpy as np
import pandas as pd
import dill as pickle
import random
import h5py
import logging
import datetime
import uuid

from ml.processing import Transforms
from ml.utils.config import get_settings

settings = get_settings("ml")

log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


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
    except Exception, e:
        print(e.message, path)


def calc_nshape(data, value):
    if value is None or not (0 < value <= 1) or data is None:
        value = 1
    limit = int(round(data.shape[0] * value, 0))
    return data[:limit]


class ReadWriteData(object):

    def auto_dtype(self, data, ttype):
        if ttype == "auto" and data is not None:
            return data.dtype
        elif ttype == "auto" and data is None:
            return "float64"
        elif ttype == "object":
            return h5py.special_dtype(vlen=unicode)
        else:
            return np.dtype(ttype)

    def _set_space_shape(self, f, name, shape, label=False):
        dtype = self.auto_dtype(None, self.dtype) if label == False else self.auto_dtype(None, self.ltype)
        f['data'].require_dataset(name, shape, dtype=dtype, chunks=True, 
            exact=True, **self.zip_params)

    def _set_space_data(self, f, name, data, label=False):
        dtype = self.auto_dtype(data, self.dtype) if label == False else self.auto_dtype(data, self.ltype)
        f['data'].require_dataset(name, data.shape, dtype=dtype, data=data, 
            exact=True, chunks=True, **self.zip_params)

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
            log.debug("Not found attribute {} in file {}".format(name, self.url()))
            return None
        except IOError:
            log.debug("Error opening {} in file {}".format(name, self.url()))
            return None

    def chunks_writer(self, f, name, data, chunks=258, init=0, type_t=None):
        from ml.utils.seq import grouper_chunk
        from tqdm import tqdm
        log.info("chunk size {}".format(chunks))
        end = init
        for row in tqdm(grouper_chunk(chunks, data)):
            seq = type_t(np.asarray(list(row)))
            end += seq.shape[0]
            f[name][init:end] = seq
            init = end
        return end

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
        log.debug("DESTROY {}".format(self.url()))

    def url(self):
        """
        return the path where is saved the dataset
        """
        return os.path.join(self.dataset_path, self.name)

    def close_reader(self):
        """
        close the hdf5 file. If is closed, no more data retrive will be perform.
        """
        if hasattr(self, 'f'):
            self.f.close()
            del self.f

    @classmethod
    def url_to_name(self, url):
        dataset_url = url.split("/")
        name = dataset_url[-1]
        path = "/".join(dataset_url[:-1])
        return name, path

    @classmethod
    def original_ds(self, name, dataset_path=None):
        from pydoc import locate
        meta_dataset = Data(name=name, dataset_path=dataset_path, rewrite=False)
        DS = locate(meta_dataset.dataset_class)
        return DS(name=name, dataset_path=dataset_path)
    

class Data(ReadWriteData):
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
    def __init__(self, name=None, dataset_path=None, transforms=None,
                apply_transforms=False, dtype='float64', description='',
                author='', compression_level=0, chunks=258, rewrite=False):

        self.name = uuid.uuid4().hex if name is None else name
        self.chunks = chunks
        self.rewrite = rewrite
        self.apply_transforms = apply_transforms
        self.header_map = ["author", "description", "timestamp", "transforms_str"]

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path
        
        if transforms is None:
            transforms = Transforms()

        #self._attrs = ["author", "dtype", "transforms"]
        ds_exist = self.exist()
        if not ds_exist or self.rewrite:
            if ds_exist:
                self.destroy()
            self.create_route()
            self.mode = "w"
            self.author = author
            self.dtype = dtype
            self.transforms = transforms
            self.description = description
            self.compression_level = compression_level
            self.timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M UTC")
            self.dataset_class = self.module_cls_name()
            self._applied_transforms = apply_transforms
            self.hash_header = self.calc_hash_H()
        else:
            self.mode = "r"

        self.zip_params = {"compression": "gzip", "compression_opts": self.compression_level}

    @property
    def author(self):
        return self._get_attr('author')

    @author.setter
    def author(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['author'] = value

    @property
    def dtype(self):
        return self._get_attr('dtype')

    @dtype.setter
    def dtype(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['dtype'] = value

    @property
    def transforms(self):
        return Transforms.from_json(self._get_attr('transforms'))

    @transforms.setter
    def transforms(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['transforms'] = value.to_json()

    @property
    def transforms_str(self):
        return self._get_attr('transforms')

    def reset_transforms(self, transforms):
        if self._applied_transforms is False:
            self.model = 'r'
            self._set_attr('transforms', transforms.to_json())
            self.mode = 'w'

    @property
    def description(self):
        return self._get_attr('description')

    @description.setter
    def description(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['description'] = value

    @property
    def timestamp(self):
        return self._get_attr('timestamp')

    @timestamp.setter
    def timestamp(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['timestamp'] = value

    @property
    def compression_level(self):
        return self._get_attr('compression_level')

    @compression_level.setter
    def compression_level(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['compression_level'] = value

    @property
    def dataset_class(self):
        return self._get_attr('dataset_class')

    @dataset_class.setter
    def dataset_class(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['dataset_class'] = value

    @property
    def _applied_transforms(self):
        return self._get_attr('applied_transforms')

    @_applied_transforms.setter
    def _applied_transforms(self, value):
        if self.mode == 'w':
            self._set_attr('applied_transforms', value)

    @property
    def hash_header(self):
        return self._get_attr('hash_H')

    @hash_header.setter
    def hash_header(self, value):
        if self.mode == 'w':
            self._set_attr('hash_H', value)

    @property
    def md5(self):
        return self._get_attr('md5')

    @md5.setter
    def md5(self, value):
        if self.mode == 'w':
            self._set_attr('md5', value)
                
    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @property
    def data(self):
        """
        eturn the data in the dataset
        """
        return self._get_data('data')

    def num_features(self):
        """
        return the number of features of the dataset
        """
        return self.data.shape[1]

    @property
    def shape(self):
        "return the shape of the dataset"
        return self.data.shape

    def desfragment(self, dataset_path=None):
        """
        Concatenate the train, valid and test data in a data array.
        Concatenate the train, valid, and test labels in another array.
        return DataLabel
        """
        log.debug("Desfragment...Data")
        return self.copy(dataset_path=dataset_path)

    def type_t(self, ttype, data):
        """
        :type ttype: string
        :param ttype: name of the type to convert the data. If ttype is 'auto' 
        the data is returned without be converted.

        :type data: array
        :param data: data to be converted

        convert the data to the especified ttype.
        """
        from ml.layers import IterLayer
        if ttype == 'auto':
            return data

        ttype = np.dtype(ttype)
        if isinstance(data, IterLayer):
            data = np.asarray(list(data))
            return data.astype(ttype)
        elif data.dtype != ttype and data.dtype != np.object:
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

    def exist(self):
        return self.md5 != None

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
        print('Applied transforms: {}'.format(self._applied_transforms))
        print('MD5: {}'.format(self.md5))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Dataset", "Shape", "dType"]
        table = []
        table.append(["dataset", self.data.shape, self.data.dtype])
        order_table_print(headers, table, "shape")

    def calc_md5(self):
        """
        calculate the md5 from the data.
        """
        import hashlib
        h = hashlib.md5(self.data[:])
        return h.hexdigest()

    def calc_hash_H(self):
        """
        hash digest for the header.
        """
        import hashlib
        header = [getattr(self, attr) for attr in self.header_map]
        h = hashlib.md5("".join(header))
        return h.hexdigest()

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
        return a value between [0, 1]. If is 0 no zeros exists, if is 1 all data is zero.
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

        with h5py.File(self.url(), 'a') as f:
            f.require_group("data")
            self._set_space_shape(f, "data", dsb.shape)
            end = self.chunks_writer(f, "/data/data", dsb.train_data[:], 
                chunks=self.chunks, type_t=self.dtype_t)
            end = self.chunks_writer(f, "/data/data", dsb.test_data[:], chunks=self.chunks, 
                                    init=end, type_t=self.dtype_t)
            self.chunks_writer(f, "/data/data", dsb.validation_data[:], chunks=self.chunks, 
                                init=end, type_t=self.dtype_t)
        
        self.md5 = self.calc_md5()

    def build_dataset_from_iter(self, iter_, shape, name, init=0, type_t=None):
        """
        Build a dataset from an iterator
        """
        if self.mode == "r":
            return

        with h5py.File(self.url(), 'a') as f:
            f.require_group("data")
            self._set_space_shape(f, name, shape)
            end = self.chunks_writer(f, "/data/{}".format(name), iter_, 
                chunks=self.chunks, init=init, type_t=type_t)
        return end

    def build_dataset(self, data):
        """
        build a datalabel dataset from data and labels
        """
        if self.mode == "r":
            return

        with h5py.File(self.url(), 'a') as f:
            f.require_group("data")
            data = self.processing(data, apply_transforms=self.apply_transforms)
            self._set_space_data(f, 'data', self.dtype_t(data))

        self.md5 = self.calc_md5()

    def empty(self, name, dataset_path=None, dtype='float64', ltype=None,
                apply_transforms=False, transforms=None):
        """
        build an empty DataLabel with the default parameters
        """
        data = Data(name=name, 
            dataset_path=dataset_path,
            transforms=self.transforms if transforms is None else transforms,
            apply_transforms=apply_transforms,
            dtype=dtype,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            chunks=self.chunks,
            rewrite=self.rewrite)
        data._applied_transforms = apply_transforms
        return data

    def convert(self, name, dtype='float64', apply_transforms=False, 
                percentaje=1, dataset_path=None, transforms=None):
        """
        :type dtype: string
        :param dtype: cast the data to the defined type

        dataset_path is not necesary to especify, this info is obtained from settings.cfg
        """
        data = self.empty(name, dataset_path=dataset_path, dtype=dtype, 
            ltype=ltype, apply_transforms=apply_transforms, transforms=transforms)
        data.build_dataset(calc_nshape(self.data, percentaje))
        data.close_reader()
        return data

    def copy(self, name=None, dataset_path=None, percentaje=1):
        """
        :type percentaje: float
        :param percentaje: value between [0, 1], this value represent the size of the dataset to copy.
        
        copy the dataset, a percentaje is permited for the size of the copy
        """
        name = self.name + "_copy_" + uuid.uuid4().hex if name is None else name
        data = self.convert(name, dtype=self.dtype,
                        apply_transforms=False, 
                        percentaje=percentaje, dataset_path=dataset_path)
        data._applied_transforms = self._applied_transforms
        return data

    def processing(self, data, apply_transforms=True):
        """
        :type data: array
        :param data: data to transform

        :type initial: bool
        :param initial: if multirow transforms are added, then this parameter
        indicates the initial data fit

        execute the transformations to the data.

        """
        if apply_transforms:
            #log.debug("Apply transforms " + str(data.shape))
            return self.transforms.apply(data)
        else:
            #log.debug("No transforms applied " + str(data.shape))
            return data if isinstance(data, np.ndarray) else np.asarray(data)

    @property
    def train_data(self):
        return self.data

    @property
    def oldtransforms2new(self):
        import json
        from collections import OrderedDict
        from pydoc import locate

        t = self._get_attr('transforms')
        transforms_list = json.loads(t, object_pairs_hook=OrderedDict)
        transforms = Transforms()
        for transforms_type in transforms_list:
            for type_, transforms_dict in transforms_type.items():              
                for fn, params in transforms_dict.items():
                    transforms.add(locate(fn), type=type_, **params)
        self._set_attr('transforms', transforms.to_json())

    @classmethod
    def to_DF(self, dataset):
        if len(dataset.shape) > 2:
            dataset = dataset.reshape(dataset.shape[0], -1)
        columns_name = map(lambda x: "c"+str(x), range(dataset.shape[-1])) + ["target"]
        return pd.DataFrame(data=dataset, columns=columns_name)

    def to_df(self):
        """
        convert the dataset to a dataframe
        """
        data = self.desfragment()
        df = self.to_DF(data.data[:])
        data.destroy()
        return df

    def outlayers(self, type_detector="isolation", n_estimators=25, max_samples=10, contamination=.2):
        """
        :type n_estimators: int
        :params n_estimators: number of estimators for IsolationForest

        :type max_samples: float
        :params max_samples: IsolationForest's max_samples

        :type contamination: float
        :params contamination: percentaje of expectect outlayers

        return the indexes of the data who are outlayers
        """
        if type_detector == "robust":
            from sklearn.covariance import MinCovDet
            robust_cov = MinCovDet().fit(self.data[:])
            robust_mahal = robust_cov.mahalanobis(self.data[:] - robust_cov.location_)
            limit = int(round(len(robust_mahal)*contamination))
            threshold = sorted(robust_mahal, reverse=True)[limit]
            y_pred = (1 if val < threshold else -1 for val in robust_mahal)
        else:
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(n_estimators=n_estimators,
                contamination=contamination,
                random_state=np.random.RandomState(42),
                max_samples=max_samples,
                n_jobs=-1)
            
            if len(self.data.shape) > 2:
                log.debug("outlayers transform shape...")
                data = self.data[:].reshape(-1, 1)
            else:
                data = self.data

            clf.fit(data)
            y_pred = clf.predict(data)
        return (i for i, v in enumerate(y_pred) if v == -1)

    def add_transforms(self, transforms, name=None):
        """
        :type name: string
        :param name: result dataset's name

        :type transforms: Transform
        :param transforms: transforms to apply in the new dataset
        """
        if self.apply_transforms == True:
            if hasattr(self, 'ltype'):
                dsb = self.convert(name, dtype=self.dtype, ltype=self.ltype, 
                    apply_transforms=True, percentaje=1, transforms=transforms)
            else:
                dsb = dsb_c.convert(name, dtype=self.dtype, apply_transforms=True, 
                    percentaje=1, transforms=transforms)
            dsb.transforms = self.transforms + transforms
        else:
            dsb = self.copy(name=name)
            dsb.transforms += transforms
        return dsb
        
    def remove_outlayers(self, outlayers):
        """
        removel the outlayers of the data
        """
        outlayers = list(self.outlayers(type_detector=type_detector))
        dl = self.desfragment()
        dl_ol = self.empty(self.name+"_n_outlayer", dtype=self.dtype, 
            apply_transforms=self.apply_transforms)
        shape = tuple([dl.shape[0] - len(outlayers)] + list(dl.shape[1:]))
        outlayers = iter(outlayers)
        outlayer = outlayers.next()
        data = np.empty(shape, dtype=self.dtype)
        counter = 0
        for index, row in enumerate(dl.data):
            if index == outlayer:
                try:
                    outlayer = outlayers.next()
                except StopIteration:
                    outlayer = None
            else:
                data[counter] = dl.data[index]
                counter += 1
        dl_ol.build_dataset(data)
        dl.destroy()
        return dl_ol

    def features2rows(self):
        """
        :type labels: bool

        transforms a matrix of dim (n, m) to a matrix of dim (n*m, 2) where
        the rows are described as [feature_column, feature_data]
        """
        data = np.empty((self.data.shape[0] * self.data.shape[1], 3))
        base = 0
        for index_column in range(1, self.data.shape[1] + 1):
            next = self.data.shape[0] + base
            data[base:next] = np.append(
                np.zeros((self.data.shape[0], 1)) + (index_column - 1), 
                self.data[:, index_column-1:index_column], 
                axis=1)
            base = next
        return data


class DataLabel(Data):
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

    :type ltype: string
    :param ltype: the type of the labels to save

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
                apply_transforms=False,
                dtype='float64',
                ltype='|S1',
                description='',
                author='',
                compression_level=0,
                chunks=258,
                rewrite=False):

        super(DataLabel, self).__init__(name=name, dataset_path=dataset_path,
            apply_transforms=apply_transforms,transforms=transforms, dtype=dtype,
            description=description, author=author, compression_level=compression_level,
            chunks=chunks, rewrite=rewrite)
        
        #self._attrs.append("ltype")
        if self.mode == "w" or self.rewrite:
            self.ltype = ltype

    @property
    def ltype(self):
        return self._get_attr('ltype')

    @ltype.setter
    def ltype(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['ltype'] = value

    @property
    def labels(self):
        """
        return the labels in the dataset
        """
        return self._get_data('labels')

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
            dataset, n_labels = self.only_labels_from_data(dl, labels)
            dl.destroy()
        except ValueError:
            label = labels[0] if len(labels) > 0 else None
            log.warning("label {} is not found in the labels set".format(label))
            return np.asarray([]), np.asarray([])
        return np.asarray(dataset), np.asarray(n_labels)

    def is_binary(self):
        return len(self.labels_info().keys()) == 2

    @classmethod
    def only_labels_from_data(self, ds, labels):
        """
        :type ds: dataset
        :param ds: dataset to select the data

        :type labels: list
        :type labels: list of labels to filter data

        return a tuple (data, labels) with the data filtered.
        """
        s_labels = set(labels)
        return zip(*filter(lambda x: x[1] in s_labels, zip(ds.data, ds.labels)))

    def ltype_t(self, labels):
        """
        :type labels: narray
        :param labels: narray to cast

        cast the labels to the predefined dataset ltype
        """
        return self.type_t(self.ltype, labels)

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
        print('Applied transforms: {}'.format(self._applied_transforms))
        print('MD5: {}'.format(self.md5))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Dataset", "Shape", "dType", "Labels"]
        table = []
        table.append(["dataset", self.data.shape, self.data.dtype, self.labels.size])
        order_table_print(headers, table, "shape")

    def build_dataset_from_dsb(self, dsb):
        """
        Transform a dataset with train, test and validation dataset into a datalabel dataset
        """
        if self.mode == "r":
            return

        labels_shape = tuple(dsb.shape[0:1] + dsb.train_labels.shape[1:])
        with h5py.File(self.url(), 'a') as f:
            f.require_group("data")
            self._set_space_shape(f, "data", dsb.shape)
            self._set_space_shape(f, "labels", labels_shape, label=True)
            end = self.chunks_writer(f, "/data/data", dsb.train_data[:], chunks=self.chunks,
                type_t=self.dtype_t)
            end = self.chunks_writer(f, "/data/data", dsb.test_data[:], chunks=self.chunks, 
                                    init=end, type_t=self.dtype_t)
            self.chunks_writer(f, "/data/data", dsb.validation_data[:], chunks=self.chunks, 
                                init=end, type_t=self.dtype_t)

            end = self.chunks_writer(f, "/data/labels", dsb.train_labels[:], 
                chunks=self.chunks, type_t=self.ltype_t)
            end = self.chunks_writer(f, "/data/labels", dsb.test_labels[:], chunks=self.chunks, 
                                    init=end, type_t=self.ltype_t)
            self.chunks_writer(f, "/data/labels", dsb.validation_labels[:], chunks=self.chunks, 
                                init=end, type_t=self.ltype_t)
       
        self.md5 = self.calc_md5()

    def build_dataset(self, data, labels):
        """
        build a datalabel dataset from data and labels
        """
        with h5py.File(self.url(), 'a') as f:
            f.require_group("data")
            data = self.processing(data, apply_transforms=self.apply_transforms)
            self._set_space_data(f, 'data', self.dtype_t(data))
            self._set_space_data(f, 'labels', self.ltype_t(labels), label=True)

        self.md5 = self.calc_md5()

    def empty(self, name, dtype='float64', ltype='|S1', 
                apply_transforms=False, dataset_path=None,
                transforms=None):
        """
        build an empty DataLabel with the default parameters
        """
        dl = DataLabel(name=name, 
            dataset_path=dataset_path,
            transforms=self.transforms if transforms is None else transforms,
            apply_transforms=apply_transforms,
            dtype=dtype,
            ltype=ltype,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            chunks=self.chunks,
            rewrite=self.rewrite)
        dl._applied_transforms = apply_transforms
        return dl

    def convert(self, name, dtype='float64', ltype='|S1', apply_transforms=False, 
                percentaje=1, dataset_path=None, transforms=None):
        """
        :type dtype: string
        :param dtype: cast the data to the defined type

        dataset_path is not necesary to especify, this info is obtained from settings.cfg
        """
        dl = self.empty(name, dtype=dtype, ltype=ltype, 
                        apply_transforms=apply_transforms, 
                        dataset_path=dataset_path, transforms=transforms)
        dl.build_dataset(calc_nshape(self.data, percentaje), calc_nshape(self.labels, percentaje))
        dl.close_reader()
        return dl

    def copy(self, name=None, dataset_path=None, percentaje=1):
        """
        :type percentaje: float
        :param percentaje: value between [0, 1], this value represent the size of the dataset to copy.
        
        copy the dataset, a percentaje is permited for the size of the copy
        """
        name = self.name + "_copy_" + uuid.uuid4().hex if name is None else name
        dl = self.convert(name, dtype=self.dtype, ltype=self.ltype, 
                        apply_transforms=False, 
                        percentaje=percentaje, dataset_path=dataset_path)
        dl._applied_transforms = self._applied_transforms
        return dl

    @classmethod
    def to_DF(self, dataset, labels):
        if len(dataset.shape) > 2:
            dataset = dataset.reshape(dataset.shape[0], -1)
        columns_name = map(lambda x: "c"+str(x), range(dataset.shape[-1])) + ["target"]
        return pd.DataFrame(data=np.column_stack((dataset, labels)), columns=columns_name)

    def to_df(self, labels2numbers=False):
        """
        convert the dataset to a dataframe
        """
        dl = self.desfragment()
        if labels2numbers == False:
            df = self.to_DF(dl.data[:], dl.labels[:])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(dl.labels[:])
            df = self.to_DF(dl.data[:], le.transform(dl.labels[:]))

        dl.destroy()
        return df

    @classmethod
    def from_DF(self, name, df, transforms=None, apply_transforms=None, path=None):
        pass
        
    def remove_outlayers(self, outlayers):
        """
        removel the outlayers of the data
        """
        dl = self.desfragment()
        dl_ol = self.empty(self.name+"_n_outlayer", dtype=self.dtype, ltype=self.ltype, 
            apply_transforms=self.apply_transforms)
        shape = tuple([dl.shape[0] - len(outlayers)] + list(dl.shape[1:]))
        outlayers = iter(outlayers)
        outlayer = outlayers.next()
        data = np.empty(shape, dtype=self.dtype)
        labels = np.empty((shape[0],), dtype=self.ltype)
        counter = 0
        for index, row in enumerate(dl.data):
            if index == outlayer:
                try:
                    outlayer = outlayers.next()
                except StopIteration:
                    outlayer = None
            else:
                data[counter] = dl.data[index]
                labels[counter] = dl.labels[index]
                counter += 1
        dl_ol.build_dataset(data, labels)
        dl.destroy()
        return dl_ol

    def features2rows(self, labels=False):
        """
        :type labels: bool
        :param labels: if true, labels are included

        transforms a matrix of dim (n, m) to a matrix of dim (n*m, 2) or (n*m, 3) where
        the rows are described as [feature_column, feature_data]
        """
        if labels == False:
            data = super(DataLabel, self).features2rows()
        else:
            data = np.empty((self.data.shape[0] * self.data.shape[1], 3))
            labels = self.labels[:].reshape(1, -1)
            base = 0
            for index_column in range(1, self.data.shape[1] + 1):
                tmp_data = np.append(
                    np.zeros((self.data.shape[0], 1)) + (index_column - 1), 
                    self.data[:, index_column-1:index_column], 
                    axis=1)
                next = self.data.shape[0] + base
                data[base:next] = np.append(tmp_data, labels.T, axis=1)
                base = next
        return data

    def to_data(self):
        dl = self.desfragment()
        name = self.name + "_data_" + uuid.uuid4().hex
        data = super(DataLabel, self).empty(name, dtype=self.dtype, apply_transforms=self.apply_transforms)
        data.build_dataset(dl.data)
        data.close_reader()
        dl.destroy()
        return data

    def plot(self, view="columns", type_g=None, columns=None):        
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if view == "columns":
            if type_g == 'box':
                sns.set(style="whitegrid", palette="pastel", color_codes=True)
                data = self.features2rows(labels=True)
                sns.boxplot(x=data[:,0], y=data[:,1], hue=data[:,2], palette="PRGn")
                sns.despine(offset=10, trim=True)
            elif type_g == "violin":
                sns.set(style="whitegrid", palette="pastel", color_codes=True)
                data = self.features2rows(labels=True)
                sns.violinplot(x=data[:,0], y=data[:,1], hue=data[:,2], palette="PRGn", inner="box")
                sns.despine(offset=10, trim=True)
            elif type_g == "hist" and self.num_features() <= 64:
                size = int(round(self.num_features() ** .5))
                f, axarr = plt.subplots(size, size, sharey=True, sharex=True)
                base = 0
                for i in range(size):
                    for j in range(size):
                        axarr[i, j].set_title('Feature {}'.format(base+1))
                        sns.distplot(self.data[:, base:base+1], bins=50, 
                            kde=False, rug=False, color="b", ax=axarr[i, j])
                        base += 1
            elif type_g == "hist_label" and self.is_binary() and self.num_features() <= 64:
                labels_info = self.labels_info()
                label_0 = labels_info.keys()[0]
                label_1 = labels_info.keys()[1]
                ds0, _ = self.only_labels_from_data(self, [label_0])
                ds1, _ = self.only_labels_from_data(self, [label_1])
                ds0 = np.asarray(ds0)
                ds1 = np.asarray(ds1)
                size = int(round(self.num_features() ** .5))
                f, axarr = plt.subplots(size, size, sharey=True, sharex=True)
                base = 0
                for i in range(size):
                    for j in range(size):
                        axarr[i, j].set_title('Feature {}'.format(base+1))
                        sns.distplot(ds0[:, base:base+1], bins=50, 
                            kde=False, rug=False, color="b", ax=axarr[i, j])
                        sns.distplot(ds1[:, base:base+1], bins=50, 
                            kde=False, rug=False, color="r", ax=axarr[i, j])
                        base += 1
            elif type_g == "corr":
                df = self.to_df()
                df = df.iloc[:, 0:self.num_features()].astype(np.float64) 
                corr = df.corr()
                mask = np.zeros_like(corr, dtype=np.bool)
                mask[np.triu_indices_from(mask)] = True
                cmap = sns.diverging_palette(220, 10, as_cmap=True)
                sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                    square=True, xticklabels=5, yticklabels=5,
                    linewidths=.5, cbar_kws={"shrink": .5})
        else:
            if columns is not None:
                columns = columns.split(",")
                if type_g == "lm":
                    df = self.to_df(labels2numbers=True)
                    sns.lmplot(x=columns[0], y=columns[1], data=df, col="target", hue="target")
            else:
                if self.shape[1] > 2:
                    from ml.ae.extended.w_keras import PTsne
                    dl = DataLabel(name=self.name+"_2d_", 
                            dataset_path=self.dataset_path,
                            transforms=None,
                            apply_transforms=False,
                            dtype=self.dtype,
                            ltype=self.ltype,
                            compression_level=9,
                            rewrite=False)

                    if not dl.exist():
                        ds = self.to_data()
                        classif = PTsne(model_name="tsne", model_version="1", 
                            check_point_path="/tmp/", dataset=ds, latent_dim=2)
                        classif.train(batch_size=50, num_steps=100)
                        data = np.asarray(list(classif.predict(self.data)))
                        dl.build_dataset(data, self.labels[:])
                        ds.destroy()
                else:
                    dl = self

                if dl.shape[1] == 2:
                    if type_g == "lm":
                        df = dl.to_df(labels2numbers=True)
                        sns.lmplot(x="c0", y="c1", data=df, hue="target")
                    elif type_g == "scatter":
                        df = dl.to_df()
                        legends = []
                        labels = self.labels_info()
                        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
                        for color, label in zip(colors, labels):
                            df_tmp = df[df["target"] == label]
                            legends.append((plt.scatter(df_tmp["c0"].astype("float64"), 
                                df_tmp["c1"].astype("float64"), color=color), label))
                        p, l = zip(*legends)
                        plt.legend(p, l, loc='lower left', ncol=3, fontsize=8, 
                            scatterpoints=1, bbox_to_anchor=(0,0))
                    elif type_g == "pairplot":
                        df = dl.to_df(labels2numbers=True)
                        sns.pairplot(df.astype("float64"), hue='target', 
                            vars=df.columns[:-1], diag_kind="kde", palette="husl")
        plt.show()

    def labels2num(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(self.labels)
        return le

    def to_libsvm(self, name=None, save_to=None):
        """
        tranforms the dataset into libsvm format
        """
        from ml.utils.seq import libsvm_row
        le = self.labels2num()
        name = self.name+".txt" if name is None else name
        file_path = os.path.join(save_to, name)
        with open(file_path, 'w') as f:
            for row in libsvm_row(self.labels, self.data, le):
                f.write(" ".join(row))
                f.write("\n")

    def stadistics(self):
        from tabulate import tabulate
        from collections import defaultdict
        from ml.utils.numeric_functions import is_binary, is_integer

        headers = ["feature", "label", "missing", "mean", "std dev", "zeros", 
            "min", "25%", "50%", "75%", "max", "type"]
        table = []
        li = self.labels_info()
        feature_column = defaultdict(dict)
        for label in li:
            mask = (self.labels == label)
            data = self.data[mask]
            for i, column in enumerate(data.T):
                percentile = np.nanpercentile(column, [0, 25, 50, 75, 100])
                values = [
                    "{0:.{1}f}%".format((np.count_nonzero(np.isnan(column)) / float(column.size)) * 100, 2),
                    np.mean(column),  
                    np.std(column),
                    "{0:.{1}f}%".format((
                    (column.size - np.count_nonzero(column)) / float(column.size)) * 100, 2),
                    ]
                values.extend(percentile)
                if is_binary(column, include_null=True):
                    values.append("binary")
                elif is_integer(column):
                    values.append("integer")
                else:
                    values.append("continuos")
                feature_column[i][label] = values

        for feature, rows in feature_column.items():
            for label, row in rows.items():
                table.append([feature, label] + row)
                feature = ""
        return tabulate(table, headers)


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
                apply_transforms=False,
                transforms=None,
                train_size=.7,
                valid_size=.1,
                validator='cross',
                dtype='float64',
                ltype='|S1',
                description='',
                author='',
                compression_level=0,
                chunks=258,
                rewrite=False):

        super(DataSetBuilder, self).__init__(name=name, dataset_path=dataset_path,
            apply_transforms=apply_transforms,transforms=transforms, dtype=dtype,
            ltype=ltype, description=description, author=author, 
            compression_level=compression_level, chunks=chunks, rewrite=rewrite)

        if self.mode == "w" or self.rewrite:
            self.validator = validator if validator is not None else ''
            self.valid_size = valid_size
            self.train_size = train_size

        if self.train_size is not None or self.valid_size is not None:
            self.test_size = round(1 - (self.train_size + self.valid_size), 2)
        else:
            self.test_size = 0

    @property
    def valid_size(self):
        return self._get_attr('valid_size')

    @valid_size.setter
    def valid_size(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['valid_size'] = value

    @property
    def train_size(self):
        return self._get_attr('train_size')

    @train_size.setter
    def train_size(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['train_size'] = value

    @property
    def validator(self):
        return self._get_attr('validator')

    @validator.setter
    def validator(self, value):
        if self.mode == 'w':
            with h5py.File(self.url(), 'a') as f:
                f.attrs['validator'] = value

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
        return np.concatenate((self.train_data[:], 
            self.test_data[:],
            self.validation_data[:]), axis=0)

    @property
    def labels(self):
        return np.concatenate((self.train_labels[:], 
            self.test_labels[:],
            self.validation_labels[:]), axis=0)

    @property
    def data_validation(self):
        return np.concatenate((self.train_data[:], self.validation_data[:]), 
                            axis=0)

    @property
    def data_validation_labels(self):
        return np.concatenate((self.train_labels[:], self.validation_labels[:]), 
                            axis=0)

    @property
    def shape(self):
        "return the shape of the dataset"
        rows = self.train_data.shape[0] + self.test_data.shape[0] +\
            self.validation_data.shape[0]
        if self.train_data.dtype != np.object or len(self.train_data.shape) > 1:
            return tuple([rows] + list(self.train_data.shape[1:]))
        else:
            return (rows,)

    def desfragment(self, name=None, dataset_path=None):
        """
        Concatenate the train, valid and test data in a data array.
        Concatenate the train, valid, and test labels in another array.
        return data, labels
        """
        log.debug("Desfragment...DSB")
        dl = DataLabel(
            name=uuid.uuid4().hex if name is None else name,
            dataset_path=dataset_path,
            transforms=self.transforms,
            apply_transforms=self.apply_transforms,
            dtype=self.dtype,
            ltype=self.ltype,
            description=self.description,
            author=self.author,
            chunks=1000,
            compression_level=self.compression_level)
        dl.build_dataset_from_dsb(self)
        return dl

    def to_datalabel(self, name=None):
        return self.desfragment(name=name)

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
        print('Applied transforms: {}'.format(self._applied_transforms))
        print('MD5: {}'.format(self.md5))
        print('Description: {}'.format(self.description))
        print('       ')
        try:
            headers = ["Dataset", "Shape", "dType", "Labels"]
            table = []
            table.append(["train set", self.train_data.shape, 
                        self.train_data.dtype, self.train_labels.size])

            if self.validation_data is not None:
                table.append(["valid set", self.validation_data.shape, 
                            self.validation_data.dtype, self.validation_labels.size])

            table.append(["test set", self.test_data.shape, 
                            self.test_data.dtype, self.test_labels.size])
            order_table_print(headers, table, "shape")
            if classes == True:
                headers = ["class", "# items", "%"]
                items = [(cls, total, (total/float(self.shape[0]))*100) 
                        for cls, total in self.labels_info().items()]
                items_p = [0, 0]
                order_table_print(headers, items, "# items")
        except KeyError:
            print("No data found")

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

        #fixme 
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        with h5py.File(self.url(), 'a') as f:
            f.require_group("data")
            if test_data is not None and test_labels is not None\
                and validation_data is not None\
                and validation_labels is not None:
                    data_labels = [
                        data, validation_data, test_data,
                        labels, validation_labels, test_labels]
            else:
                if self.validator == 'cross':
                    data_labels = self.cross_validators(data, labels)
                elif self.validator == '':
                    index = data.shape[0] / 3
                    data_labels = [data[0:index], data[index:2*index], data[index*2:index*3],
                    labels[0:index], labels[index:index*2], labels[index*2:index*3]]
                else:
                    data_labels = self.cross_validators(data, labels)

            train_data = self.processing(data_labels[0], 
                                        apply_transforms=self.apply_transforms)
            self._set_space_data(f, 'train_data', self.dtype_t(train_data))

            test_data = self.processing(data_labels[2], 
                                        apply_transforms=self.apply_transforms)
            self._set_space_data(f, 'test_data', self.dtype_t(test_data))
            
            validation_data = self.processing(data_labels[1],
                                             apply_transforms=self.apply_transforms)
            self._set_space_data(f, 'validation_data', self.dtype_t(validation_data))

            self._set_space_data(f, 'train_labels', self.ltype_t(data_labels[3]), label=True)
            self._set_space_data(f, 'test_labels', self.ltype_t(data_labels[5]), label=True)
            self._set_space_data(f, 'validation_labels', self.ltype_t(data_labels[4]), label=True)
        
        self.md5 = self.calc_md5()

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

    def convert(self, name, dtype='float64', ltype='|S1', apply_transforms=False, 
                percentaje=1, dataset_path=None, transforms=None):
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
        dsb = self.empty(name, dtype=dtype, ltype=ltype, 
            apply_transforms=apply_transforms, dataset_path=dataset_path,
            transforms=transforms)
        dsb.build_dataset(
            calc_nshape(self.train_data, percentaje), 
            calc_nshape(self.train_labels, percentaje),
            test_data=calc_nshape(self.test_data, percentaje),
            test_labels=calc_nshape(self.test_labels, percentaje),
            validation_data=calc_nshape(self.validation_data, percentaje),
            validation_labels=calc_nshape(self.validation_labels, percentaje))
        dsb.close_reader()
        return dsb

    def empty(self, name, dtype='float64', ltype='|S1', apply_transforms=False,
                dataset_path=None, transforms=None):
        """
        build an empty DataLabel with the default parameters
        """
        dsb = DataSetBuilder(name=name, 
            dataset_path=dataset_path,
            transforms=self.transforms if transforms is None else transforms,
            apply_transforms=apply_transforms,
            train_size=self.train_size,
            valid_size=self.valid_size,
            validator=self.validator,
            dtype=dtype,
            ltype=ltype,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            chunks=self.chunks,
            rewrite=self.rewrite)
        dsb._applied_transforms = apply_transforms
        return dsb

    def to_libsvm(self, name=None, save_to=None, validation=True):
        """
        tranforms the dataset to libsvm format
        """
        from ml.utils.seq import libsvm_row
        le = self.labels2num()
        f_names = ["train", "test"]
        if validation == True:
            f_names.append("validation")
        
        for f_name in f_names:
            n_name = self.name if name is None else name
            n_name = "{}.{}.txt".format(n_name, f_name)
            file_path = os.path.join(save_to, n_name)
            with open(file_path, 'w') as f:
                labels = getattr(self, f_name+"_labels")
                data = getattr(self, f_name+"_data")
                for row in libsvm_row(labels, data, le):
                    f.write(" ".join(row))
                    f.write("\n")


class DataSetBuilderImage(DataSetBuilder):
    """
    Class for images dataset build. Get the data from a directory where each directory's name is the label.
    
    :type image_size: int
    :param image_size: define the image size to save in the dataset

    kwargs are the same that DataSetBuilder's options

    :type data_folder_path: string
    :param data_folder_path: path to the data what you want to add to the dataset, split the data in train, test and validation. If you want manualy split the data in train and test, check test_folder_path.
    """
    def __init__(self, name=None, image_size=None, training_data_path=None, **kwargs):
        super(DataSetBuilderImage, self).__init__(name, **kwargs)
        self.image_size = image_size
        self.training_data_path = training_data_path

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
        the data is extracted from the training_datar_path, and then saved.
        """
        data, labels = self.images_to_dataset(self.training_data_path)
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

    def copy(self, name=None, dataset_path=None, percentaje=1):
        dataset = super(DataSetBuilderImage, self).copy(
            name=name, percentaje=percentaje, dataset_path=dataset_path)
        dataset.image_size = self.image_size
        return dataset


class DataSetBuilderFile(DataSetBuilder):
    """
    Class for csv dataset build. Get the data from a csv's file.
    
    :type training_data_path: list
    :param training_data_path: list of files paths

    :type sep: list
    :param sep: list of strings separators for each file

    kwargs are the same that DataSetBuilder's options
    """

    def __init__(self, name=None, training_data_path=None, test_data_path=None,
                sep=None, merge_field="id", na_values=None, **kwargs):
        super(DataSetBuilderFile, self).__init__(name, **kwargs)
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.sep = sep
        self.merge_field = merge_field
        self.na_values = na_values

    def from_csv(self, folder_path, label_column, nrows=None, exclude_columns=None):
        """
        :type folder_path: string
        :param folder_path: path to the csv.

        :type label_column: string
        :param label_column: column's name where are the labels
        """
        df = pd.read_csv(folder_path, nrows=nrows, na_values=self.na_values)
        data, labels = self.df2dataset_label(df, label_column, 
                                            exclude_columns=exclude_columns)
        return data, labels

    @classmethod
    def df2dataset_label(self, df, labels_column, ids=None, exclude_columns=None):
        if ids is not None:
            drops = ids + [labels_column]
        else:
            drops = [labels_column]

        if exclude_columns is not None:
            if isinstance(exclude_columns, list):
                drops.extend(exclude_columns)
            else:
                drops.extend([exclude_columns])

        dataset = df.drop(drops, axis=1).as_matrix()
        labels = df[labels_column].as_matrix()
        return dataset, labels        

    def build_dataset(self, labels_column=None, nrows=None, exclude_columns=None):
        """
         :type label_column: string
         :param label_column: column's name where are the labels
        """
        if isinstance(self.training_data_path, list):
            if not isinstance(self.sep, list):
                sep = [self.sep for _ in self.training_data_path]
            else:
                sep = self.sep
            old_df = None
            for sep, path in zip(sep, self.training_data_path):
                data_df = pd.read_csv(path, sep=sep, nrows=nrows)
                if old_df is not None:
                    old_df = pd.merge(old_df, data_df, on=self.merge_field)
                else:
                    old_df = data_df
            data, labels = self.df2dataset_label(old_df, 
                labels_column=labels_column, ids=[self.merge_field],
                exclude_columns=exclude_columns)
        else:
            data, labels = self.from_csv(self.training_data_path, labels_column, 
                                        nrows=nrows, 
                                        exclude_columns=exclude_columns)
        super(DataSetBuilderFile, self).build_dataset(data, labels)


class DataSetBuilderFold(object):
    """
    Class for create datasets folds from datasets.
    
    :type n_splits: int
    :param n_plists: numbers of splits for apply to the dataset
    """
    def __init__(self, n_splits=2, dataset_path=None):
        self.name = uuid.uuid4().hex
        self.splits = []
        self.n_splits = n_splits
        self.dataset_path = settings["dataset_folds_path"] if dataset_path is None else dataset_path
    
    def create_folds(self, dl):
        """
        :type dl: DataLabel
        :param dl: datalabel to split

        return an iterator of splited datalabel in n_splits DataSetBuilder datasets
        """
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_splits)
        for i, (train, test) in enumerate(skf.split(dl.data, dl.labels)):
            validation_index = int(round(train.shape[0] * .1, 0))
            validation = train[:validation_index]
            train = train[validation_index:]
            dsb = DataSetBuilder(name=self.name+"_"+str(i), 
                dataset_path=self.dataset_path,
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
            yield DataSetBuilder(name=split, dataset_path=self.dataset_path)

    def destroy(self):
        for split in self.get_splits():
            split.destroy()
