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
from ml.layers import IterLayer
from ml import fmtypes as Fmtypes

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
    except IOError as e:
        log.info(e)
        return {}
    except Exception, e:
        print(e.message, path)


def calc_nshape(data, value):
    if value is None or not (0 < value <= 1) or data is None:
        value = 1
    limit = int(round(data.shape[0] * value, 0))
    return data[:limit]


def cache(func):
    def fn_wrapper(self):
        attr = "{}_cache".format(func.__name__)
        if not hasattr(self, attr) or getattr(self, attr) is None:
            setattr(self, attr, func(self))
        return getattr(self, attr)
    return fn_wrapper


def clean_cache(func):
    def fn_wrapper(self, value):
        attr = "{}_cache".format(func.__name__)
        if hasattr(self, attr) and getattr(self, attr) is not None:
            setattr(self, attr, None)
        return func(self, value)
    return fn_wrapper


class ReadWriteData(object):

    def __enter__(self):
        self.f = h5py.File(self.url(), "a")
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def auto_dtype(self, ttype):
        if ttype == np.dtype("O"):
            return h5py.special_dtype(vlen=unicode)
        else:
            return ttype

    def _set_space_shape(self, name, shape, dtype):
        self.f.require_group("data")
        dtype = self.auto_dtype(dtype)
        self.f['data'].require_dataset(name, shape, dtype=dtype, chunks=True, 
            exact=True, **self.zip_params)

    def _get_data(self, name):
        key = '/data/' + name
        return self.f[key]

    def _set_space_fmtypes(self, num_features):
        self.f.require_group("pp")
        self.f['pp'].require_dataset("fmtypes", (num_features,), dtype=np.dtype(int), 
            exact=True, chunks=True, **self.zip_params)        

    def _get_pp(self, name):
        """ get the preprocesed meta data"""
        key = '/pp/' + name
        return self.f.get(key, None)

    def _set_attr(self, name, value):
        if self.f is None:
            with h5py.File(self.url(), 'a') as f:
                f.attrs[name] = value
        else:
            self.f.attrs[name] = value
            
    def _get_attr(self, name):
        try:
            if self.f is None:
                with h5py.File(self.url(), 'r') as f:
                    return f.attrs[name]
            else:
                return self.f.attrs[name]
        except KeyError:
            log.debug("Not found attribute {} in file {}".format(name, self.url()))
            return None
        except IOError as e:
            log.debug("Error opening {} in file {}".format(name, self.url()))
            return None

    def chunks_writer(self, name, data, init=0):
        from tqdm import tqdm
        log.info("Writing with chunks size {}".format(data.chunks_size))
        end = init
        if data.chunks_size != 0 and data.chunks_size < data.shape[0]:
            total_data = int(round(data.shape[0] / float(data.chunks_size), 0))
        elif data.chunks_size > data.shape[0]:
            total_data = 1
        else:
            total_data = int(data.shape[0])
        for smx in tqdm(data, total=total_data):
            if hasattr(smx, 'shape') and len(smx.shape) >= 1 and data.has_chunks:
                end += smx.shape[0]
            else:
                end += 1

            if isinstance(smx, pd.DataFrame):
                array = smx.values
            elif not hasattr(smx, '__iter__'):
                array = (smx,)
            else:
                array = smx

            try:
                self.f[name][init:end] = array
                init = end
            except TypeError as e:
                if self.dtype == "|O":
                    type_ = "string"
                else:
                    type_ = self.dtype
                raise TypeError("All elements in array must be of type '{}' but found '{}'".format(
                    type_, self.dtype))
        return end

    def chunks_writer_split(self, data_key, labels_key, data, labels_column):
        from tqdm import tqdm
        log.info("Writing with chunks size {}".format(data.chunks_size))
        if data.chunks_size != 0 and data.chunks_size < data.shape[0]:
            total_data = int(round(data.shape[0] / float(data.chunks_size), 0))
        elif data.chunks_size > data.shape[0]:
            total_data = 1
        else:
            total_data = int(data.shape[0])
        init = 0
        end = init
        for smx in tqdm(data, total=total_data):
            if hasattr(smx, 'shape') and len(smx.shape) >= 1 and data.has_chunks:
                end += smx.shape[0]
            else:
                end += 1

            if isinstance(smx, pd.DataFrame):
                #array_data = smx[smx.columns.difference([labels_column])].values
                array_data = smx.drop([labels_column], axis=1).values
                array_labels = smx[labels_column].values
            else:
                array_data = np.delete(smx, labels_column)
                array_labels = smx[:, labels_column]

            try:
                self.f[data_key][init:end] = array_data
                self.f[labels_key][init:end] = array_labels
                init = end
            except TypeError as e:
                if self.dtype == "|O":
                    type_ = "string"
                else:
                    type_ = self.dtype
                raise TypeError("All elements in array must be of type '{}' but found '{}'".format(
                    type_, self.dtype))

    def create_route(self):
        """
        create directories if the dataset_path does not exist
        """
        if os.path.exists(self.dataset_path) is False:
            os.makedirs(self.dataset_path)

    def destroy(self):
        """
        delete the correspondly hdf5 file
        """
        from ml.utils.files import rm
        rm(self.url())
        log.debug("DESTROY {}".format(self.url()))

    def url(self):
        """
        return the path where is saved the dataset
        """
        return os.path.join(self.dataset_path, self.name)

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
        return DS(name=name, dataset_path=dataset_path, rewrite=False)
    

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
                apply_transforms=False, description='', author='', 
                compression_level=0, rewrite=False):

        self.name = uuid.uuid4().hex if name is None else name
        self.apply_transforms = apply_transforms
        self.header_map = ["author", "description", "timestamp", "transforms_str"]
        self.f = None

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path
        
        if transforms is None:
            transforms = Transforms()

        ds_exist = self.exist()
        if not ds_exist or rewrite:
            if ds_exist:
                self.destroy()
            self.create_route()
            self.author = author
            self.transforms = transforms
            self.description = description
            self.compression_level = compression_level
            self.timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M UTC")
            self.dataset_class = self.module_cls_name()
            self._applied_transforms = apply_transforms
            self.hash_header = self.calc_hash_H()

        self.zip_params = {"compression": "gzip", "compression_opts": self.compression_level}

    @property
    def author(self):
        return self._get_attr('author')

    @author.setter
    def author(self, value):
        self._set_attr('author', value)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    @cache
    def transforms(self):
        return Transforms.from_json(self._get_attr('transforms'))

    @transforms.setter
    @clean_cache
    def transforms(self, value):
        self._set_attr('transforms', value.to_json())

    @property
    def transforms_str(self):
        return self._get_attr('transforms')

    @property
    def description(self):
        return self._get_attr('description')

    @description.setter
    def description(self, value):
        self._set_attr('description', value)

    @property
    def timestamp(self):
        return self._get_attr('timestamp')

    @timestamp.setter
    def timestamp(self, value):
        self._set_attr('timestamp', value)

    @property
    def compression_level(self):
        return self._get_attr('compression_level')

    @compression_level.setter
    def compression_level(self, value):
        self._set_attr('compression_level', value)

    @property
    def dataset_class(self):
        return self._get_attr('dataset_class')

    @dataset_class.setter
    def dataset_class(self, value):
        self._set_attr('dataset_class', value)

    @property
    def _applied_transforms(self):
        return bool(self._get_attr('applied_transforms'))

    @_applied_transforms.setter
    def _applied_transforms(self, value):
        self._set_attr('applied_transforms', value)

    @property
    def hash_header(self):
        return self._get_attr('hash_H')

    @hash_header.setter
    def hash_header(self, value):
        self._set_attr('hash_H', value)

    @property
    def md5(self):
        return self._get_attr('md5')

    @md5.setter
    def md5(self, value):
        self._set_attr('md5', value)
                
    @property
    def fmtypes(self):
        return self._get_pp("fmtypes")

    @fmtypes.setter
    def fmtypes(self, value):
        data = self._get_pp('fmtypes')
        data[...] = value

    def set_fmtypes(self, column, fmtype):
        ndata = self.fmtypes[:]
        ndata[column] = fmtype.id
        self.fmtypes = ndata

    def build_fmtypes(self, fmtypes=None):
        from ml.utils.numeric_functions import unique_size, data_type
        if fmtypes is None:
            fmtypes = np.empty((self.num_features()), dtype=np.dtype(int))
            for ci in range(self.num_features()):
                feature = self.data[:, ci]
                if feature.dtype != np.object:
                    usize = unique_size(feature)
                    data_t = data_type(usize, feature.size)
                    fmtypes[ci] = data_t.id
                else:
                    fmtypes[ci] = Fmtypes.TEXT.id
        elif not isinstance(fmtypes, np.ndarray):
            fmtypes = np.asarray(fmtypes, dtype=np.dtype(int))
        self.fmtypes = fmtypes

    def features_fmtype(self, fmtype):
        from ml.utils.numeric_functions import features_fmtype
        return features_fmtype(self.fmtypes, fmtype)

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @property
    def data(self):
        """
        eturn the data in the dataset
        """
        return self._get_data('data')

    def to_iter(self, dtype=dtype, chunks_size=258):
        def iter_(data):
            c = 0
            init = 0
            end = chunks_size
            max_iter = round(self.data.shape[0] / float(chunks_size), 0)
            max_iter = 1 if max_iter == 0 else max_iter
            while c < max_iter:
                for e in self.data[init:end]:
                    yield e
                init = end
                end += chunks_size
                c += 1

        return IterLayer(iter_(self.data), shape=self.shape, dtype=dtype).to_chunks(chunks_size=chunks_size)

    def num_features(self):
        """
        return the number of features of the dataset
        """
        if len(self.data.shape) != 2:
            raise Exception("Bad input shape. The input array must have a shape of (X, Y) and not {}".format(self.data.shape))
        else:
            return self.data.shape[1]

    @property
    def shape(self):
        "return the shape of the dataset"
        return self.data.shape

    def exist(self):
        return os.path.exists(self.url())

    def info(self, classes=False):
        """
        :type classes: bool
        :param classes: if true, print the detail of the labels

        This function print the details of the dataset.
        """
        from ml.utils.order import order_table
        print('       ')
        print('DATASET NAME: {}'.format(self.name))
        print('Author: {}'.format(self.author))
        print('Transforms: {}'.format(self.transforms.to_json()))
        print('Applied transforms: {}'.format(self._applied_transforms))
        print('Header Hash: {}'.format(self.hash_header))
        print('Body Hash: {}'.format(self.md5))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Dataset", "Shape", "dType"]
        table = []
        table.append(["dataset", self.shape, self.dtype])
        print(order_table(headers, table, "shape"))

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
        if not isinstance(self.dtype, object):
            data = self.data[:].reshape(self.data.shape[0], -1)
        else:
            data = np.asarray([row.reshape(1, -1)[0] for row in self.data])
        y = set((elem for row in data for elem in row))
        return float(len(y)) / data.size

    def sparcity(self):
        """
        return a value between [0, 1]. If is 0 no zeros exists, if is 1 all data is zero.
        """
        if not isinstance(self.dtype, object):
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

    def build_dataset(self, data, chunks_size=258):
        """
        build a datalabel dataset from data and labels
        """
        data = self.processing(data, apply_transforms=self.apply_transforms,
                            chunks_size=chunks_size)
        self._set_space_shape('data', data.shape, dtype=data.global_dtype)
        end = self.chunks_writer("/data/data", data)
        self._set_space_fmtypes(self.num_features())
        self.build_fmtypes()
        self.md5 = self.calc_md5()

    def empty(self, name, dataset_path=None, apply_transforms=False, transforms=None):
        """
        build an empty Data with the default parameters
        """
        if apply_transforms and self._applied_transforms:
            new_transforms = transforms
            old_applied_transforms = self.transforms
        elif apply_transforms is False and self._applied_transforms is True:
            raise Exception("Error")
        elif apply_transforms is True and self._applied_transforms is False:
            new_transforms = self.transforms + transforms
            old_applied_transforms = None
        else:
            new_transforms = self.transforms + transforms
            old_applied_transforms = None

        data = Data(name=name, 
            dataset_path=dataset_path,
            transforms=new_transforms,
            apply_transforms=apply_transforms,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            rewrite=True)
        data._applied_transforms = apply_transforms
        return data, old_applied_transforms

    def convert(self, name, apply_transforms=False, chunks_size=258,
                percentaje=1, dataset_path=None, transforms=None):
        """
        :type dtype: string
        :param dtype: cast the data to the defined type

        dataset_path is not necesary to especify, this info is obtained from settings.cfg
        """
        data, old_applied_t = self.empty(name, dataset_path=dataset_path, 
            apply_transforms=apply_transforms, transforms=transforms)
        with data:
            data.build_dataset(calc_nshape(self.data, percentaje), chunks_size=chunks_size)
            if old_applied_t is not None:
                transforms = old_applied_t + data.transforms
                data.transforms = transforms
        return data

    def processing(self, data, apply_transforms=True, chunks_size=258):
        """
        :type data: array
        :param data: data to transform

        :type initial: bool
        :param initial: if multirow transforms are added, then this parameter
        indicates the initial data fit

        execute the transformations to the data.

        """
        if apply_transforms and not self.transforms.is_empty():
            return self.transforms.apply(data, chunks_size=chunks_size)
        else:
            if not isinstance(data, IterLayer):
                return IterLayer(data, shape=data.shape, dtype=data.dtype).to_chunks(chunks_size)
            return data

    @classmethod
    def to_DF(self, dataset):
        if len(dataset.shape) > 2:
            dataset = dataset.reshape(dataset.shape[0], -1)
        columns_name = map(lambda x: "c"+str(x), range(dataset.shape[-1]))
        return pd.DataFrame(data=dataset, columns=columns_name)

    def to_df(self):
        """
        convert the dataset to a dataframe
        """
        return self.to_DF(self.data[:])

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
    @property
    def ltype(self):
        return self.labels.dtype

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
        return dict(zip(*np.unique(self.labels, return_counts=True)))

    def only_labels(self, labels):
        """
        :type labels: list
        :param labels: list of labels

        return a tuple of arrays with data and only the selected labels.
        """
        try:
            dataset, n_labels = self.only_labels_from_data(self, labels)
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

    def info(self, classes=False):
        """
        :type classes: bool
        :param classes: if true, print the detail of the labels

        This function print the details of the dataset.
        """
        from ml.utils.order import order_table
        print('       ')
        print('DATASET NAME: {}'.format(self.name))
        print('Author: {}'.format(self.author))
        print('Transforms: {}'.format(self.transforms.to_json()))
        print('Applied transforms: {}'.format(self._applied_transforms))
        print('Header Hash: {}'.format(self.hash_header))
        print('Body Hash: {}'.format(self.md5))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Dataset", "Shape", "dType", "Labels", "ltype"]
        table = []
        table.append(["dataset", self.shape, self.dtype, self.labels.size, self.ltype])
        print(order_table(headers, table, "shape"))
        if classes == True:
            headers = ["class", "# items", "%"]
            items = [(cls, total, (total/float(self.shape[0]))*100) 
                    for cls, total in self.labels_info().items()]
            items_p = [0, 0]
            print(order_table(headers, items, "# items"))

    def build_dataset(self, data, labels, chunks_size=258):
        """
        build a datalabel dataset from data and labels
        """
        data = self.processing(data, apply_transforms=self.apply_transforms, 
                            chunks_size=chunks_size)
        if not isinstance(labels, IterLayer):
            labels = IterLayer(labels, shape=labels.shape, dtype=labels.dtype).to_chunks(chunks_size)
        self._set_space_shape('data', data.shape, data.global_dtype)
        self._set_space_shape('labels', labels.shape, labels.dtype)
        self.chunks_writer("/data/data", data)
        self.chunks_writer("/data/labels", labels)
        self._set_space_fmtypes(self.num_features())
        self.build_fmtypes()
        self.md5 = self.calc_md5()

    def from_data(self, data, labels, chunks_size=258):
        if isinstance(labels, str):
            data = self.processing(data, apply_transforms=self.apply_transforms, 
                chunks_size=chunks_size)
            data_shape = list(data.shape[:-1]) + [data.shape[-1] - 1]
            self._set_space_shape('data', data_shape, data.global_dtype)
            self._set_space_shape('labels', (data.shape[0],), data.global_dtype)
            self.chunks_writer_split("/data/data", "/data/labels", data, labels)
            self.md5 = self.calc_md5()
        else:
            self.build_dataset(data, labels, chunks_size=chunks_size)

    def empty(self, name, apply_transforms=False, dataset_path=None,
                transforms=None):
        """
        build an empty DataLabel with the default parameters
        """
        if apply_transforms and self._applied_transforms:
            new_transforms = transforms
            old_applied_transforms = self.transforms
        elif apply_transforms is False and self._applied_transforms is True:
            raise Exception("Error")
        elif apply_transforms is True and self._applied_transforms is False:
            new_transforms = self.transforms + transforms
            old_applied_transforms = None
        else:
            new_transforms = self.transforms + transforms
            old_applied_transforms = None

        dl = DataLabel(name=name, 
            dataset_path=dataset_path,
            transforms=new_transforms,
            apply_transforms=apply_transforms,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            rewrite=True)
        dl._applied_transforms = apply_transforms
        return dl, old_applied_transforms

    def convert(self, name, apply_transforms=False, percentaje=1, 
                dataset_path=None, transforms=None, chunks_size=258):
        """
        :type dtype: string
        :param dtype: cast the data to the defined type

        dataset_path is not necesary to especify, this info is obtained from settings.cfg
        """
        dl, old_applied_t = self.empty(name, apply_transforms=apply_transforms, 
                        dataset_path=dataset_path, transforms=transforms)
        with dl:
            dl.build_dataset(calc_nshape(self.data, percentaje), 
                calc_nshape(self.labels, percentaje), chunks_size=chunks_size)
            if old_applied_t is not None:
                transforms = old_applied_t + dl.transforms
                dl.transforms = transforms
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
        if labels2numbers == False:
            df = self.to_DF(self.data[:], self.labels[:])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(self.labels[:])
            df = self.to_DF(self.data[:], le.transform(self.labels[:]))
        return df

    @classmethod
    def from_DF(self, name, df, transforms=None, apply_transforms=None, path=None):
        pass

    def to_data(self):
        name = self.name + "_data_" + uuid.uuid4().hex
        data, _ = super(DataLabel, self).empty(name, apply_transforms=self.apply_transforms)
        with data:
            data.build_dataset(self.data)
        return data

    def plot(self, view=None, type_g=None, columns=None):        
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if type_g == 'box':
            sns.set(style="whitegrid", palette="pastel", color_codes=True)
            col = int(view)
            sns.boxplot(x=self.labels, y=self.data[:,col], palette="PRGn")
            #sns.despine(offset=10, trim=True)
        elif type_g == "violin":
            sns.set(style="whitegrid", palette="pastel", color_codes=True)
            col = int(view)
            sns.violinplot(x=self.labels, y=self.data[:,col], palette="PRGn", inner="box")
            #sns.despine(offset=10, trim=True)
        elif type_g == "hist" and self.num_features() <= 64:
            #size = int(round(self.num_features() ** .5))
            #f, axarr = plt.subplots(size, size, sharey=True, sharex=True)
            #base = 0
            #for i in range(size):
            #    for j in range(size):
            #        axarr[i, j].set_title('Feature {}'.format(base+1))
            col = int(view)
            sns.distplot(self.data[:, col], bins=50, 
                kde=False, rug=False, color="b")#, ax=axarr[i, j])
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
            df = df.iloc[:, :self.num_features()].astype(np.float64) 
            corr = df.corr()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5})
        else:
            if self.shape[1] > 2:
                from ml.ae.extended.w_keras import PTsne
                dl = DataLabel(name=self.name+"_2d_", 
                        dataset_path=self.dataset_path,
                        transforms=None,
                        apply_transforms=False,
                        compression_level=9,
                        rewrite=False)

                if not dl.exist():
                    ds = self.to_data()
                    classif = PTsne(model_name="tsne", model_version="1", 
                        check_point_path="/tmp/", dataset=ds, latent_dim=2)
                    classif.train(batch_size=50, num_steps=120)
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
        from ml.utils.numeric_functions import unique_size, data_type
        from ml.utils.numeric_functions import missing, zeros
        from ml.fmtypes import fmtypes_map

        headers = ["feature", "label", "missing", "mean", "std dev", "zeros", 
            "min", "25%", "50%", "75%", "max", "type", "unique"]
        table = []
        li = self.labels_info()
        feature_column = defaultdict(dict)
        for label in li:
            mask = (self.labels[:] == label)
            data = self.data[:][mask]
            for i, column in enumerate(data.T):
                percentile = np.nanpercentile(column, [0, 25, 50, 75, 100])
                values = [
                    "{0:.{1}f}%".format(missing(column), 2),
                    np.nanmean(column),  
                    np.nanstd(column),
                    "{0:.{1}f}%".format(zeros(column), 2),
                    ]
                values.extend(percentile)
                feature_column[i][label] = values

        data = self.data
        fmtypes = self.fmtypes
        for feature, rows in feature_column.items():
            column = data[:, feature]            
            usize = unique_size(column)
            if fmtypes is None or len(fmtypes) != self.num_features():
                data_t = data_type(usize, column.size).name
            else:
                data_t = fmtypes_map[fmtypes[feature]]

            for label, row in rows.items():
                table.append([feature, label] + row + [data_t, str(usize)])
                feature = "-"
                data_t = "-"
                usize = "-"

        return tabulate(table, headers)
    
    def cv(self, train_size=.7, valid_size=.1):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[:], self.labels[:], train_size=round(train_size+valid_size, 2), random_state=0)
        size = self.data.shape[0]

        valid_size_index = int(round(size * valid_size))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def cv_ds(self, train_size=.7, valid_size=.1, dataset_path=None, apply_transforms=True):
        data = self.cv(train_size=train_size, valid_size=valid_size)
        train_ds = DataLabel(dataset_path=dataset_path, transforms=self.transforms, 
                            apply_transforms=apply_transforms)
        with train_ds:
            train_ds.build_dataset(data[0], data[3])
        validation_ds = DataLabel(dataset_path=dataset_path, transforms=self.transforms,
                                apply_transforms=apply_transforms)
        with validation_ds:
            validation_ds.build_dataset(data[1], data[4])
        test_ds = DataLabel(dataset_path=dataset_path, transforms=self.transforms,
                        apply_transforms=apply_transforms)
        with test_ds:
            test_ds.build_dataset(data[2], data[5])
        
        return train_ds, validation_ds, test_ds


class DataSetBuilderImage(DataLabel):
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


class DataLabelSetFile(DataLabel):
    """
    Class for csv dataset build. Get the data from a csv's file.
    
    :type training_data_path: list
    :param training_data_path: list of files paths

    :type sep: list
    :param sep: list of strings separators for each file

    kwargs are the same that DataSetBuilder's options
    """

    def __init__(self, name=None, training_data_path=None,
                sep=None, merge_field="id", na_values=None, **kwargs):
        super(DataLabelSetFile, self).__init__(name, **kwargs)
        self.training_data_path = training_data_path
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
                sep_l = [self.sep for _ in self.training_data_path]
            else:
                sep_l = self.sep
            old_df = None
            for sep, path in zip(sep_l, self.training_data_path):
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
        super(DataLabelSetFile, self).build_dataset(data, labels)
    
    def get_sep_path(self):
        if isinstance(self.training_data_path, list):
            if not isinstance(self.sep, list):
                delimiters = [self.sep for _ in self.training_data_path]
            else:
                delimiters = [self.sep]
            return self.training_data_path, delimiters
        else:
            return [self.training_data_path], [self.sep]

    def to_db(self):
        from ml.db.utils import build_schema, insert_rows
        import csv
        from ml.utils.files import filename_n_ext_from_path

        training_data_path, delimiters = self.get_sep_path()
        for data_path, delimiter in zip(training_data_path, delimiters):
            table_name = filename_n_ext_from_path(data_path)
            with open(data_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=delimiter)
                for row in csv_reader:
                    header = row
                    break
                build_schema(table_name, header, 
                    [Fmtypes.TEXT]*len(header),
                    self.merge_field)
                insert_rows(csv_reader, table_name, header)


class DataLabelFold(object):
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
        with dl:
            for i, (train, test) in enumerate(skf.split(dl.data, dl.labels)):
                dsb = DataLabel(name=self.name+"_"+str(i), 
                    dataset_path=self.dataset_path,
                    transforms=None,
                    apply_transforms=False,
                    description="",
                    author="",
                    compression_level=9,
                    rewrite=True)
                data = dl.data[:]
                labels = dl.labels[:]
                with dsb:                    
                    dsb.build_dataset(data[train], labels[train]) 
                    #test_data=data[test], 
                    #    test_labels=labels[test], validation_data=data[validation], 
                    #    validation_labels=labels[validation])
                    yield dsb

    def build_dataset(self, dataset=None):
        """
        :type dataset: DataLabel
        :param dataset: dataset to fold

        construct the dataset fold from an DataSet class
        """
        for dsb in self.create_folds(dataset):
            self.splits.append(dsb.name)

    def get_splits(self):
        """
        return an iterator of datasets with the splits of original data
        """
        for split in self.splits:
            yield DataLabel(name=split, dataset_path=self.dataset_path, rewrite=False)

    def destroy(self):
        for split in self.get_splits():
            split.destroy()
