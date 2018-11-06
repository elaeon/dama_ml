"""

"""

import datetime
import logging
import os
import uuid
import json

import dill as pickle
import h5py
import numpy as np
import pandas as pd
from ml.data.abc import AbsDataset
from ml.data.it import Iterator
from ml.random import downsample
from ml.random import sampling_size
from ml.utils.config import get_settings
from ml.utils.files import build_path
from ml.utils.basic import Hash, unique_dtypes, StructArray

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
    except Exception as e:
        log.error("{} {}".format(e, path))


def calc_nshape(data, value):
    if value is None or not (0 < value <= 1) or data is None:
        value = 1
    return int(round(data.shape[0] * value, 0))
    

class Memory:
    def __init__(self):
        self.spaces = {}
        self.attrs = {}

    def __contains__(self, item):
        return item in self.spaces

    def __getitem__(self, key):
        levels = [e for e in key.split("/") if e != ""]
        v =  self._get_level(self.spaces, levels)
        return v

    def __setitem__(self, key, value):
        levels = [e for e in key.split("/") if e != ""]
        v = self._get_level(self.spaces, levels[:-1])
        if v is None:
            self.spaces[levels[-1]] = value
        else:
            v[levels[-1]] = value

    def require_group(self, name):
        if name not in self:
            self[name] = Memory()

    def require_dataset(self, name, shape, dtype='float', **kwargs):
        if name not in self:
            self[name] = np.empty(shape, dtype=dtype)

    def get(self, name, value):
        try:
            return self[name]
        except KeyError:
            return value

    def _get_level(self, spaces, levels):
        if len(levels) == 1:
            return spaces[levels[0]]
        elif len(levels) == 0:
            return None
        else:
            return self._get_level(spaces[levels[0]], levels[1:])

    def close(self):
        pass

    def keys(self):
        return self.spaces.keys()


class HDF5Dataset(AbsDataset):
    def __enter__(self):
        if self.driver == "core":
            if self.f is None:
                self.f = Memory()
        else:
            if self.f is None:
                self.f = h5py.File(self.url(), mode=self.mode)
        return self

    def __exit__(self, type, value, traceback):
        if self.f is not None:
            self.f.close()
            if self.driver != "core":
                self.f = None

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._get_data(key)
            except KeyError:
                index = self.columns.index(key)
                return self.data[:, index]
        elif isinstance(key, slice):
            if key.start is None:
                start = 0
            else:
                start = key.start

            if key.stop is None:
                stop = self.shape[0]
            else:
                stop = key.stop
            return self.data[start:stop]
        elif key is None:
            return self.data[:]
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __next__(self):
        return next(self.__iter__())

    def auto_dtype(self, ttype):
        if ttype == np.dtype("O") or ttype.kind == "U":
            return h5py.special_dtype(vlen=str)
        else:
            return ttype

    def _set_space_shape(self, name, shape, dtype):
        with self:
            #self.f.require_group(key)
            dtype = self.auto_dtype(dtype)
            #self.f[key].require_dataset(name, shape, dtype=dtype, chunks=True,
            #    exact=True, **self.zip_params)
            self.f.require_dataset(name, shape, dtype=dtype, chunks=True,
                exact=True, **self.zip_params)

    def _get_data(self, key):
        return self.f[key]

    def _set_attr(self, name, value):
        with self:
            self.f.attrs[name] = value
            
    def _get_attr(self, name):
        try:
            return self.f.attrs[name]
        except KeyError:
            log.debug("Not found attribute {} in file {}".format(name, self.url()))
            return None
        except IOError as e:
            log.debug("Error opening {} in file {}".format(name, self.url()))
            return None

    def chunks_writer(self, name, data, init=0):
        from tqdm import tqdm
        log.info("Writing with chunks size {}".format(data.batch_size))
        end = init
        with self:
            for smx in tqdm(data, total=data.num_splits()):
                if hasattr(smx, 'shape') and len(smx.shape) >= 1 and data.has_batchs:
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
                    if type(array) == np.ndarray:
                        array_type = array.dtype
                        if array.dtype != self.f[name].dtype:                        
                            must_type = self.f[name].dtype
                        else:
                            raise TypeError(e)
                    else:
                        array_type = type(array[0])
                        must_type = self.f[name].dtype
                    raise TypeError("All elements in array must be of type '{}' but found '{}'".format(
                        must_type, array_type))
            return end

    def chunks_writer_columns(self, keys, data, init=0):
        from tqdm import tqdm
        log.info("Writing with chunks size {}".format(data.batch_size))
        end = init
        with self:
            for smx in tqdm(data, total=data.num_splits()):
                if hasattr(smx, 'shape') and len(smx.shape) >= 1 and data.has_batchs:
                    end += smx.shape[0]
                else:
                    end += 1
                for key in keys:
                    self.f[key][init:end] = smx[key]
                init = end

    def destroy(self):
        """
        delete the hdf5 file
        """
        from ml.utils.files import rm
        rm(self.url())
        log.debug("DESTROY {}".format(self.url()))

    def url(self) -> str:
        """
        return the path where is saved the dataset
        """
        if self.group_name is None:
            return os.path.join(self.dataset_path, self.name)
        else:
            return os.path.join(self.dataset_path, self.group_name, self.name)

    def exists(self) -> bool:
        return os.path.exists(self.url())

    # def reader(self, batch_size: int=0, dtype: list=None) -> Iterator:
    #    return Iterator(self, dtype=self.dtype).batchs(batch_size, dtype=dtype)

    @property
    def shape(self) -> tuple:
        if 'data' not in self.f.keys():
            return self._get_data(self.columns[0]).shape
        else:
            return self.data.shape

    @property
    def columns(self) -> list:
        dtypes = self.dtypes
        return [c for c, _ in dtypes]

    @columns.setter
    def columns(self, value):
        with self:
            if len(value) == len(self.dtypes):
                dtypes = [(col, dtypes[1]) for col, dtypes in zip(value, self.dtypes)]
            else:
                raise Exception
        self.dtypes = dtypes

    @property
    def dtypes(self) -> list:
        return [(col, np.dtype(dtype)) for col, dtype in self.f.get("dtypes", None)]

    @dtypes.setter
    def dtypes(self, value):
        self._set_space_shape("dtypes", (len(value), 2), 'object')
        with self:
            for i, (c, dtype) in enumerate(value):
                self.f["dtypes"][i] = (c, dtype.name)

    def num_features(self):
        """
        return the number of features of the dataset
        """
        if len(self.shape) > 1:
            return self.shape[-1]
        else:
            return 1


class Data(HDF5Dataset):
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
    def __init__(self, name=None, dataset_path=None, description='', author='', 
                compression_level=0, clean=False, mode='a', driver='default', group_name=None):

        if name is None:
            raise Exception("I can't build a dataset without a name, plese add a name to this dataset.")

        self.name = name
        self.header_map = ["author", "description"]
        self.f = None
        self.driver = driver
        self.mode = mode
        self.group_name = group_name

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path

        ds_exist = self.exists()
        if ds_exist and clean:
            self.destroy()
            ds_exist = False

        if not ds_exist and (self.mode == 'w' or self.mode == 'a'):
            build_path([self.dataset_path, self.group_name])
            self.create_route()
            self.author = author
            self.description = description
            self.timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M UTC")
            self.zip_params = json.dumps({"compression": "gzip", "compression_opts": compression_level})

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
    def hash(self):
        return self._get_attr('hash')

    @hash.setter
    def hash(self, value):
        self._set_attr('hash', value)

    @property
    def zip_params(self):
        return json.loads(self._get_attr('zip_params'))

    @zip_params.setter
    def zip_params(self, value):
        self._set_attr('zip_params', value)

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @property
    def data(self):
        try:
            columns = [("c0", self._get_data('data'))]
        except KeyError:
            columns = [(column, self._get_data(column)) for column in self.columns]
        return StructArray(columns)

    def info(self, classes=False):
        """
        :type classes: bool
        :param classes: if true, print the detail of the labels

        This function print the details of the dataset.
        """
        from ml.utils.order import order_table
        print('       ')
        print('Dataset NAME: {}'.format(self.name))
        print('Author: {}'.format(self.author))
        print('Hash: {}'.format(self.hash))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Dataset", "Shape"]
        table = []
        with self:
            table.append(["dataset", self.shape])
        print(order_table(headers, table, "shape"))
        ###print columns

    def calc_hash(self, hash_fn: str='sha1', batch_size: int=1080) -> str:
        hash_obj = Hash(hash_fn=hash_fn)
        header = [getattr(self, attr) for attr in self.header_map]
        hash_obj.hash.update("".join(header).encode("utf-8"))
        hash_obj.update(self.reader(batch_size=batch_size, dtype=self.data.global_dtype))
        return str(hash_obj)

    def from_data(self, data, length=None, batch_size: int=258):
        """
        build a datalabel dataset from data and labels
        """
        if isinstance(data, AbsDataset):
            print("ok")
        else:
            if length is None and data.shape[0] is not None:
                length = data.shape[0]
            data = Iterator(data).batchs(batch_size)
            data = data.it_length(length)
        #self.hash = self.calc_hash()
        self.dtypes = data.dtypes

        with self:
            u_dtypes = unique_dtypes(self.dtypes)
            if len(u_dtypes) == 1:
                dtype = np.dtype(u_dtypes[0])
                self._set_space_shape('data', data.shape, dtype=dtype)
                self.chunks_writer("/data", data)
            else:
                columns = []
                for col, dtype in self.dtypes:
                    shape = (data.shape[0],)#data[col].shape
                    self._set_space_shape(col, shape, dtype=dtype)
                    columns.append(col)
                self.chunks_writer_columns(columns, data)

    def to_df(self):
        return self.data.to_df()

    def to_ndarray(self, dtype=None):
        return self.data.to_ndarray(dtype=dtype)

    @staticmethod
    def concat(datasets, batch_size:int=0, name:str=None):
        ds0 = datasets.pop(0)
        i = 0
        to_destroy = []
        while len(datasets) > 0:
            ds1 = datasets.pop(0)
            if len(datasets) == 0:
                name_ds = name
            else:
                name_ds = "test_"+str(i)
            data = Data(name=name_ds, dataset_path="/tmp", clean=True)
            with ds0, ds1, data:
                it = ds0.reader(batch_size=batch_size).concat(ds1.reader(batch_size=batch_size))
                data.from_data(it)
            i += 1
            ds0 = data
            to_destroy.append(data)
        for ds in to_destroy[:-1]:
            ds.destroy()
        return ds0

    def create_route(self):
        """
        create directories if the dataset_path does not exist
        """
        if os.path.exists(self.dataset_path) is False:
            os.makedirs(self.dataset_path)

    @staticmethod
    def url_to_name(url):
        dataset_url = url.split("/")
        name = dataset_url[-1]
        path = "/".join(dataset_url[:-1])
        return name, path

    def to_libsvm(self, target, save_to=None):
        """
        tranforms the dataset into libsvm format
        """
        from ml.utils.seq import libsvm_row
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(self._get_data(target))
        columns = [col for col in self.columns if col != target]
        with open(save_to, 'w') as f:
            for row in libsvm_row(labels, self.data[columns]):
                f.write(" ".join(row))
                f.write("\n")

    def cv(self, train_size=.7, valid_size=.1, unbalanced=None):
        from sklearn.model_selection import train_test_split

        train_size = round(train_size+valid_size, 2)
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[:], self.labels[:], train_size=train_size, random_state=0)
        size = self.data.shape[0]
        valid_size_index = int(round(size * valid_size, 0))
        X_validation = X_train[:valid_size_index]
        y_validation = y_train[:valid_size_index]
        X_train = X_train[valid_size_index:]
        y_train = y_train[valid_size_index:]

        if unbalanced is not None:
            X_unb = [X_train, X_validation]
            y_unb = [y_train, y_validation]
            log.debug("Unbalancing data")
            for X_, y_ in [(X_test, y_test)]:
                X = np.c_[X_, y_]
                y_index = X_.shape[-1]
                unbalanced = sampling_size(unbalanced, y_)
                it = downsample(X, unbalanced, y_index, y_.shape[0])
                v = it.to_memory()
                if v.shape[0] == 0:
                    X_unb.append(v)
                    y_unb.append(v)
                    continue
                X_unb.append(v[:, :y_index])
                y_unb.append(v[:, y_index])
            return X_unb + y_unb

    def cv_ds(self, train_size=.7, valid_size=.1, dataset_path=None):
        data = self.cv(train_size=train_size, valid_size=valid_size)
        train_ds = Data(name="train", dataset_path=dataset_path)
        with train_ds:
            train_ds.from_data(data[0], data[3], data[0].shape[0])
        validation_ds = Data(name="validation", dataset_path=dataset_path)
        with validation_ds:
            validation_ds.from_data(data[1], data[4], data[1].shape[0])
        test_ds = Data(name="test", dataset_path=dataset_path)
        with test_ds:
            test_ds.from_data(data[2], data[5], data[2].shape[0])

        return train_ds, validation_ds, test_ds


class DataLabel(Data):

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

