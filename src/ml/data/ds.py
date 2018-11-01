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
        log.info("Writing with chunks size {}".format(data.chunks_size))
        end = init
        with self:
            for smx in tqdm(data, total=data.num_splits()):
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
        log.info("Writing with chunks size {}".format(data.chunks_size))
        end = init
        with self:
            for smx in tqdm(data, total=data.num_splits()):
                if hasattr(smx, 'shape') and len(smx.shape) >= 1 and data.has_chunks:
                    end += smx.shape[0]
                else:
                    end += 1
                for key in keys:
                    self.f[key][init:end] = smx[key]
                init = end

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
        if self.group_name is None:
            return os.path.join(self.dataset_path, self.name)
        else:
            return os.path.join(self.dataset_path, self.group_name, self.name)

    def exists(self):
        return os.path.exists(self.url())

    def reader(self, chunksize:int=0, df=True) -> Iterator:
        if df is True:
            dtypes = self.dtype
            dtype = [(col, dtypes) for col in self.columns]
        else:
            dtype = self.dtype
        if chunksize == 0:
            it = Iterator(self, dtype=dtype)
            return it
        else:
            it = Iterator(self, dtype=dtype).to_chunks(chunksize)
            return it

    @property
    def shape(self):
        "return the shape of the dataset"
        if 'data' not in self.f.keys():
            return self._get_data(self.columns[0]).shape
        else:
            return self.data.shape

    @property
    def columns(self):
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
    def dtypes(self):
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
            columns = []
            for column in self.columns:
                columns.append((column, self._get_data(column)))
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

    def calc_hash(self, hash_fn:str='sha1', chunksize:int=1080):
        hash = Hash(hash_fn=hash_fn)
        header = [getattr(self, attr) for attr in self.header_map]
        hash.hash.update("".join(header).encode("utf-8"))
        hash.chunks(self.reader(chunksize=chunksize, df=False))
        return str(hash)

    def from_data(self, data, length=None, chunksize=258):
        """
        build a datalabel dataset from data and labels
        """
        if isinstance(data, AbsDataset):
            print("ok")
        else:
            if length is None and data.shape[0] is not None:
                length = data.shape[0]
            data = Iterator(data).to_chunks(chunksize)
            data = data.it_length(length)
        #self.hash = self.calc_hash()
        self.dtypes = data.dtypes

        with self:
            u_dtypes = unique_dtypes(self.dtypes)
            if len(u_dtypes) == 1:
                dtype = np.dtype(u_dtypes[0])
                self._set_space_shape('data', data.shape, dtype=dtype)
                end = self.chunks_writer("/data", data)
            else:
                columns = []
                for col, dtype in self.dtypes:
                    shape = (data.shape[0],)#data[col].shape
                    self._set_space_shape(col, shape, dtype=dtype)
                    columns.append(col)
                self.chunks_writer_columns(columns, data)

    def empty(self, name, dataset_path=None):
        """
        build an empty Data with the default parameters
        """
        data = Data(name=name, 
            dataset_path=dataset_path,
            description=self.description,
            author=self.author,
            compression_level=self.compression_level,
            clean=True)
        return data

    def to_df(self):
        """
        convert the dataset to a dataframe
        """
        return self.data.to_df()

    @staticmethod
    def concat(datasets, chunksize:int=0, name:str=None):
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
                it = ds0.reader(chunksize=chunksize).concat(ds1.reader(chunksize=chunksize))
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

    #@staticmethod
    #def original_ds(name, dataset_path=None):
    #    from pydoc import locate
    #    meta_dataset = Data(name=name, dataset_path=dataset_path, clean=False)
    #    DS = locate(str(meta_dataset.dataset_class))
    #    if DS is None:
    #        return
    #    return DS(name=name, dataset_path=dataset_path, clean=False)

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

    def cv_ds(self, train_size=.7, valid_size=.1, dataset_path=None, apply_transforms=True):
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
        #print('Transforms: {}'.format(self.transforms.to_json()))
        print('Header Hash: {}'.format(self.hash_header))
        print('Body Hash: {}'.format(self.md5))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Dataset", "Shape", "dType", "Labels", "ltype"]
        table = []
        with self:
            table.append(["dataset", self.shape, self.dtype, self.labels.size, self.ltype])
        print(order_table(headers, table, "shape"))
        if classes == True:
            headers = ["class", "# items", "%"]
            with self:
                items = [(cls, total, (total / float(self.shape[0])) * 100)
                         for cls, total in self.labels_info().items()]
            print(order_table(headers, items, "# items"))

    def from_data(self, data, labels, length=None, chunks_size=258, transform=True):
        if length is None and data.shape[0] is not None:
            length = data.shape[0]
        data = self.processing(data, apply_transforms=transform,
            chunks_size=chunks_size)
        if isinstance(labels, str):
            data = data.it_length(length)
            data_shape = list(data.shape[:-1]) + [data.shape[-1] - 1]
            self._set_space_shape('data', data_shape, data.global_dtype)
            if isinstance(data.dtype, list):
                dtype_dict = dict(data.dtype)
                self._set_space_shape('labels', (data.shape[0],), dtype_dict[labels])
            else:
                self._set_space_shape('labels', (data.shape[0],), data.global_dtype)
            self.chunks_writer_split("/data/data", "/data/labels", data, labels)
        else:
            if not isinstance(labels, Iterator):
                labels = Iterator(labels, dtype=labels.dtype).to_chunks(chunks_size)
            data = data.it_length(length)
            labels = labels.it_length(length)
            self._set_space_shape('data', data.shape, data.global_dtype)
            self._set_space_shape('labels', labels.shape, labels.dtype)
            self.chunks_writer("/data/data", data)
            self.chunks_writer("/data/labels", labels)

        #self.md5 = self.calc_md5()
        columns = data.columns
        self._set_space_fmtypes(len(columns))
        if columns is not None:
            self.columns = columns

    def to_df(self, include_target=True):
        """
        convert the dataset to a dataframe
        """
        if len(self.shape) > 2:
            raise Exception("I could don't convert a multiarray to tabular")

        if include_target == True:
            columns_name = list(self.columns) + ["target"]
            return pd.DataFrame(data=np.column_stack((self[:], self.labels[:])), columns=columns_name)
        else:
            columns_name = list(self.columns)
            return pd.DataFrame(data=self[:], columns=columns_name)

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
                    description="",
                    author="",
                    compression_level=3,
                    clean=True)
                data = dl.data[:]
                labels = dl.labels[:]
                with dsb:
                    dsb.from_data(data[train], labels[train])
                    yield dsb

    def from_data(self, dataset=None):
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
            yield DataLabel(name=split, dataset_path=self.dataset_path)

    def destroy(self):
        for split in self.get_splits():
            split.destroy()
