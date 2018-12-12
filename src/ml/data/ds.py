import datetime
import logging
import os
import json

import dill as pickle

import numpy as np
import pandas as pd
import xarray as xr

from tqdm import tqdm
from ml.abc.data import AbsDataset
from ml.data.it import Iterator, BaseIterator
from ml.utils.config import get_settings
from ml.utils.files import build_path
from ml.utils.basic import Hash, StructArray, isnamedtupleinstance
from ml.abc.driver import AbsDriver
from ml.data.drivers import Memory

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
    

# class Memory:
#    def __init__(self):
#        self.spaces = {}
#        self.attrs = {}

#    def __contains__(self, item):
#        return item in self.spaces

#    def __getitem__(self, key):
#        if key is not None:
#            levels = [e for e in key.split("/") if e != ""]
#        else:
#            levels = ["c0"]
#        v =  self._get_level(self.spaces, levels)
#        return v

#    def __setitem__(self, key, value):
#        if key is not None:
#            levels = [e for e in key.split("/") if e != ""]
#        else:
#            levels = ["c0"]
#        v = self._get_level(self.spaces, levels[:-1])
#        if v is None:
#            self.spaces[levels[-1]] = value
#        else:
#            v[levels[-1]] = value

#    def require_group(self, name):
#        if name not in self:
#            self[name] = Memory()

#    def require_dataset(self, name, shape, dtype='float', **kwargs):
#        logging.info(kwargs)
#        if name not in self:
#            self[name] = np.empty(shape, dtype=dtype)

#    def get(self, name, value):
#        try:
#            return self[name]
#        except KeyError:
#            return value

#    def _get_level(self, spaces, levels):
#        if len(levels) == 1:
#            return spaces[levels[0]]
#        elif len(levels) == 0:
#            return None
#        else:
#            return self._get_level(spaces[levels[0]], levels[1:])

#    def close(self):
#        pass

#    def keys(self):
#        return self.spaces.keys()


class Data(AbsDataset):
    """
    Base class for dataset build. Get data from memory.
    create the initial values for the dataset.
    """
    def __init__(self, name: str = None, dataset_path: str = None, description: str = '', author: str = '',
                 clean: bool = False, driver: AbsDriver = None, group_name: str = None):

        if name is None:
            raise Exception("I can't build a dataset without a name, plese add a name to this dataset.")

        self.name = name
        self.header_map = ["author", "description"]
        if driver is None:
            self.driver = Memory()
        else:
            self.driver = driver
        self.group_name = group_name
        self._it = None

        if dataset_path is None:
            self.dataset_path = settings["dataset_path"]
        else:
            self.dataset_path = dataset_path

        ds_exist = self.exists()
        if ds_exist and clean:
            self.destroy()
            ds_exist = False

        if not ds_exist and (self.driver.mode == 'w' or self.driver.mode == 'a'):
            build_path([self.dataset_path, self.group_name])
            self.create_route()
            self.author = author
            self.description = description
            self.timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M UTC")
            self.compressor_params = self.driver.compressor_params
            self.dtypes = None
            self.hash = None

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
        if value is not None:
            self._set_attr('hash', value)

    @property
    def compressor_params(self):
        return json.loads(self._get_attr('compressor_params'))

    @compressor_params.setter
    def compressor_params(self, value):
        self._set_attr('compressor_params', json.dumps(value))

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @property
    def data(self):
        groups_data = [(group, self._get_data(group)) for group in self.groups]
        return StructArray(groups_data)

    def __enter__(self):
        self.driver.enter(self.url)
        return self

    def __exit__(self, exc_type, value, traceback):
        self.driver.exit()
        self._it = None

    def __getitem__(self, key) -> StructArray:
        return self.data[key]

    def __setitem__(self, key, value):
        # self.data[key] = value
        return NotImplemented

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = self.run()
        return next(self._it)

    def run(self):
        i = 0
        data = self.data
        while i < len(data):
            yield data[i]
            i += 1

    def _set_group_shape(self, name: str, shape: tuple, dtype: np.dtype, group: str) -> None:
        dtype = self.driver.auto_dtype(dtype)
        self.driver.require_group(group)
        self.driver.require_dataset(group, name, shape, dtype=dtype)

    def _get_data(self, group):
        return self.driver["data"][group]

    def _set_attr(self, name, value):
        if value is not None:
            with self:
                self.driver.attrs[name] = value

    def _get_attr(self, name):
        try:
            return self.driver.attrs[name]
        except KeyError:
            log.debug("Not found attribute {} in file {}".format(name, self.url))
            return None
        except IOError as e:
            log.debug(e)
            log.debug("Error opening {} in file {}".format(name, self.url))
            return None

    def batchs_writer(self, groups, data):
        log.info("Writing with batch size {}".format(getattr(data, 'batch_size', 0)))
        if isinstance(data, StructArray):
            for group in data.groups:
                self.driver["data"][group] = data[group].to_ndarray()
        elif getattr(data, 'batch_size', 0) > 0 and data.batch_type == "structured":
            log.debug("WRITING STRUCTURED BATCH")
            end = {}
            init = {}
            for group in groups:
                end[group] = 0
                init[group] = 0
            for smx in tqdm(data, total=data.num_splits()):
                for group in groups:
                    end[group] += smx[group].shape[0]
                    self.driver["data"][group][init[group]:end[group]] = smx[group].to_ndarray()
                    init[group] = end[group]
        elif getattr(data, 'batch_size', 0) > 0 and data.batch_type == "array":
            log.debug("WRITING ARRAY BATCH")
            init = 0
            end = 0
            for smx in tqdm(data, total=data.num_splits()):
                end += smx.shape[0]
                for group in groups:
                    self.driver["data"][group][init:end] = smx
                init = end
        elif getattr(data, 'batch_size', 0) > 0 and data.batch_type == "df":
            log.debug("WRITING DF BATCH")
            init = 0
            end = 0
            for smx in tqdm(data, total=data.num_splits()):
                end += smx.shape[0]
                for group in groups:
                    self.driver["data"][group][init:end] = smx[group]
                init = end
        elif getattr(data, 'batch_size', 0) > 0:
            log.debug("WRITING GENERIC BATCH")
            init = 0
            end = 0
            for smx in tqdm(data, total=data.num_splits()):
                if hasattr(smx, 'shape') and len(smx.shape) > 0:
                    end += smx.shape[0]
                else:
                    end += 1
                for i, group in enumerate(groups):
                    self.driver["data"][group][init:end] = smx[i]
                init = end
        elif data.type_elem == tuple or data.type_elem == list or data.type_elem == np.ndarray and\
                (data.type_elem != pd.DataFrame or not isnamedtupleinstance(data.type_elem)):
            log.debug("WRITING ELEMS IN LIST")
            init = 0
            end = 0
            for smx in tqdm(data, total=data.num_splits()):
                end += 1
                for i, group in enumerate(groups):
                    self.driver["data"][group][init:end] = [smx]
                init = end
        elif data.type_elem == pd.DataFrame or isnamedtupleinstance(data.type_elem):
            log.debug("WRITING ELEMS IN TABULAR")
            init = 0
            end = 0
            for smx in tqdm(data, total=data.num_splits()):
                end += 1
                for group in groups:
                    self.driver["data"][group][init:end] = getattr(smx, group)
                init = end
        else:
            log.debug("WRITING SCALARS")
            init = 0
            end = 0
            if len(groups) > 0:
                group = groups[0]
                for smx in tqdm(data, total=data.num_splits()):
                    end += 1
                    self.driver["data"][group][init:end] = smx
                    init = end

    def destroy(self):
        """
        delete the hdf5 file
        """
        from ml.utils.files import rm
        rm(self.url)
        log.debug("DESTROY {}".format(self.url))

    @property
    def url(self) -> str:
        """
        return the path where is saved the dataset
        """
        filename = "{}.{}".format(self.name, self.driver.ext)
        if self.group_name is None:
            return os.path.join(self.dataset_path, filename)
        else:
            return os.path.join(self.dataset_path, self.group_name, filename)

    def exists(self) -> bool:
        if self.driver.persistent is True:
            return os.path.exists(self.url)
        else:
            return False

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def groups(self) -> list:
        return [c for c, _ in self.dtypes]

    @groups.setter
    def groups(self, value) -> None:
        with self:
            if len(value) == len(self.dtypes):
                dtypes = [(col, dtypes[1]) for col, dtypes in zip(value, self.dtypes)]
            else:
                raise Exception
        self.dtypes = dtypes

    @property
    def dtypes(self) -> list:
        if "meta" in self.driver:
            return [(col, np.dtype(dtype)) for col, dtype in self.driver["meta"].get("dtypes", [])]
        else:
            return []

    @dtypes.setter
    def dtypes(self, value):
        if value is not None:
            with self:
                self._set_group_shape("dtypes", (len(value), 2), np.dtype('object'), group="meta")
                for i, (group, dtype) in enumerate(value):
                    self.driver["meta"]["dtypes"][i] = (group, dtype.str)

    def info(self):
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
        # print columns

    def calc_hash(self, with_hash: str = 'sha1', batch_size: int = 1080) -> str:
        hash_obj = Hash(hash_fn=with_hash)
        header = [getattr(self, attr) for attr in self.header_map]
        hash_obj.hash.update("".join(header).encode("utf-8"))
        for group in self.groups:
            it = Iterator(self.data[group]).batchs(batch_size=batch_size, batch_type="array")
            hash_obj.update(it)
        return str(hash_obj)

    def from_data(self, data, batch_size: int = 258, with_hash: str = "sha1"):
        """
        build a datalabel dataset from data and labels
        """
        if isinstance(data, AbsDataset):
            print("ok")
        elif isinstance(data, StructArray):
            data = Iterator(data).batchs(batch_size=batch_size, batch_type="structured")
        elif isinstance(data, Iterator):
            data = data.batchs(batch_size=batch_size, batch_type="structured")
        elif isinstance(data, dict):
            str_arrays = []
            for elem in data.values():
                if isinstance(elem, StructArray):
                    str_arrays.append(elem)
                else:
                    if len(str_arrays) > 0:
                        raise NotImplementedError
            data = sum(str_arrays)
            data = Iterator(data).batchs(batch_size=batch_size, batch_type="structured")
        elif not isinstance(data, BaseIterator):
            data = Iterator(data).batchs(batch_size=batch_size, batch_type="structured")
        self.dtypes = data.dtypes
        with self:
            groups = []
            if data.is_multidim():
                for group, dtype in self.dtypes:
                    self._set_group_shape(group, data.shape[group], dtype, group="data")
                    groups.append(group)
            else:
                for group, dtype in self.dtypes:
                    self._set_group_shape(group, data.shape, dtype, group="data")
                    groups.append(group)
            self.batchs_writer(groups, data)
            if with_hash is not None:
                c_hash = self.calc_hash(with_hash=with_hash)
            else:
                c_hash = None
        self.hash = c_hash

    def to_df(self) -> pd.DataFrame:
        return self.data.to_df()

    def to_ndarray(self, dtype=None) -> np.ndarray:
        return self.data.to_ndarray(dtype=dtype)

    def to_xrds(self) -> xr.Dataset:
        return self.data.to_xrds()

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
        target_t = le.fit_transform(self._get_data(target))
        groups = [group for group in self.groups if group != target]
        with open(save_to, 'w') as f:
            for row in libsvm_row(target_t, self.data[groups].to_ndarray()):
                f.write(" ".join(row))
                f.write("\n")



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

