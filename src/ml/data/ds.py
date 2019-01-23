import datetime
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import re

from tqdm import tqdm
from ml.abc.data import AbsDataset
from ml.data.it import Iterator, BaseIterator, BatchIterator
from ml.utils.files import build_path
from ml.utils.basic import Hash, StructArray, Array
from ml.abc.driver import AbsDriver
from ml.data.drivers import Memory
from ml.utils.logger import log_config
from ml.utils.config import get_settings
from ml.utils.decorators import cache, clean_cache
from ml.utils.files import get_dir_file_size, rm
from ml.utils.order import order_table


settings = get_settings("paths")
log = log_config(__name__)


class Data(AbsDataset):
    def __init__(self, name: str = None, dataset_path: str = None, driver: AbsDriver = None,
                 group_name: str = None):

        if name is None and not isinstance(self.driver, Memory):
            raise Exception("I can't build a dataset without a name, plese add a name to this dataset.")

        self.name = name
        self.header_map = ["author", "description"]
        if driver is None:
            self.driver = Memory()
        else:
            self.driver = driver
        self.group_name = group_name

        if dataset_path is None:
            self.dataset_path = settings["data_path"]
        else:
            self.dataset_path = dataset_path

        self.dtypes = None
        self.hash = None
        self.author = None
        self.description = None
        self.timestamp = None
        self.compressor_params = None

    def set_attrs(self):
        ds_exist = self.driver.exists(self.url)
        if ds_exist and self.driver.mode == "w":
            self.destroy()
            build_path(self.dir_levels())
            self.timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M UTC")
            self.compressor_params = self.driver.compressor_params

    @property
    def author(self):
        return self._get_attr('author')

    @author.setter
    def author(self, value):
        if value is not None:
            self._set_attr('author', value)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def description(self):
        return self._get_attr('description')

    @description.setter
    def description(self, value):
        if value is not None:
            self._set_attr('description', value)

    @property
    def timestamp(self):
        return self._get_attr('timestamp')

    @timestamp.setter
    def timestamp(self, value):
        if value is not None:
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
        if value is not None:
            self._set_attr('compressor_params', json.dumps(value))

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    @property
    @cache
    def data(self):
        groups_data = [(group, self.driver[self.name][group]) for group in self.groups]
        return StructArray(groups_data)

    @data.setter
    @clean_cache
    def data(self, v):
        pass

    def __enter__(self):
        self.driver.enter(self.url)
        return self

    def __exit__(self, exc_type, value, traceback):
        self.driver.exit()
        self.data = None

    def __getitem__(self, key) -> StructArray:
        return self.data[key]

    def __setitem__(self, key, value):
        # print(key, value, self.groups, "DS")
        for group in self.groups:
            self.driver[self.name][group][key] = value

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)

    @property
    def basic_params(self):
        return {"name": self.name, "dataset_path": self.dataset_path,
                "driver": self.driver.module_cls_name(), "group_name": self.group_name}

    def _set_attr(self, name, value):
        if value is not None:
            log.debug("SET attribute {name} {value}".format(name=name, value=value))
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

    def batchs_writer(self, data):
        log.info("Writing with batch size {}".format(getattr(data, 'batch_size', 0)))
        log.debug("WRITING STRUCTURED BATCH")
        for smx in tqdm(data, total=data.num_splits()):
            self.driver[self.name][smx.slice] = smx.batch

    def destroy(self):
        self.driver.destroy(scope=self.name)
        meta_url = self.metadata_url()
        if meta_url is not None:
            rm(meta_url)
            log.debug("METADATA DESTROYED {}".format(meta_url))

    def dir_levels(self) -> list:
        if self.group_name is None:
            return [self.dataset_path, self.driver.cls_name()]
        else:
            return [self.dataset_path, self.driver.cls_name(), self.group_name]

    @property
    def url(self) -> str:
        """
        return the path where is saved the dataset
        """
        filename = "{}.{}".format(self.name, self.driver.ext)
        dir_levels = self.dir_levels() + [filename]
        return os.path.join(*dir_levels)

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
        if len(value) == len(self.dtypes):
            dtypes = [(col, dtypes[1]) for col, dtypes in zip(value, self.dtypes)]
        else:
            raise Exception
        self.dtypes = dtypes

    @property
    def dtypes(self) -> list:
        return self.driver.dtypes(self.name)

    @dtypes.setter
    def dtypes(self, value):
        if value is not None:
            self.driver.set_schema(self.name, value)

    def info(self):
        print('       ')
        print('Dataset NAME: {}'.format(self.name))
        print('Author: {}'.format(self.author))
        print('Hash: {}'.format(self.hash))
        print('Description: {}'.format(self.description))
        print('       ')
        headers = ["Group", "Shape"]
        table = []
        for group, shape in self.shape.items():
            table.append([group, shape])
        print(order_table(headers, table, "Group"))

    def metadata(self) -> dict:
        meta_dict = dict()
        meta_dict["hash"] = self.hash
        meta_dict["dir_levels"] = self.dir_levels()
        meta_dict["driver"] = self.driver.module_cls_name()
        meta_dict["name"] = self.name
        meta_dict["size"] = get_dir_file_size(self.url)
        meta_dict["timestamp"] = self.timestamp
        meta_dict["author"] = self.author
        meta_dict["description"] = self.description if self.description is None else self.description[:100]
        return meta_dict

    def metadata_url(self) -> str:
        metadata = self.metadata()
        if metadata["hash"] is not None:
            pattern = "\$.+\$"
            hash_name = re.sub(pattern, "", metadata["hash"])
            filename = "{}.json".format(hash_name)
            return os.path.join(self.dataset_path, 'metadata', filename)

    def write_metadata(self):
        if self.driver.persistent is True:
            build_path([self.dataset_path, "metadata"])
            path = self.metadata_url()
            if path is not None:
                metadata = self.metadata()
                with open(path, "w") as f:
                    json.dump(metadata, f)

    def calc_hash(self, with_hash: str = 'sha1', batch_size: int = 1080) -> str:
        hash_obj = Hash(hash_fn=with_hash)
        header = [getattr(self, attr) for attr in self.header_map]
        header = [attr for attr in header if attr is not None]
        hash_obj.hash.update("".join(header).encode("utf-8"))
        for group in self.groups:
            it = Iterator(self.data[group]).batchs(batch_size=batch_size, batch_type="array")
            hash_obj.update(it)
        return str(hash_obj)

    def from_data(self, data, batch_size: int = 258, with_hash: str = "sha1"):
        self.set_attrs()
        if isinstance(data, da.Array):
            data = Array.from_da(data)
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
                        raise NotImplementedError("Mixed content is not supported.")
            if len(str_arrays) > 0:
                data = sum(str_arrays)
            data = Iterator(data).batchs(batch_size=batch_size, batch_type="structured")
        elif not isinstance(data, BaseIterator):
            data = Iterator(data).batchs(batch_size=batch_size, batch_type="structured")
        self.dtypes = data.dtypes
        self.driver.set_data_shape(data.shape)
        if isinstance(data, BatchIterator) or isinstance(data, Iterator):
            self.batchs_writer(data)
        else:
            data.store(self)

        if with_hash is not None:
            c_hash = self.calc_hash(with_hash=with_hash)
        else:
            c_hash = None
        self.hash = c_hash
        self.write_metadata()

    def to_df(self) -> pd.DataFrame:
        return self.data.to_df()

    def to_ndarray(self, dtype=None) -> np.ndarray:
        return self.data.to_ndarray(dtype=dtype)

    def to_xrds(self) -> xr.Dataset:
        return self.data.to_xrds()

    def to_libsvm(self, target, save_to=None):
        """
        tranforms the dataset into libsvm format
        """
        from ml.utils.seq import libsvm_row
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        target_t = le.fit_transform(self.driver[self.name][target])
        groups = [group for group in self.groups if group != target]
        with open(save_to, 'w') as f:
            for row in libsvm_row(target_t, self.data[groups].to_ndarray()):
                f.write(" ".join(row))
                f.write("\n")

    def stadistics(self):
        from collections import defaultdict
        from ml.utils.numeric_functions import unique_size, data_type
        from ml.utils.numeric_functions import missing, zeros
        from ml.fmtypes import fmtypes_map
        import dask.array as da
        import dask

        headers = ["missing", "mean", "std dev", "zeros", "min", "25%", "50%", "75%", "max", "type", "unique"]
        for group in ["x"]:#self.groups:
        #    for x in Iterator(self.data[group]).batchs(batch_size=20, batch_type='array'):
        #        print(x)
            shape = self.data[group].shape.to_tuple()
            if len(shape) > 1:
                chunks = (100, shape[1])
            else:
                chunks = (100,)
            array = da.from_array(self.data[group], chunks=chunks)
            mean = array.mean()
            std = array.std()
            min = array.min()
            max = array.max()
            #percentile = da.percentile(array, 4)
            unique = da.unique(array)
            print(dask.compute([mean, std, min, max]))
            break
        #for x in Iterator(self.data).batchs(batch_size=20, batch_type='array'):
        #     print(x)
        #table = []
        #li = self.labels_info()
        #feature_column = defaultdict(dict)
        #for label in li:
        #    mask = (self.labels[:] == label)
        #    data = self.data[:][mask]
        #    for i, column in enumerate(data.T):
        #        percentile = np.nanpercentile(column, [0, 25, 50, 75, 100])
        #        values = [
        #            "{0:.{1}f}%".format(missing(column), 2),
        #            np.nanmean(column),
        #            np.nanstd(column),
        #            "{0:.{1}f}%".format(zeros(column), 2),
        #            ]
        #        values.extend(percentile)
        #        feature_column[i][label] = values

        #data = self.data
        #fmtypes = self.fmtypes
        #for feature, rows in feature_column.items():
        #    column = data[:, feature]
        #    usize = unique_size(column)
        #    if fmtypes is None or len(fmtypes) != self.num_features():
        #        data_t = data_type(usize, column.size).name
        #    else:
        #        data_t = fmtypes_map[fmtypes[feature]]

        #    for label, row in rows.items():
        #        table.append([feature, label] + row + [data_t, str(usize)])
        #        feature = "-"
        #        data_t = "-"
        #        usize = "-"

        #return tabulate(table, headers)

