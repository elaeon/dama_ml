import datetime
import os
import json
import numpy as np
import pandas as pd
import dask.array as da

from tqdm import tqdm
from ml.abc.data import AbsData
from ml.data.it import Iterator, BaseIterator, BatchIterator
from ml.utils.files import build_path
from ml.utils.core import Hash, Login, Metadata, Chunks
from ml.abc.driver import AbsDriver
from ml.data.drivers.core import Memory
from ml.abc.group import AbsGroup
from ml.utils.logger import log_config
from ml.utils.config import get_settings
from ml.utils.decorators import cache, clean_cache
from ml.utils.files import get_dir_file_size
from ml.utils.order import order_table
from ml.data.groups.core import DaGroup


settings = get_settings("paths")
log = log_config(__name__)


class Data(AbsData):
    def __init__(self, name: str = None, dataset_path: str = None, driver: AbsDriver = None,
                 group_name: str = None, chunks=None):

        if driver is None:
            self.driver = Memory()
        else:
            self.driver = driver

        if name is None and not isinstance(self.driver, Memory):
            raise Exception("I can't build a dataset without a name, plese add a name to this dataset.")

        if dataset_path is None:
            self.dataset_path = settings["data_path"]
        else:
            self.dataset_path = dataset_path

        self.name = name
        self.header_map = ["author", "description"]
        self.group_name = group_name
        self.dtypes = None
        self.hash = None
        self.author = None
        self.description = None
        self.timestamp = None
        self.compressor_params = None
        self.chunks = chunks
        if self.driver.login is None:
            self.driver.login = Login(url=self.url)
        else:
            self.driver.login.url = self.url

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
    def data(self) -> AbsGroup:
        return self.driver.data(chunks=self.chunks)

    @data.setter
    @clean_cache
    def data(self, v):
        pass

    def clean_data_cache(self):
        self.data = None

    def __enter__(self):
        build_path(self.dir_levels())
        self.driver.enter()

        if self.driver.data_tag is None:
            self.driver.data_tag = self.name

        if self.driver.mode in ["w", "a", "r+"]:
            if len(self.driver.compressor_params) > 0:
                self.compressor_params = self.driver.compressor_params
        return self

    def __exit__(self, exc_type, value, traceback):
        self.driver.exit()
        self.data = None

    def __getitem__(self, key):
        return self.data[key]

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
        batch_size = getattr(data, 'batch_size', 0)
        log.info("Writing with chunks {}".format(batch_size))
        if batch_size > 0:
            absgroup = self.driver.absgroup()
            for smx in tqdm(data, total=data.num_splits()):
                #self.driver.data[smx.slice] = smx
                #absgroup.conn[smx.slice] = smx
                #self.data[smx.slice] = smx
                absgroup.set(smx.slice, smx)
        else:
            for i, smx in tqdm(enumerate(data), total=data.num_splits()):
                for j, group in enumerate(self.groups):
                    self.data[group][i] = smx[j]

    def destroy(self):
        hash = self.hash
        self.driver.destroy()
        login = Login(url=self.metadata_url(), table="metadata")
        metadata = Metadata(login)
        metadata.remove_data(hash)

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
    def groups(self) -> tuple:
        return self.data.groups

    @groups.setter
    def groups(self, value) -> None:
        raise NotImplementedError

    @property
    def dtypes(self) -> np.dtype:
        return self.data.dtypes

    @dtypes.setter
    def dtypes(self, value):
        if value is not None:
            self.driver.set_schema(value)

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
        return os.path.join(settings["metadata_path"], "metadata.sqlite3")

    def metadata_to_json(self, f):
        metadata = self.metadata()
        json.dump(metadata, f)

    def write_metadata(self):
        if self.driver.persistent is True:
            build_path([settings["metadata_path"]])
            login = Login(url=self.metadata_url(), table="metadata")
            metadata = Metadata(login, self.metadata())
            dtypes = np.dtype([("hash", object), ("name", object), ("author", object),
                              ("description", object), ("size", int), ("driver", object),
                              ("dir_levels", object), ("timestamp", np.dtype("datetime64[ns]"))])
            timestamp = metadata["timestamp"]
            metadata["timestamp"] = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M UTC')
            dir_levels = metadata["dir_levels"]
            metadata["dir_levels"] = os.path.join(*dir_levels)
            metadata.build_schema(dtypes, unique_key="hash")
            metadata.insert_data()

    def calc_hash(self, with_hash: str = 'sha1', batch_size: int = 1080) -> str:
        hash_obj = Hash(hash_fn=with_hash)
        header = [getattr(self, attr) for attr in self.header_map]
        header = [attr for attr in header if attr is not None]
        hash_obj.hash.update("".join(header).encode("utf-8"))
        for group in self.groups:
            it = Iterator(self.data[group]).batchs(chunks=self.chunks)
            hash_obj.update(it.only_data())
        return str(hash_obj)

    def from_data(self, data, chunks=None, with_hash: str = "sha1"):
        if isinstance(data, da.Array):
            data = DaGroup.from_da(data)
        elif isinstance(data, Iterator):
            self.chunks = Chunks.build_from(chunks, data.groups)
            data = data.batchs(chunks=self.chunks)
            self.chunks = data.chunks
        elif isinstance(data, dict):
            self.chunks = Chunks.build_from(chunks, tuple(data.keys()))
            data = DaGroup(data, chunks=self.chunks)
        elif isinstance(data, DaGroup) or type(data) == DaGroup:
            pass
        elif not isinstance(data, BaseIterator):
            data = Iterator(data)
            self.chunks = data.shape.to_chunks(chunks)
            data = data.batchs(chunks=self.chunks)
            self.chunks = data.chunks
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
        self.timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M UTC")
        self.write_metadata()

    def to_df(self) -> pd.DataFrame:
        return self.data.to_df()

    def to_ndarray(self, dtype=None) -> np.ndarray:
        return self.data.to_ndarray(dtype=dtype)

    def to_libsvm(self, target, save_to=None):
        """
        tranforms the dataset into libsvm format
        """
        from ml.utils.seq import libsvm_row
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        target_t = le.fit_transform(self.driver.data[target].to_ndarray())
        groups = [group for group in self.groups if group != target]
        with open(save_to, 'w') as f:
            for row in libsvm_row(target_t, self.data[groups].to_ndarray()):
                f.write(" ".join(row))
                f.write("\n")

    def concat(self, datasets: tuple, axis=0):
        da_groups = []
        for ds in datasets:
            da_groups.append(ds.data)
        da_group = DaGroup.concat(da_groups, axis=axis)
        self.from_data(da_group)

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
