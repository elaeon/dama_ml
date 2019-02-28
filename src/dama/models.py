import os
import numpy as np
from abc import ABC, abstractmethod
from dama.data.ds import Data
from dama.data.it import Iterator, BatchIterator
from dama.utils.files import check_or_create_path_dir
from dama.measures import ListMeasure
from dama.utils.logger import log_config
from dama.utils.config import get_settings
from dama.groups.core import DaGroup
from dama.drivers.sqlite import Sqlite
from dama.utils.core import Login, Metadata
from dama.utils.files import rm
import json

settings = get_settings("paths")
settings.update(get_settings("vars"))
log = log_config(__name__)


class MLModel:
    def __init__(self, fit_fn=None, predictors=None, load_fn=None, save_fn=None,
                 input_transform=None, model=None, to_json_fn=None):
        self.fit_fn = fit_fn
        self.predictors = predictors
        self.load_fn = load_fn
        self.save_fn = save_fn
        self.to_json_fn = to_json_fn
        if input_transform is None:
            self.input_transform = lambda x: x
        else:
            self.input_transform = input_transform
        self.model = model

    def fit(self, *args, **kwargs):
        return self.fit_fn(*args, **kwargs)

    def predict(self, data: DaGroup, output_format_fn=None, output=None, batch_size: int = 258) -> BatchIterator:
        data = self.input_transform(data)
        #if hasattr(data, '__iter__'):
            #if data:
            #    for chunk in data:
            #        yield self.predictors(self.transform_data(chunk))
            #else:
        def _it():
            for row in data:  # fixme add batch_size
                batch = row.to_ndarray().reshape(1, -1)
                predict = self.predictors(batch)
                yield output_format_fn(predict, output=output)[0]
        return Iterator(_it(), length=len(data)).batchs(chunks=None)
        #else:
        #    return Iterator(output_format_fn(self.predictors(data), output=output)).batchs(chunks=(batch_size, ))

    def load(self, path):
        return self.load_fn(path)

    def save(self, path):
        return self.save_fn(path)

    def to_json(self) -> dict:
        if self.to_json_fn is not None:
            return self.to_json_fn()
        return {}


class MetadataX(object):
    metaext = "json"

    @staticmethod
    def save_json(file_path, data):
        with open(file_path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_json(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except IOError as e:
            log.error(e)
            return {}
        except Exception as e:
            log.error("{} {}".format(e, path))

    @staticmethod
    def get_metadata(path_metadata_version: str = None):
        metadata = {}
        if path_metadata_version is not None:
            metadata["train"] = MetadataX.load_json(path_metadata_version)
        else:
            metadata["train"] = {}
        return metadata

    @staticmethod
    def make_model_version_file(name, path, classname, ext, model_version):
        model_name_v = "version.{}".format(model_version)
        check_point = check_or_create_path_dir(path, classname)
        destination = check_or_create_path_dir(check_point, name)
        model = check_or_create_path_dir(destination, model_name_v)
        filename = os.path.join(model, "meta")
        return "{}.{}".format(filename, ext)


class BaseModel(MetadataX, ABC):
    def __init__(self, metrics=None, metadata_path=None):
        self.model_name = None
        self.group_name = None
        self.model_version = None
        self.base_path = None
        self.path_model_version = None
        self.path_metadata_version = None
        self.model = None
        self.ext = "ckpt.pkl"
        self.metrics = metrics
        self.model_params = None
        self.num_steps = None
        self.model_version = None
        self.ds = None
        self.data_groups = None
        self.model_params = None
        self.num_steps = None
        self.batch_size = None
        if metadata_path is None:
            self.metadata_path = settings["metadata_path"]
        else:
            self.metadata_path = metadata_path
        super(BaseModel, self).__init__()

    @abstractmethod
    def scores(self, measures=None, batch_size=2000):
        return NotImplemented

    @abstractmethod
    def output_format(self, prediction, output=None):
        return NotImplemented

    @abstractmethod
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None) -> MLModel:
        return NotImplemented

    @abstractmethod
    def load_fn(self, path):
        return NotImplemented

    def predict(self, data, output=None, batch_size: int = 258):
        return self.model.predict(data, output_format_fn=self.output_format, output=output, batch_size=batch_size)

    def metadata_model(self):
        return {
            "group_name": self.group_name,
            "model_module": self.module_cls_name(),
            "name": self.model_name,
            "base_path": self.base_path,
            "hash": self.ds.hash,
            "from_ds": self.ds.from_ds_hash
        }

    def metadata_train(self):
        return {
            "model_version": self.model_version,
            "hyperparams": self.model_params,
            "num_steps": self.num_steps,
            "score": self.scores(measures=self.metrics).measures_to_dict(),
            "batch_size": self.batch_size,
            "model_json": self.model.to_json(),
            "data_groups": self.data_groups,
        }

    def __enter__(self):
        self.ds = self.get_dataset()
        self.ds.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ds.close()

    def get_dataset(self) -> Data:
        log.debug("LOADING DS FOR MODEL: {} {} {} {}".format(self.cls_name(), self.model_name,
                                                             self.model_version, self.base_path))
        group_name = "s/n" if self.group_name is None else self.group_name
        with Metadata(Sqlite(login=Login(table=settings["model_tag"]), path=self.metadata_path)) as metadata:
            query = "SELECT hash FROM {} WHERE name=? AND version=? AND model_module=? AND group_name=? AND base_path=?".format(
                settings["model_tag"])
            data_hash = metadata.query(query,
                                       (self.model_name, self.model_version, self.module_cls_name(),
                                        group_name, self.base_path))
        if len(data_hash) > 0:
            driver = Sqlite(login=Login(table=settings["data_tag"]), path=self.metadata_path)
            with Data.load(data_hash[0][0], metadata_driver=driver) as dataset:
                dataset.auto_chunks = True
            return dataset

    def preload_model(self):
        self.model = MLModel(fit_fn=None,  predictors=None, load_fn=self.load_fn, save_fn=None)

    def write_metadata(self):
        metadata_driver = Sqlite(login=Login(table=settings["model_tag"]), path=self.metadata_path)
        metadata = self.metadata_model()
        metadata_train = self.metadata_train()
        metadata["version"] = self.model_version
        metadata["model_path"] = self.path_model_version
        metadata["metadata_path_train"] = self.path_metadata_version

        with Metadata(metadata_driver, metadata) as metadata:
            dtypes = np.dtype([("hash", object), ("name", object), ("model_path", object), ("group_name", object),
                               ("is_valid", bool), ("version", int), ("model_module", object), ("score_name", object),
                               ("score", float), ("metadata_path_train", object), ("base_path", object),
                               ("from_ds", object)])
            metadata["is_valid"] = True
            metadata["group_name"] = "s/n" if self.group_name is None else self.group_name
            keys = ["base_path", "name", "group_name", "version", "model_module", "score_name"]
            metadata.set_schema(dtypes, unique_key=[keys])
            if len(metadata_train["score"]) == 0:
                metadata["score_name"] = "s/n"
                metadata["score"] = 0
                metadata.insert_update_data(keys=keys)
            else:
                for score_name in metadata_train["score"].keys():
                    if score_name != "":
                        metadata["score_name"] = score_name
                        metadata["score"] = metadata_train["score"][score_name]["values"][0]
                        metadata.insert_update_data(keys=keys)

    def save(self, name, path: str = None, model_version="1"):
        self.model_version = model_version
        self.model_name = name
        if path is None:
            self.base_path = settings["models_path"]
        else:
            self.base_path = path
        self.path_metadata_version = MetadataX.make_model_version_file(name, self.base_path, self.cls_name(),
                                                                       self.metaext, self.model_version)
        self.path_model_version = MetadataX.make_model_version_file(name, self.base_path, self.cls_name(), self.ext,
                                                                   model_version=model_version)
        log.debug("SAVING model's data")
        self.model.save(self.path_model_version)
        log.debug("SAVING json metadata train info")
        metadata_train = self.metadata_train()
        MetadataX.save_json(self.path_metadata_version, metadata_train)
        self.write_metadata()

    def load_model(self):
        self.preload_model()
        if self.path_model_version is not None:
            self.model.load(self.path_model_version)

    def load_metadata(self, path_metadata_version):
        metadata = MetadataX.get_metadata(path_metadata_version)
        self.model_version = metadata["train"]["model_version"]
        self.model_params = metadata["train"]["hyperparams"]
        self.num_steps = metadata["train"]["num_steps"]
        self.batch_size = metadata["train"]["batch_size"]
        self.data_groups = metadata["train"]["data_groups"]

    @abstractmethod
    def train(self, ds: Data, batch_size: int = 0, num_steps: int = 0, n_splits=None, obj_fn=None,
              model_params: dict = None, data_train_group="train_x", target_train_group='train_y',
              data_test_group="test_x", target_test_group='test_y', data_validation_group="validation_x",
              target_validation_group="validation_y"):
        return NotImplemented

    def scores2table(self):
        meta = MetadataX.get_metadata(self.path_metadata_version)
        try:
            scores = meta["train"]["score"]
        except KeyError:
            return
        else:
            return ListMeasure.dict_to_measures(scores)

    @classmethod
    def load(cls, model_name: str, model_version: str, group_name: str = None, path: str = None,
             metadata_path: str = None):
        model = cls(metadata_path=metadata_path)
        model.model_name = model_name
        model.model_version = model_version
        if group_name is None:
            group_name = "s/n"
        model.group_name = group_name
        model.base_path = path
        path_metadata_version = MetadataX.make_model_version_file(model_name, path, model.cls_name(),
                                                                  model.metaext, model_version=model_version)
        model.path_metadata_version = path_metadata_version
        model.path_model_version = MetadataX.make_model_version_file(model_name, path, model.cls_name(),
                                                                     model.ext, model_version=model_version)
        model.load_metadata(path_metadata_version)
        model.load_model()
        return model

    def destroy(self):
        if self.path_metadata_version is not None:
            rm(self.path_model_version)
            rm(self.path_metadata_version)
        if hasattr(self, 'ds'):
            self.ds.destroy()

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)


class SupervicedModel(BaseModel):
    def __init__(self, metrics=None, metadata_path=None):
        super(SupervicedModel, self).__init__(metrics=metrics, metadata_path=metadata_path)

    def train(self, ds: Data, batch_size: int = 0, num_steps: int = 0, n_splits=None, obj_fn=None,
              model_params: dict = None, data_train_group="train_x", target_train_group='train_y',
              data_test_group="test_x", target_test_group='test_y', data_validation_group="validation_x",
              target_validation_group="validation_y"):
        self.ds = ds
        log.info("Training")
        self.model_params = model_params
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.data_groups = {
            "data_train_group": data_train_group, "target_train_group": target_train_group,
            "data_test_group": data_test_group, "target_test_group": target_test_group,
            "data_validation_group": data_validation_group, "target_validation_group": target_validation_group
        }
        self.model = self.prepare_model(obj_fn=obj_fn, num_steps=num_steps, model_params=model_params,
                                        batch_size=batch_size)


class UnsupervisedModel(BaseModel):

    def train(self, ds: Data, batch_size: int = 0, num_steps: int = 0, n_splits=None, obj_fn=None,
              model_params: dict = None, data_train_group="train_x", target_train_group='train_y',
              data_test_group="test_x", target_test_group='test_y', data_validation_group="validation_x",
              target_validation_group="validation_y"):
        self.ds = ds
        log.info("Training")
        self.model_params = model_params
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.data_groups = {
            "data_train_group": data_train_group, "target_train_group": target_train_group,
            "data_test_group": data_test_group, "target_test_group": target_test_group,
            "data_validation_group": data_validation_group, "target_validation_group": target_validation_group
        }
        self.model = self.prepare_model(obj_fn=obj_fn, num_steps=num_steps, model_params=model_params,
                                        batch_size=batch_size)

    def scores(self, measures=None, batch_size: int = 258) -> ListMeasure:
        return ListMeasure()

    def output_format(self, prediction, output=None):
        return prediction
