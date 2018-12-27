import os
from abc import ABC, abstractmethod
from ml.data.ds import Data
from ml.data.it import Iterator
from ml.utils.files import check_or_create_path_dir
from ml.measures import ListMeasure
from ml.utils.logger import log_config
from pydoc import locate
from ml.utils.config import get_settings
import json

settings = get_settings("ml")
log = log_config(__name__)


class MLModel:
    def __init__(self, fit_fn=None, predictors=None, load_fn=None, save_fn=None,
                 input_transform=None, model=None):
        self.fit_fn = fit_fn
        self.predictors = predictors
        self.load_fn = load_fn
        self.save_fn = save_fn
        if input_transform is None:
            self.input_transform = lambda x: x
        else:
            self.input_transform = input_transform
        self.model = model

    def fit(self, *args, **kwargs):
        return self.fit_fn(*args, **kwargs)

    def predict(self, data, output_format_fn=None, output=None, batch_size: int = 258) -> Iterator:
        data = self.input_transform(data)
        if hasattr(data, '__iter__'):
            #if data:
            #    for chunk in data:
            #        yield self.predictors(self.transform_data(chunk))
            #else:
            def _it():
                for row in data:  # fixme add batch_size
                    predict = self.predictors(row.to_ndarray().reshape(1, -1))
                    #if len(predict.shape) > 1:
                    #    yield output_format_fn(predict[0], output=output)
                    #else:
                    yield output_format_fn(predict, output=output)[0]
            return Iterator(_it()).batchs(batch_size=batch_size)
        else:
            return Iterator(output_format_fn(self.predictors(data), output=output)).batchs(batch_size=batch_size)

    def load(self, path):
        return self.load_fn(path)

    def save(self, path):
        return self.save_fn(path)


class Metadata(object):
    def __init__(self):
        self.model_name = None
        self.group_name = None
        self.model_version = None
        self.base_path = None
        self.path_metamodel = None
        self.path_model_version = None
        self.path_metamodel_version = None
        self.metaext = "json"

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
            log.info(e)
            return {}
        except Exception as e:
            log.error("{} {}".format(e, path))

    @staticmethod
    def get_metadata(path_metamodel, path_metamodel_version=None):
        if path_metamodel is not None:
            metadata = {"model": Metadata.load_json(path_metamodel)}
            if path_metamodel_version is not None:
                metadata["train"] = Metadata.load_json(path_metamodel_version)
            else:
                metadata["train"] = {}
            return metadata

    @staticmethod
    def make_model_file(name, path, classname, metaext):
        check_point = check_or_create_path_dir(path, classname)
        destination = check_or_create_path_dir(check_point, name)
        filename = os.path.join(destination, "meta")
        return "{}.{}".format(filename, metaext)

    @staticmethod
    def make_model_version_file(name, path, classname, ext, model_version):
        model_name_v = "version.{}".format(model_version)
        check_point = check_or_create_path_dir(path, classname)
        destination = check_or_create_path_dir(check_point, name)
        model = check_or_create_path_dir(destination, model_name_v)
        filename = os.path.join(model, "meta")
        return "{}.{}".format(filename, ext)

    def print_meta(self):
        print(Metadata.get_metadata(self.path_metamodel, self.path_metamodel_version))

    def destroy(self):
        """remove the dataset associated to the model and his checkpoints"""
        from ml.utils.files import rm
        if self.path_metamodel is not None:
            rm(self.path_metamodel)
        if self.path_metamodel_version is not None:
            rm(self.path_model_version)
            rm(self.path_metamodel_version)
        if hasattr(self, 'ds'):
            self.ds.destroy()

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)


class BaseModel(Metadata, ABC):
    def __init__(self, metrics=None):
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
        self.target = None
        super(BaseModel, self).__init__()

    @abstractmethod
    def scores(self, measures=None, chunks_size=2000):
        return NotImplemented

    @classmethod
    @abstractmethod
    def load(cls, model_name: str, model_version: str, group_name: str = None, path: str = None):
        return NotImplemented

    @abstractmethod
    def output_format(self, prediction, output=None):
        return NotImplemented

    @abstractmethod
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None):
        return NotImplemented

    @abstractmethod
    def load_fn(self):
        return NotImplemented

    def predict(self, data, output=None, batch_size: int = 258):
        return self.model.predict(data, output_format_fn=self.output_format, output=output, batch_size=batch_size)

    def metadata_model(self):
        with self.ds:
            return {
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "meta_path": self.path_metamodel,
                "base_path": self.base_path,
                "ds_basic_params": self.ds.basic_params,
                "hash": self.ds.hash,
                "data_groups": self.data_groups,
                "versions": []
            }

    def metadata_train(self):
        return {
            "model_version": self.model_version,
            "params": self.model_params,
            "num_steps": self.num_steps,
            "score": self.scores(measures=self.metrics).measures_to_dict(),
            "meta_path": self.path_metamodel_version,
            "model_path": self.path_model_version
        }

    def get_dataset(self):
        log.debug("LOADING DS FOR MODEL: {} {} {} {}".format(self.cls_name(), self.model_name,
                                                             self.model_version, self.base_path))
        meta = Metadata.get_metadata(self.path_metamodel).get("model", {})
        driver = locate(meta["ds_basic_params"]["driver"])
        dataset = Data(name=meta["ds_basic_params"]["name"], group_name=meta["ds_basic_params"]["group_name"],
                       dataset_path=meta["ds_basic_params"]["dataset_path"],
                       driver=driver())
        with dataset:
            if meta.get('hash', None) != dataset.hash:
                log.info("The dataset hash is not equal to the model '{}'".format(
                    self.__class__.__name__))
        return dataset

    def preload_model(self):
        self.model = MLModel(fit_fn=None,  predictors=None, load_fn=self.load_fn, save_fn=None)

    def save(self, name, path: str = None, model_version="1"):
        self.model_version = model_version
        self.model_name = name
        if path is None:
            self.base_path = settings["checkpoints_path"]
        else:
            self.base_path = path
        self.path_metamodel = Metadata.make_model_file(name, self.base_path, self.cls_name(), self.metaext)
        self.path_metamodel_version = self.make_model_version_file(name, path, self.cls_name(), self.metaext,
                                                                   self.model_version)
        self.path_model_version = Metadata.make_model_version_file(name, path, self.cls_name(), self.ext,
                                                                   model_version=model_version)
        log.debug("SAVING model")
        self.model.save(self.path_model_version)
        log.debug("SAVING model metadata")
        metadata_tmp = Metadata.get_metadata(self.path_metamodel)
        metadata_model = self.metadata_model()
        if len(metadata_model["versions"]) == 0:
            metadata_model["versions"] = self.model_version
        if self.model_version not in metadata_model["versions"]:
            metadata_tmp["model"]["versions"].append(model_version)
            metadata_model["versions"] = metadata_tmp["model"]["versions"]
        Metadata.save_json(self.path_metamodel, metadata_model)
        metadata_train = self.metadata_train()
        Metadata.save_json(self.path_metamodel_version, metadata_train)

    def load_model(self):
        self.preload_model()
        if self.path_model_version is not None:
            self.model.load(self.path_model_version)

    def load_metadata(self, path_metamodel, path_metamodel_version):
        metadata = Metadata.get_metadata(path_metamodel, path_metamodel_version)
        self.group_name = metadata["model"]["group_name"]
        self.model_name = metadata["model"]["model_name"]
        self.model_version = metadata["train"]["model_version"]
        self.model_params = metadata["train"]["params"]
        self.path_metamodel_version = metadata["train"]["meta_path"]
        self.path_metamodel = metadata["model"]["meta_path"]
        self.path_model_version = metadata["train"]["model_path"]
        self.num_steps = metadata["train"]["num_steps"]
        self.base_path = metadata["model"]["base_path"]

    @abstractmethod
    def train(self, ds: Data, batch_size: int = 0, num_steps: int = 0, n_splits=None, obj_fn=None,
              model_params: dict = None, data_train_group="train_x", target_train_group='train_y',
              data_test_group="test_x", target_test_group='test_y',
              data_validation_group="validation_x", target_validation_group="validation_y"):
        return NotImplemented

    def scores2table(self):
        meta = Metadata.get_metadata(self.path_metamodel, self.path_metamodel_version)
        try:
            scores = meta["train"]["score"]
        except KeyError:
            return
        else:
            return ListMeasure.dict_to_measures(scores)


class SupervicedModel(BaseModel):
    def __init__(self, metrics=None):
        super(SupervicedModel, self).__init__(metrics=metrics)

    @classmethod
    def load(cls, model_name: str, model_version: str, group_name: str = None, path: str = None):
        model = cls()
        path_metamodel = Metadata.make_model_file(model_name, path, model.cls_name(), model.metaext)
        path_metamodel_version = Metadata.make_model_version_file(model_name, path, model.cls_name(),
                                                                  model.metaext, model_version=model_version)
        model.load_metadata(path_metamodel, path_metamodel_version)
        model.ds = model.get_dataset()
        model.load_model()
        return model

    def train(self, ds: Data, batch_size: int = 0, num_steps: int = 0, n_splits=None, obj_fn=None,
              model_params: dict = None, data_train_group="train_x", target_train_group='train_y',
              data_test_group="test_x", target_test_group='test_y',
              data_validation_group="validation_x", target_validation_group="validation_y"):
        self.ds = ds
        log.info("Training")
        self.model_params = model_params
        self.num_steps = num_steps
        self.data_groups = {
            "data_train_group": data_train_group, "target_train_group": target_train_group,
            "data_test_group": data_test_group, "target_test_group": target_test_group,
            "data_validation_group": data_validation_group, "target_validation_group": target_validation_group
        }
        self.model = self.prepare_model(obj_fn=obj_fn, num_steps=num_steps, model_params=model_params)


class UnsupervisedModel(BaseModel):
    def train(self, ds: Data, batch_size: int = 0, num_steps: int = 0, n_splits=None, obj_fn=None,
              model_params: dict = None, data_train_group="train_x", target_train_group='train_y',
              data_test_group="test_x", target_test_group='test_y', num_epochs: int = 0,
              data_validation_group="validation_x", target_validation_group="validation_y"):
        log.info("Training")
        self.model = self.prepare_model(obj_fn=None, num_steps=num_steps, num_epochs=num_epochs,
                                        model_params=model_params)
