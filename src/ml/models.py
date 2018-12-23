import os

from ml.data.ds import Data
from ml.data.it import Iterator
from ml.utils.files import check_or_create_path_dir
from ml.measures import ListMeasure
from ml.utils.logger import log_config
from pydoc import locate
from ml.utils.config import get_settings


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


class DataDrive(object):
    def __init__(self, check_point_path=None, model_name=None,
                group_name=None):
        if check_point_path is None:
            self.check_point_path = settings["checkpoints_path"]
        else:
            self.check_point_path = check_point_path
        self.model_name = model_name
        self.group_name = group_name
        self.model_version = None
        self.path_m = None
        self.path_mv = None

    def _metadata(self):
        pass

    def save_meta(self, keys=None):
        from ml.data.ds import save_metadata
        if self.check_point_path is not None and keys is not None:
            metadata_tmp = self.load_meta()
            if "model" in keys:
                self.path_m = self.make_model_file()
                metadata = self._metadata(["model"])
                if len(metadata["model"]["versions"]) == 0:
                    metadata["model"]["versions"] = self.model_version
                if not self.model_version in metadata["model"]["versions"] and\
                    self.model_version is not None:
                    metadata_tmp["model"]["versions"].append(self.model_version)
                    metadata["model"]["versions"] = metadata_tmp["model"]["versions"]
                save_metadata(self.path_m+".xmeta", metadata["model"])
            if "train" in keys:
                metadata = self._metadata(["train"])
                save_metadata(self.path_mv+".xmeta", metadata["train"])

    def load_meta(self):
        from ml.data.ds import load_metadata
        if self.check_point_path is not None:
            metadata = {}
            self.path_m = self.make_model_file()
            metadata["model"] = load_metadata(self.path_m+".xmeta")
            if self.model_version is not None:
                self.path_mv = self.make_model_version_file()
                metadata["train"] = load_metadata(self.path_mv+".xmeta")
            else:
                metadata["train"] = {}
            return metadata

    def make_model_file(self):
        check_point = check_or_create_path_dir(self.check_point_path, self.__class__.__name__)
        destination = check_or_create_path_dir(check_point, self.model_name)
        return os.path.join(destination, "meta")

    def make_model_version_file(self):
        model_name_v = "version.{}".format(self.model_version)
        check_point = check_or_create_path_dir(self.check_point_path, self.__class__.__name__)
        destination = check_or_create_path_dir(check_point, self.model_name)
        model = check_or_create_path_dir(destination, model_name_v)
        return os.path.join(model, "meta")

    def print_meta(self):
        print(self.load_meta())

    def destroy(self):
        """remove the dataset associated to the model and his checkpoints"""
        from ml.utils.files import rm
        if self.path_m is not None:
            rm(self.path_m+".xmeta")
        if self.path_mv is not None:
            rm(self.path_mv+"."+self.ext)
            rm(self.path_mv+".xmeta")
        if hasattr(self, 'ds'):
            self.ds.destroy()

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def exist(self):
        meta = self.load_meta()
        return meta.get('original_dataset_name', "") != ""


class BaseModel(DataDrive):
    def __init__(self, model_name=None, check_point_path=None, 
                group_name=None, metrics=None):
        self.model = None
        self.original_dataset_hash = None
        self.original_dataset_path = None
        self.original_dataset_name = None
        self.test_ds = None
        self.ext = "ckpt.pkl"
        self.metrics = metrics
        self.model_params = None
        self.num_steps = None
        self.model_version = None
        self.target = None

        super(BaseModel, self).__init__(
            check_point_path=check_point_path,
            model_name=model_name,
            group_name=group_name)

    def scores(self, measures=None, chunks_size=2000):
        pass

    def confusion_matrix(self):
        pass

    def only_is(self, op):
        pass

    def reformat_all(self, dataset):
        pass

    def reformat_labels(self, labels):
        pass

    def load(self):
        pass

    def set_dataset(self, dataset, reformat=True):
        pass
            
    @property
    def num_features(self):
        with self.test_ds:
            return self.test_ds.num_features()

    def predict(self, data, output=None, batch_size: int=258):
        return self.model.predict(data, output_format_fn=self.output_format, output=output, batch_size=batch_size)

    def metadata_model(self):
        with self.ds:
            return {
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_path": self.path_m,
                "versions": []
            }

    def metadata_train(self):
        return {
            "model_version": self.model_version,
            "params": self.model_params,
            "num_steps": self.num_steps,
            "score": self.scores(measures=self.metrics).measures_to_dict(),
            "model_path": self.path_mv
        }

    def _metadata(self, keys={}):
        metadata_model_train = {}
        if "model" in keys:
            metadata_model_train["model"] = self.metadata_model()
        if "train" in keys:
            metadata_model_train["train"] = self.metadata_train()
        return metadata_model_train

    def get_dataset(self):
        log.debug("LOADING DS FOR MODEL: {} {} {} {}".format(self.cls_name(), self.model_name, 
            self.model_version, self.check_point_path))
        meta = self.load_meta()["model"]
        driver = locate(meta["ds_basic_params"]["driver"])
        dataset = Data(name=meta["ds_basic_params"]["name"], group_name=meta["ds_basic_params"]["group_name"],
                       dataset_path=meta["ds_basic_params"]["dataset_path"],
                       driver=driver())
        self.group_name = meta.get('group_name', None)
        with dataset:
            if meta.get('hash', None) != dataset.hash:
                log.info("The dataset hash is not equal to the model '{}'".format(
                    self.__class__.__name__))
        return dataset

    def preload_model(self):
        self.model = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_fn,
                            save_fn=None)

    def save(self, model_version="1"):
        self.model_version = model_version
        if self.check_point_path is not None:
            self.path_mv = self.make_model_version_file()
            self.model.save('{}.{}'.format(self.path_mv, self.ext))
            self.save_meta(keys=["model", "train"])

    def has_model_file(self):
        import os
        path = self.get_model_path()
        return os.path.exists('{}.{}'.format(path, self.ext))

    def load_model(self):        
        self.preload_model()
        if self.check_point_path is not None and self.model_version is not None:
            self.path_mv = self.make_model_version_file()
            self.model.load('{}.{}'.format(self.path_mv, self.ext))

    def train(self, batch_size=0, num_steps=0, n_splits=None, obj_fn=None, model_params={}):
        log.info("Training")
        self.model_params = model_params
        self.num_steps = num_steps
        self.model = self.prepare_model(obj_fn=obj_fn, num_steps=num_steps, **model_params)

    def scores2table(self):
        meta = self.load_meta()
        try:
            scores = meta["train"]["score"]
        except KeyError:
            return
        else:
            return ListMeasure.dict_to_measures(scores)

    def load_original_ds(self):
        return Data.original_ds(self.original_dataset_name, self.original_dataset_path)


class SupervicedModel(BaseModel):
    def __init__(self, model_name=None, check_point_path=None, group_name=None, 
            metrics=None):
        self.ds = None
        self.data_groups = None
        super(SupervicedModel, self).__init__(
            check_point_path=check_point_path,
            model_name=model_name,
            group_name=group_name,
            metrics=metrics)

    def load(self, model_version):
        pass

    def metadata_model(self):
        with self.ds:
            return {
                "ds_basic_params": self.ds.basic_params,
                "hash": self.ds.hash,
                "data_groups": self.data_groups,
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_path": self.path_m,
                "versions": []
            }

    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, 
                    model_params={}):
        from sklearn.model_selection import StratifiedKFold
        model = self.prepare_model_k(obj_fn=obj_fn)
        cv = StratifiedKFold(n_splits=n_splits)
        with self.train_ds:
            data = self.train_ds.data[:]
            labels = self.train_ds.labels[:]
            for k, (train, test) in enumerate(cv.split(data, labels), 1):
                model.fit(data[train], labels[train])
                print("fold ", k)
        return model

    def train(self, ds: Data, batch_size=0, num_steps=0, n_splits=None, obj_fn=None, model_params={},
              data_train_group="train_x", target_train_group='train_y',
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
        if n_splits is not None:
            self.model = self.train_kfolds(batch_size=batch_size, num_steps=num_steps,
                            n_splits=n_splits, obj_fn=obj_fn, model_params=model_params)
        else:
            self.model = self.prepare_model(obj_fn=obj_fn, num_steps=num_steps, **model_params)


class UnsupervisedModel(BaseModel):
    def train(self, batch_size=0, num_steps=0, num_epochs=0, model_params={}):
        log.info("Training")
        self.model = self.prepare_model(obj_fn=obj_fn, num_steps=num_steps, 
            num_epochs=num_epochs, **model_params)
