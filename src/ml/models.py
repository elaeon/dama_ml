from ml.utils.config import get_settings
import os
import logging

from ml.data.ds import Data
from ml.data.it import Iterator
from ml.utils.files import check_or_create_path_dir
from ml.measures import ListMeasure


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


class MLModel:
    def __init__(self, fit_fn=None, predictors=None, load_fn=None, save_fn=None,
                transform_data=None, model=None):
        self.fit_fn = fit_fn
        self.predictors = predictors
        self.load_fn = load_fn
        self.save_fn = save_fn
        if transform_data is None:
            self.transform_data = lambda x: x
        else:
            self.transform_data = transform_data
        self.model = model

    def fit(self, *args, **kwargs):
        return self.fit_fn(*args, **kwargs)

    def predict(self, data):
        #if data:
        #    for chunk in data:
        #        yield self.predictors(self.transform_data(chunk))
        #else:
        for row in data:
            predict = self.predictors(row.to_ndarray().reshape(1, -1))
            if len(predict.shape) > 1:
                yield predict[0]
            else:
                yield predict

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
        if hasattr(self, 'dataset'):
            self.dataset.destroy()
        if hasattr(self, 'test_ds'):
            self.test_ds.destroy()
        if hasattr(self, 'train_ds'):
            self.train_ds.destroy()
        if hasattr(self, 'validation_ds'):
            self.validation_ds.destroy()

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
        plain_prediction = self.model.predict(data)
        prediction = self.output_format(plain_prediction, output=output)
        return Iterator(prediction).batchs(batch_size=batch_size)

    def metadata_model(self):
        with self.test_ds:
            return {
                "test_ds_path": self.test_ds.dataset_path,
                "test_ds_name": self.test_ds.name,
                "hash": self.test_ds.hash,
                "original_dataset_md5": self.original_dataset_hash,
                "original_dataset_path": self.original_dataset_path,
                "original_dataset_name": self.original_dataset_name,
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
        from ml.data.ds import Data
        log.debug("LOADING DS FOR MODEL: {} {} {} {}".format(self.cls_name(), self.model_name, 
            self.model_version, self.check_point_path))
        meta = self.load_meta()["model"]
        dataset = Data.original_ds(name=meta["test_ds_name"], dataset_path=meta["test_ds_path"])
        self.original_dataset_hash = meta["original_dataset_hash"]
        self.original_dataset_path = meta["original_dataset_path"]
        self.original_dataset_name = meta["original_dataset_name"]
    
        self.group_name = meta.get('group_name', None)
        if meta.get('md5', None) != dataset.md5:
            log.info("The dataset md5 is not equal to the model '{}'".format(
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
        self.train_ds = None
        self.validation_ds = None
        self.target_group = None
        self.data_group = None
        super(SupervicedModel, self).__init__(
            check_point_path=check_point_path,
            model_name=model_name,
            group_name=group_name,
            metrics=metrics)

    def load(self, model_version):
        pass

    def set_dataset(self, train_ds:Data, test_ds:Data, validation_ds: Data=None, pipeline=None):
       # with dataset:
       #     self.original_dataset_hash = dataset.hash
       #     self.original_dataset_path = dataset.dataset_path
       #     self.original_dataset_name = dataset.name
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.validation_ds = validation_ds
        self.pipeline = pipeline
        self.save_meta(keys="model")

    def metadata_model(self):
        print(self.target_group)
        with self.test_ds, self.train_ds, self.validation_ds:
            return {
                "test_ds_path": self.test_ds.dataset_path,
                "test_ds_name": self.test_ds.name,
                "train_ds_path": self.train_ds.dataset_path,
                "train_ds_name": self.train_ds.name,
                "validation_ds_path": self.validation_ds.dataset_path,
                "validation_ds_name": self.validation_ds.name,
                "hash": self.test_ds.hash,
                "target_group": self.target_group,
                "data_group": self.data_group,
                #"original_dataset_hash": self.original_dataset_hash,
                #"original_dataset_path": self.original_dataset_path,
                #"original_dataset_name": self.original_dataset_name,
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_path": self.path_m,
                "versions": []
            }

    def get_train_validation_ds(self):
        from ml.data.ds import Data
        log.debug("LOADING DS FOR MODEL: {} {} {} {}".format(self.cls_name(), self.model_name, 
            self.model_version, self.check_point_path))
        meta = self.load_meta()["model"]
        self.train_ds = Data.original_ds(name=meta["train_ds_name"], 
            dataset_path=meta["train_ds_path"])
        self.validation_ds = Data.original_ds(name=meta["validation_ds_name"], 
            dataset_path=meta["validation_ds_path"])

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

    def train(self, batch_size=0, num_steps=0, n_splits=None, obj_fn=None, model_params={},
              data_group=None, target_group=None):
        log.info("Training")
        self.model_params = model_params
        self.num_steps = num_steps
        self.target_group = target_group
        self.data_group = data_group
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
