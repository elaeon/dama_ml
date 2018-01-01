from ml.utils.config import get_settings
import os
import uuid
import logging
import numpy as np

from ml.ds import DataLabel, Data
from ml.layers import IterLayer


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


class MLModel:
    def __init__(self, fit_fn=None, predictors=None, load_fn=None, save_fn=None,
                transform_data=None):
        self.fit_fn = fit_fn
        self.predictors = predictors
        self.load_fn = load_fn
        self.save_fn = save_fn
        self.transform_data = transform_data

    def fit(self, *args, **kwargs):
        return self.fit_fn(*args, **kwargs)

    def predict(self, data):
        if isinstance(data, IterLayer):
            data = data.to_narray()
        if self.transform_data is not None:
            prediction = self.predictors[0](self.transform_data(data))
        else:
            prediction = self.predictors[0](data)
        return prediction

    def load(self, path):
        return self.load_fn(path)

    def save(self, path):
        return self.save_fn(path)


class DataDrive(object):
    def __init__(self, check_point_path=None, model_version=None, model_name=None,
                group_name=None):
        if check_point_path is None:
            self.check_point_path = settings["checkpoints_path"]
        else:
            self.check_point_path = check_point_path
        self.model_version = model_version
        self.model_name = uuid.uuid4().hex if model_name is None else model_name
        self.group_name = group_name

    def _metadata(self):
        pass

    def save_meta(self):
        from ml.ds import save_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", self._metadata())

    def edit_meta(self, key, value):
        from ml.ds import save_metadata
        meta = self.load_meta()
        meta[key] = value
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", meta)

    def load_meta(self):
        from ml.ds import load_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            return load_metadata(path+".xmeta")

    def get_model_path(self):
        model_name_v = self.get_model_name_v()
        path = os.path.join(self.check_point_path, self.__class__.__name__, model_name_v)
        return os.path.join(path, model_name_v)

    @classmethod
    def read_meta(self, data_name, path):        
        from ml.ds import load_metadata
        if data_name is not None:
            return load_metadata(path+".xmeta").get(data_name, None)
        return load_metadata(path+".xmeta")

    def get_model_name_v(self):
        if self.model_version is None:
            id_ = "0"
        else:
            id_ = self.model_version
        return "{}.{}".format(self.model_name, id_)

    def make_model_file(self):
        from ml.utils.files import check_or_create_path_dir
        model_name_v = self.get_model_name_v()
        check_point = check_or_create_path_dir(self.check_point_path, self.__class__.__name__)
        destination = check_or_create_path_dir(check_point, model_name_v)
        return os.path.join(check_point, model_name_v, model_name_v)

    def print_meta(self):
        print(self.load_meta())

    def destroy(self):
        """remove the dataset associated to the model and his checkpoints"""
        from ml.utils.files import rm
        rm(self.get_model_path()+"."+self.ext)
        rm(self.get_model_path()+".xmeta")
        if hasattr(self, 'dataset'):
            self.dataset.destroy()
        if hasattr(self, 'test'):
            self.test.destroy()

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)


class BaseModel(DataDrive):
    def __init__(self, model_name=None, dataset=None, check_point_path=None, 
                model_version=None, autoload=True, group_name=None, metrics=None,
                dtype='float64', ltype='float'):
        self.model = None
        self.original_dataset_md5 = None
        self.original_dataset_path = None
        self.original_dataset_name = None
        self.train_ds = None
        self.test_ds = None
        self.validation_ds = None
        self.ext = "ckpt.pkl"
        self.metrics = metrics
        self.ltype = ltype
        self.dtype = dtype

        super(BaseModel, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)

        if autoload is True:
            self.load_dataset(dataset)

    def scores(self, measures=None):
        pass

    def confusion_matrix(self):
        pass

    def only_is(self, op):
        pass

    def reformat_all(self, dataset):
        pass

    def reformat_labels(self, labels):
        pass

    def load_dataset(self, dataset):
        if dataset is None:
            self.test_ds = self.get_dataset()
        else:
            self.set_dataset(dataset)

        with self.test_ds:
            self.num_features = self.test_ds.num_features()

    def set_dataset(self, dataset):
        with dataset:
            self.original_dataset_md5 = dataset.md5
            self.original_dataset_path = dataset.dataset_path
            self.original_dataset_name = dataset.name
            self.train_ds, self.test_ds, self.validation_ds = self.reformat_all(dataset)

    def chunk_iter(self, data, chunk_size=1, transform_fn=None, uncertain=False, transform=True):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            data = np.asarray(list(chunk))
            size = data.shape[0]
            for prediction in self._predict(transform_fn(data, size, transform), raw=uncertain):
                yield prediction

    def predict(self, data, raw=False, transform=True, chunk_size=258):
        def fn(x, s=None, t=True):
            with self.test_ds:
                return self.test_ds.processing(x, apply_transforms=t)

        if self.model is None:
            self.load_model()

        if not isinstance(chunk_size, int):
            log.info("The parameter chunk_size must be an integer.")            
            log.info("Chunk size is set to 1")
            chunk_size = 258

        if isinstance(data, IterLayer):
            return IterLayer(self._predict(fn(x, t=transform), raw=raw))
        else:
            if chunk_size > 0:
                return IterLayer(self.chunk_iter(data, chunk_size, transform_fn=fn, 
                                                uncertain=raw, transform=transform))
            else:
                return IterLayer(self._predict(fn(data, t=transform), raw=raw))

    def _metadata(self):
        list_measure = self.scores(measures=self.metrics)
        with self.test_ds:
            return {"test_ds_path": self.test_ds.dataset_path,
                    "test_ds_name": self.test_ds.name,
                    "md5": self.test_ds.md5,
                    "original_ds_md5": self.original_dataset_md5,
                    "original_ds_path": self.original_dataset_path,
                    "original_ds_name": self.original_dataset_name,
                    "original_dataset_name": self.original_dataset_name,
                    "group_name": self.group_name,
                    "model_module": self.module_cls_name(),
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "score": list_measure.measures_to_dict()}
        
    def get_dataset(self):
        from ml.ds import Data
        log.debug("LOADING DS FOR MODEL: {} {} {} {}".format(self.cls_name(), self.model_name, 
            self.model_version, self.check_point_path))
        meta = self.load_meta()
        dataset = Data.original_ds(name=meta["test_ds_name"], dataset_path=meta["test_ds_path"])
        self.original_dataset_md5 = meta["original_ds_md5"]
        self.original_dataset_path = meta["original_ds_path"]
        self.original_dataset_name = meta["original_ds_name"]
    
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

    def save_model(self):
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.save('{}.{}'.format(path, self.ext))
            self.save_meta()

    def has_model_file(self):
        import os
        path = self.get_model_path()
        return os.path.exists('{}.{}'.format(path, self.ext))

    def load_model(self):        
        self.preload_model()
        if self.check_point_path is not None:
            path = self.get_model_path()
            self.model.load('{}.{}'.format(path, self.ext))

    def _predict(self, data, raw=False):
        pass

    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, 
                    model_params={}):
        from sklearn.model_selection import StratifiedKFold
        model = self.prepare_model_k(obj_fn=obj_fn)
        cv = StratifiedKFold(n_splits=n_splits)
        with self.train_ds:
            data = self.train_ds.data
            labels = self.train_ds.labels
            for k, (train, test) in enumerate(cv.split(data, labels), 1):
                model.fit(data[train], labels[train])
                print("fold ", k)
        return model

    def train(self, batch_size=0, num_steps=0, n_splits=None, obj_fn=None, model_params={}):
        log.info("Training")
        if n_splits is not None:
            self.model = self.train_kfolds(batch_size=batch_size, num_steps=num_steps, 
                            n_splits=n_splits, obj_fn=obj_fn, model_params=model_params)
        else:
            self.model = self.prepare_model(obj_fn=obj_fn, **model_params)
        log.info("Saving model")
        self.train_ds.destroy()
        self.validation_ds.destroy()
        self.save_model()

    def scores2table(self):
        from ml.clf.measures import ListMeasure
        return ListMeasure.dict_to_measures(self.load_meta().get("score", None))
