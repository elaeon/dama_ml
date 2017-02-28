import os
import numpy as np
import tensorflow as tf
import logging

from ml.utils.config import get_settings
from ml.models import MLModel
from ml.clf.wrappers import DataDrive
from ml.ds import Data

settings = get_settings("ml")

logging.basicConfig()
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(console)


class BaseAe(DataDrive):
    def __init__(self, model_name=None, dataset=None, check_point_path=None, 
                model_version=None, dataset_train_limit=None, info=True, 
                auto_load=True, group_name=None, latent_dim=2):
        self.model = None
        self.dataset_train_limit = dataset_train_limit
        self.print_info = info
        self._original_dataset_md5 = None
        self.latent_dim = latent_dim
        super(BaseAe, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)
        if auto_load is True:
            self.load_dataset(dataset)

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def reformat(self, dataset):
        dataset = self.transform_shape(dataset)
        return dataset

    def transform_shape(self, data, size=None):
        if size is None:
            size = data.shape[0]
        return data[:].reshape(size, -1)

    def reformat_all(self, dataset):
        log.info("Reformating {}...".format(self.cls_name()))
        ds = Data(
            name=dataset.name+"_"+self.model_name+"_"+self.cls_name(),
            dataset_path=settings["dataset_model_path"],
            apply_transforms=True,
            compression_level=9,
            dtype=dataset.dtype,
            transforms=dataset.transforms,
            chunks=1000,
            rewrite=False)

        if ds.mode == "w":
            ds._applied_transforms = dataset.apply_transforms
            data = self.reformat(dataset.data)
            ds.build_dataset(data)
        self.num_features = ds.data.shape[-1]
        ds.close_reader()
        return ds

    def load_dataset(self, dataset):
        if dataset is None:
            self.dataset = self.get_dataset()
        else:
            self.set_dataset(dataset)

    def set_dataset(self, dataset):
        self._original_dataset_md5 = dataset.md5()
        self.dataset = self.reformat_all(dataset)
        
    def chunk_iter(self, data, chunk_size=1, transform_fn=None, uncertain=False):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            data = np.asarray(list(chunk))
            size = data.shape[0]
            for prediction in self._predict(transform_fn(data, size), raw=uncertain):
                yield prediction

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        if self.model is None:
            self.load_model()

        if not isinstance(chunk_size, int):
            log.warning("The parameter chunk_size must be an integer.")            
            log.warning("Chunk size is set to 1")
            chunk_size = 1

        if transform is True and chunk_size > 0:
            fn = lambda x, s: self.transform_shape(
                self.dataset.processing(x, initial=False), size=s)
            return self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw)
        elif transform is True and chunk_size == 0:
            data = self.transform_shape(self.dataset.processing(data, initial=False))
            return self._predict(data, raw=raw)
        elif transform is False and chunk_size > 0:
            fn = lambda x, s: self.transform_shape(x, size=s)
            return self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw)
        elif transform is False and chunk_size == 0:
            if len(data.shape) == 1:
                data = self.transform_shape(data)
            return self._predict(data, raw=raw)

    def _metadata(self):
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "md5": self._original_dataset_md5, #not reformated dataset
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_version": self.model_version}

    def get_dataset(self):
        from ml.ds import DataSetBuilder, Data
        meta = self.load_meta()
        #dataset = Data.original_ds(meta["dataset_name"])
        dataset = DataSetBuilder(meta["dataset_name"], dataset_path=meta["dataset_path"])
        print(dataset.name, meta["dataset_path"])
        f = dataset._open_attrs()
        print(f.attrs.items())
        f.close()
        print(dataset.md5(), meta["md5"])
        self._original_dataset_md5 = meta["md5"]
        self.group_name = meta.get('group_name', None)
        if meta.get('md5', None) != dataset.md5():
            log.warning("The dataset md5 is not equal to the model '{}'".format(
                self.__class__.__name__))
        return dataset


class Keras(BaseAe):
    def default_model(self, model):
        return MLModel(fit_fn=model.fit_generator, 
                predictors=[model.predict],
                load_fn=self.load_fn,
                save_fn=model.save)

    def load_fn(self, path):
        from keras.models import load_model
        net_model = load_model(path, custom_objects=self.custom_objects())
        self.model = self.default_model(net_model)

    def preload_model(self):
        self.model = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_fn,
                            save_fn=None)

    def save_model(self):
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.save('{}.ckpt'.format(path))
            self.save_meta()

    def load_model(self):
        self.preload_model()
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.load('{}.ckpt'.format(path))

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        return super(Keras, self).predict(data, raw=raw, transform=transform, chunk_size=chunk_size)

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            yield prediction
