import os
import numpy as np
import tensorflow as tf
import logging

from ml.utils.config import get_settings
from ml.models import MLModel
from ml.ds import Data

settings = get_settings("ml")

logging.basicConfig()
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(console)


class DataDrive(object):
    def __init__(self, check_point_path=None, model_version=None, model_name=None,
                group_name=None):
        if check_point_path is None:
            self.check_point_path = settings["checkpoints_path"]
        else:
            self.check_point_path = check_point_path
        self.model_version = model_version
        self.model_name = model_name
        self.group_name = group_name
        self.models_path = None

    def _metadata(self):
        pass

    def save_meta(self):
        from ml.ds import save_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", self._metadata())

    def load_meta(self):
        from ml.ds import load_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            return load_metadata(path+".xmeta")

    @classmethod
    def read_meta(self, data_name, path):        
        from ml.ds import load_metadata
        if data_name is not None:
            return load_metadata(path+".xmeta").get(data_name, None)
        return load_metadata(path+".xmeta")

    def get_model_path(self):
        model_name_v = self.get_model_name_v()
        path = os.path.join(self.check_point_path, self.__class__.__name__)
        return os.path.join(path, model_name_v)

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


class BaseAe(DataDrive):
    def __init__(self, model_name=None, dataset=None, check_point_path=None, 
                model_version=None, dataset_train_limit=None, info=True, 
                auto_load=True, group_name=None, latent_dim=2):
        self.model = None
        self.model_encoder = None
        self.model_decoder = None
        self.print_info = info
        self.original_dataset = None
        self.dataset = None
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
            rewrite=True)

        if ds.mode == "w":
            ds._applied_transforms = dataset.apply_transforms
            data = self.reformat(dataset.data)
            ds.build_dataset(data)
        ds.close_reader()
        return ds

    def load_dataset(self, dataset):
        if dataset is None:
            self.dataset = self.get_dataset()
        else:
            self.set_dataset(dataset)        
        self.num_features = self.dataset.num_features()

    def set_dataset(self, dataset):
        self.original_dataset = dataset
        self.dataset = self.reformat_all(dataset)
        
    def chunk_iter(self, data, chunk_size=1, transform_fn=None, uncertain=False, decoder=True):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            data = np.asarray(list(chunk))
            size = data.shape[0]
            for prediction in self._predict(transform_fn(data, size), raw=uncertain, decoder=decoder):
                yield prediction

    def predict(self, data, raw=False, transform=True, chunk_size=1, model_type="decoder"):
        if self.model is None:
            self.load_model()

        decoder = model_type == "decoder"
        if not isinstance(chunk_size, int):
            log.warning("The parameter chunk_size must be an integer.")            
            log.warning("Chunk size is set to 1")
            chunk_size = 1

        if transform is True and chunk_size > 0:
            fn = lambda x, s: self.transform_shape(
                self.dataset.processing(x, initial=False), size=s)
            return self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw, decoder=decoder)
        elif transform is True and chunk_size == 0:
            data = self.transform_shape(self.dataset.processing(data, initial=False))
            return self._predict(data, raw=raw, decoder=decoder)
        elif transform is False and chunk_size > 0:
            fn = lambda x, s: self.transform_shape(x, size=s)
            return self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw, decoder=decoder)
        elif transform is False and chunk_size == 0:
            if len(data.shape) == 1:
                data = self.transform_shape(data)
            return self._predict(data, raw=raw, decoder=decoder)

    def _metadata(self):
        log.info("Generating metadata...")
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "o_dataset_path": self.original_dataset.dataset_path,
                "o_dataset_name": self.original_dataset.name,
                "md5": self.original_dataset.md5(), #not reformated dataset
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "models_path": self.models_path}

    def get_dataset(self):
        from ml.ds import Data
        meta = self.load_meta()
        self.original_dataset = Data.original_ds(meta["o_dataset_name"], 
            dataset_path=meta["o_dataset_path"])
        dataset = Data.original_ds(meta["dataset_name"], dataset_path=meta["dataset_path"])
        self.group_name = meta.get('group_name', None)
        if meta.get('md5', None) != self.original_dataset.md5():
            log.warning("The dataset md5 is not equal to the model '{}'".format(
                self.__class__.__name__))
        return dataset


class Keras(BaseAe):
    def default_model(self, model, load_fn):
        return MLModel(fit_fn=model.fit_generator, 
                predictors=[model.predict],
                load_fn=load_fn,
                save_fn=model.save)

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path, custom_objects=self.custom_objects())
        self.model = self.default_model(model, self.load_fn)

    def load_d_fn(self, path):
        from keras.models import load_model
        model = load_model(path, custom_objects=self.custom_objects())
        self.decoder_m = self.default_model(model, self.load_d_fn)

    def load_e_fn(self, path):
        from keras.models import load_model
        model = load_model(path, custom_objects=self.custom_objects())
        self.encoder_m = self.default_model(model, self.load_e_fn)

    def preload_model(self):
        self.model = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_fn,
                            save_fn=None)
        self.encoder_m = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_e_fn,
                            save_fn=None)
        self.decoder_m = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_d_fn,
                            save_fn=None)

        return [self.model, self.encoder_m, self.decoder_m]

    def save_model(self):
        if self.check_point_path is not None:
            self.models_path = self.make_model_file()
            self.model.save('{}.ckpt'.format(self.models_path))
            self.save_meta()

    def load_model(self):
        log.info("loading models...")
        models = self.preload_model()
        if self.check_point_path is not None:
            meta_path = os.path.join(self.get_model_path(), self.get_model_name_v())
            path = self.read_meta("models_path", meta_path)
            for model in models:
                model.load('{}.ckpt'.format(path))

    def _predict(self, data, raw=False, decoder=True):
        if decoder is True:
            model = self.model if not hasattr(self, 'decoder_m') else self.decoder_m
        else:
            model = self.encoder_m

        for prediction in model.predict(data):
            yield prediction
