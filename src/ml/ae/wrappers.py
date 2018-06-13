import numpy as np
import tensorflow as tf
import logging

from ml.utils.config import get_settings
from ml.models import MLModel, BaseModel
from ml.data.ds import Data, DataLabel
from ml.data.it import Iterator

settings = get_settings("ml")

logging.basicConfig()
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(console)

from keras import backend
if backend._BACKEND == "theano":
    raise Exception, "Theano does not support the autoencoders wrappers, change it with export KERAS_BACKEND=tensorflow"


class BaseAe(BaseModel):
    def __init__(self, model_name=None, check_point_path=None, 
                group_name=None, latent_dim=2):
        self.model_encoder = None
        self.model_decoder = None
        self.latent_dim = latent_dim
        super(BaseAe, self).__init__(
            check_point_path=check_point_path,
            model_name=model_name,
            group_name=group_name)

    def set_dataset(self, dataset, chunks_size=30000):
        with dataset:
            self.original_dataset_md5 = dataset.md5
            self.original_dataset_path = dataset.dataset_path
            self.original_dataset_name = dataset.name
            self.train_ds, self.test_ds = self.reformat_all(dataset, chunks_size=chunks_size)
        self.save_meta(keys="model")

    def load(self, model_version):
        self.model_version = model_version
        self.test_ds = self.get_dataset()
        self.train_ds = self.test_ds
        self.load_model()

    def reformat_all(self, dataset, chunks_size=30000):
        if dataset.module_cls_name() == DataLabel.module_cls_name():
            log.info("Reformating {}...".format(self.cls_name()))
            train_ds = Data(
                dataset_path=settings["dataset_model_path"],
                compression_level=3,
                clean=True)
            train_ds.transforms = dataset.transforms

            with train_ds:
                train_ds.from_data(dataset.data, chunks_size=chunks_size, 
                    transform=False)
                train.columns = dataset.columns
        else:
            train_ds = dataset

        return train_ds, train_ds
        
    def scores(self, measures=None):
        from ml.clf.measures import ListMeasure
        return ListMeasure()

    def predict(self, data, output=None, transform=True, chunks_size=258, model_type="decoder"):
        def fn(x, t=True):
            with self.test_ds:
                return self.test_ds.processing(x, apply_transforms=t, chunks_size=chunks_size)

        decoder = model_type == "decoder"
        return Iterator(self._predict(fn(data, t=transform), output=output, decoder=decoder),
             chunks_size=chunks_size)


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

    def load_model(self):
        models = self.preload_model()
        if self.check_point_path is not None:
            path = self.make_model_version_file()
            for model in models:
                model.load('{}.{}'.format(path, self.ext))

    def load(self, model_version):
        self.model_version = model_version
        self.test_ds = self.get_dataset()
        self.load_model()

    def _predict(self, data, output=None, decoder=True):
        if decoder is True:
            model = self.model if not hasattr(self, 'decoder_m') else self.decoder_m
        else:
            model = self.encoder_m

        return model.predict(data)
