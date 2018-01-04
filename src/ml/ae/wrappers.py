import numpy as np
import tensorflow as tf
import logging

from ml.utils.config import get_settings
from ml.models import MLModel, BaseModel
from ml.ds import Data, DataLabel
from ml.layers import IterLayer

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
    def __init__(self, model_name=None, dataset=None, check_point_path=None, 
                model_version=None, autoload=True, group_name=None, latent_dim=2):
        self.model_encoder = None
        self.model_decoder = None
        self.latent_dim = latent_dim
        super(BaseAe, self).__init__(
            dataset=dataset,
            autoload=autoload,
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)

    def reformat_all(self, dataset):
        if dataset.module_cls_name() == DataLabel.module_cls_name() or\
            dataset._applied_transforms is False and not dataset.transforms.is_empty():
            log.info("Reformating {}...".format(self.cls_name()))
            train_ds = Data(
                dataset_path=settings["dataset_model_path"],
                apply_transforms=not dataset._applied_transforms,
                compression_level=9,
                transforms=dataset.transforms,
                rewrite=True)

            with train_ds:
                train_ds.build_dataset(dataset.data, chunks_size=1000)
                train_ds.apply_transforms = True
                train_ds._applied_transforms = dataset._applied_transforms
        else:
            train_ds = dataset

        return train_ds, train_ds, train_ds
        
    def scores(self, measures=None):
        from ml.clf.measures import ListMeasure
        return ListMeasure()

    def chunk_iter(self, data, chunk_size=1, transform_fn=None, uncertain=False, decoder=True, transform=True):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            data = np.asarray(list(chunk))
            for prediction in self._predict(transform_fn(data, transform), raw=uncertain, decoder=decoder):
                yield prediction

    def predict(self, data, raw=False, transform=True, chunks_size=258, model_type="decoder"):
        def fn(x, t=True):
            with self.test_ds:
                return self.test_ds.processing(x, apply_transforms=t, chunks_size=chunks_size)

        if self.model is None:
            self.load_model()

        decoder = model_type == "decoder"
        if not isinstance(chunks_size, int):
            log.info("The parameter chunk_size must be an integer.")            
            log.info("Chunk size is set to 1")
            chunks_size = 258

        if isinstance(data, IterLayer):
            return IterLayer(self._predict(fn(x, t=transform), raw=raw))
        else:
            if chunks_size > 0:
                return IterLayer(self.chunk_iter(data, chunks_size, transform_fn=fn, 
                                                uncertain=raw, transform=transform, decoder=decoder))
            else:
                return IterLayer(self._predict(fn(data, t=transform), raw=raw, decoder=decoder))


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
            models_path = self.make_model_file()
            self.model.save('{}.{}'.format(models_path, self.ext))
            self.save_meta()

    def load_model(self):
        log.info("loading models...")
        models = self.preload_model()
        if self.check_point_path is not None:
            path = self.get_model_path()
            for model in models:
                model.load('{}.{}'.format(path, self.ext))

    def _predict(self, data, raw=False, decoder=True):
        if decoder is True:
            model = self.model if not hasattr(self, 'decoder_m') else self.decoder_m
        else:
            model = self.encoder_m

        for prediction in model.predict(data):
            yield prediction
