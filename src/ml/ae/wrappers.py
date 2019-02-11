from ml.utils.config import get_settings
from ml.models import MLModel, UnsupervisedModel
from ml.utils.logger import log_config
from keras import backend

settings = get_settings("ml")
log = log_config(__name__)

if backend._BACKEND == "theano":
    raise Exception("Theano does not support the autoencoders wrappers, change it with export KERAS_BACKEND=tensorflow")


class BaseAe(UnsupervisedModel):
    def __init__(self, latent_dim=2):
        self.model_encoder = None
        self.model_decoder = None
        self.latent_dim = latent_dim
        super(BaseAe, self).__init__()


class KerasAe(BaseAe):
    def custom_objects(self):
        return None

    def ml_model(self, model) -> MLModel:
        return MLModel(fit_fn=model.fit_generator,
                       predictors=model.predict,
                       load_fn=self.load_fn,
                       save_fn=model.save,
                       to_json_fn=model.to_json)

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path, custom_objects=self.custom_objects())
        self.model = self.ml_model(model)

