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

    #def load(self, model_version):
    #    self.model_version = model_version
    #    self.test_ds = self.get_dataset()
    #    self.train_ds = self.test_ds
    #    self.load_model()

    #def reformat_all(self, dataset, chunks_size=30000):
    #    if dataset.module_cls_name() == DataLabel.module_cls_name():
    #        log.info("Reformating {}...".format(self.cls_name()))
    #        train_ds = Data(
    #            dataset_path=settings["dataset_model_path"],
    #            compression_level=3,
    #            clean=True)
    #        train_ds.transforms = dataset.transforms

    #        with dataset:
    #            train_ds.from_data(dataset.data, chunks_size=chunks_size,
    #                transform=False)
    #            train_ds.columns = dataset.columns
    #    else:
    #        train_ds = dataset

    #    return train_ds, train_ds

    #def encode(self, data, transform=True, chunks_size=258):
    #    def fn(x, t=True):
    #        with self.test_ds:
    #            return self.test_ds.processing(x, apply_transforms=t, chunks_size=chunks_size)
    #    return Iterator(self.intermedian_layer_model().predict(fn(data, t=transform)),
    #         chunks_size=chunks_size)

    #def save(self, model_version="1"):
    #    self.model_version = model_version
    #    if self.check_point_path is not None:
    #        self.path_mv = self.make_model_version_file()
    #        self.model.save('{}.{}'.format(self.path_mv, self.ext))
    #        self.save_meta(keys=["model", "train"])


class KerasAe(BaseAe):
    def custom_objects(self):
        return None

    def ml_model(self, model) -> MLModel:
        return MLModel(fit_fn=model.fit_generator,
                       predictors=model.predict,
                       load_fn=self.load_fn,
                       save_fn=model.save,
                       to_json_fn=model.to_json)

    #def intermedian_layer_model(self):
    #    from keras.models import Model
    #    model =  Model(inputs=self.model.model.input,
    #        outputs=self.model.model.get_layer('intermedian_layer').output)
    #    return self.default_model(model, self.load_fn)

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path, custom_objects=self.custom_objects())
        self.model = self.ml_model(model)

    #def preload_model(self):
    #    self.model = MLModel(fit_fn=None,
    #                        predictors=None,
    #                        load_fn=self.load_fn,
    #                        save_fn=None)

    #def load_model(self):
    #    self.preload_model()
    #    if self.check_point_path is not None:
    #        path = self.make_model_version_file()
    #        self.model.load('{}.{}'.format(path, self.ext))

    #def load(self, model_version):
    #    self.model_version = model_version
    #    self.test_ds = self.get_dataset()
    #    self.load_model()

    #def calculate_batch(self, X, batch_size=1):
    #    while 1:
    #        n = int(round(X.shape[0] / batch_size, 0))
    #        for i in range(0, n):
    #            yield (X[i:i + batch_size], X[i:i + batch_size])

    #def train(self, batch_size=100, num_steps=50, num_epochs=50):
    #    with self.train_ds:
    #        limit = int(round(self.train_ds.data.shape[0] * .9))
    #        X = self.train_ds.data[:limit]
    #        Z = self.train_ds.data[limit:]
    #    batch_size_x = min(X.shape[0], batch_size)
    #    batch_size_z = min(Z.shape[0], batch_size)
    #    self.batch_size = min(batch_size_x, batch_size_z)
    #    self.prepare_model()
    #    x = self.calculate_batch(X, batch_size=self.batch_size)
    #    z = self.calculate_batch(Z, batch_size=self.batch_size)
    #    self.model.fit(x,
    #        steps_per_epoch=num_steps,
    #        epochs=num_epochs,
    #        validation_data=z,
    #        nb_val_samples=num_steps)
