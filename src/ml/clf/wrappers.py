import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from ml.models import MLModel, SupervicedModel
from ml.data.it import Iterator
from ml import measures as metrics
from ml.utils.logger import log_config
from ml.measures import ListMeasure


log = log_config(__name__)


class ClassifModel(SupervicedModel):
    def __init__(self, **params):
        self.num_classes = None
        super(ClassifModel, self).__init__(**params)

    def scores(self, measures=None, batch_size: int = 2000) -> ListMeasure:
        if measures is None or isinstance(measures, str):
            measure = metrics.MeasureBatch(name=self.model_name, batch_size=batch_size)
            measures = measure.make_metrics(measures=measures)
        test_data = self.ds[self.data_groups["data_test_group"]]
        for measure_fn in measures:
            test_target = Iterator(self.ds[self.data_groups["target_test_group"]]).batchs(batch_size=batch_size)
            predictions = self.predict(test_data, output=measure_fn.output, batch_size=batch_size)
            for pred, target in zip(predictions, test_target):
                measures.update_fn(pred, target, measure_fn)
        return measures.to_list()


class SKL(ClassifModel):
    # (np.arange(self.num_classes) == target).astype(np.float32)

    def ml_model(self, model) -> MLModel:
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit,
                       predictors=model.predict,
                       load_fn=self.load_fn,
                       save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)

    def output_format(self, prediction, output=None) -> np.ndarray:
        if output == 'uncertain' or output == 'n_dim':
            return prediction
        else:
            return (prediction > .5).astype(int)


class SKLP(ClassifModel):

    def ml_model(self, model) -> MLModel:
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit,
                       predictors=model.predict_proba,
                       load_fn=self.load_fn,
                       save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)

    def output_format(self, prediction, output=None) -> np.ndarray:
        if output == 'uncertain' or output == 'n_dim':
            return prediction
        else:
            return np.argmax(prediction, axis=1)


class XGB(ClassifModel):
    def ml_model(self, model, bst=None) -> MLModel:
        self.bst = bst
        return MLModel(fit_fn=model.train,
                       predictors=self.bst.predict,
                       load_fn=self.load_fn,
                       save_fn=self.bst.save_model,
                       input_transform=XGB.array2dmatrix)

    def load_fn(self, path):
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(path)
        self.model = self.ml_model(xgb, bst=bst)

    @staticmethod
    def array2dmatrix(data):
        import xgboost as xgb
        return xgb.DMatrix(data.to_ndarray())

    def output_format(self, prediction, output=None) -> np.ndarray:
        if output == 'uncertain' or output == 'n_dim':
            return prediction
        else:
            return (prediction > .5).astype(int)


class LGB(ClassifModel):
    def ml_model(self, model, bst=None) -> MLModel:
        self.bst = bst
        return MLModel(fit_fn=model.train,
                       predictors=self.bst.predict,
                       load_fn=self.load_fn,
                       save_fn=self.bst.save_model,
                       input_transform=None)

    def load_fn(self, path):
        import lightgbm as lgb
        bst = lgb.Booster(model_file=path)
        self.model = self.ml_model(lgb, bst=bst)

    def output_format(self, prediction, output=None) -> np.ndarray:
        if output == 'uncertain' or output == 'n_dim':
            return prediction
        else:
            return (prediction > .5).astype(int)


#class TFL(ClassifModel):

#    def reformat_labels(self, labels):
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
#        return (np.arange(self.num_classes) == labels[:, None]).astype(np.float)

#    def load_fn(self, path):
#        model = self.prepare_model()
#        self.model = MLModel(fit_fn=model.fit,
#                             predictors=model.predict,
#                             load_fn=self.load_fn,
#                             save_fn=model.save)

#    def predict(self, data, output=None, transform=True, chunks_size=1):
#        with tf.Graph().as_default():
#            return super(TFL, self).predict(data, output=output)

#    def train(self, batch_size=10, num_steps=1000, n_splits=None):
#        with tf.Graph().as_default():
#            self.model = self.prepare_model()
#            self.model.fit(self.dataset.train_data,
#                self.dataset.train_labels,
#                n_epoch=num_steps,
#                validation_set=(self.dataset.validation_data, self.dataset.validation_labels),
#                show_metric=True,
#                batch_size=batch_size,
#                run_id="tfl_model")


class Keras(ClassifModel):
    def __init__(self, **kwargs):
        super(Keras, self).__init__(**kwargs)
        self.ext = "ckpt"

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path)
        self.model = self.ml_model(model)

    def ml_model(self, model) -> MLModel:
        return MLModel(fit_fn=model.fit,
                       predictors=model.predict,
                       load_fn=self.load_fn,
                       save_fn=model.save,
                       to_json_fn=model.to_json)

    def output_format(self, prediction, output=None) -> np.ndarray:
        if output == 'uncertain' or output == 'n_dim':
            return prediction
        else:
            return np.argmax(prediction, axis=1)
