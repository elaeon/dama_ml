import numpy as np
import tensorflow as tf

from sklearn.externals import joblib
from ml.models import MLModel, SupervicedModel
from ml.data.it import Iterator
from ml import measures as metrics
from ml.utils.logger import log_config

log = log_config(__name__)


class ClassifModel(SupervicedModel):
    def __init__(self, **params):
        self.target_dim = None
        self.num_classes = None
        super(ClassifModel, self).__init__(**params)

    def load(self, model_version):
        self.model_version = model_version
        self.ds = self.get_dataset()
        self.load_model()

    def scores(self, measures=None, batch_size: int=2000):
        if measures is None or isinstance(measures, str):
            measure = metrics.MeasureBatch(name=self.model_name, batch_size=batch_size)
            measures = measure.make_metrics(measures=measures)
        with self.ds:
            test_data = self.ds[self.data_groups["data_test_group"]]
            for measure_fn in measures:
                test_target = Iterator(self.ds[self.data_groups["target_test_group"]]).batchs(batch_size=batch_size)
                predictions = self.predict(test_data, output=measure_fn.output, batch_size=batch_size)
                for pred, target in zip(predictions, test_target):
                    measures.update_fn(pred, target, measure_fn)
        return measures.to_list()

    def erroneous_clf(self):
        import operator
        return self.only_is(operator.ne)

    def correct_clf(self):
        import operator
        return self.only_is(operator.eq)

    def reformat_target(self, target):
        return target

    def position_index(self, labels):
        if len(labels.shape) >= 2:
            return np.argmax(labels, axis=1)
        else:
            return labels

    def output_format(self, prediction, output=None):
        if output == 'uncertain' or output == 'n_dim':
            return prediction
        else:
            return (prediction > .5).astype(int)


class SKL(ClassifModel):
    def convert_label(self, target, output=None):
        if output is 'n_dim':
            return (np.arange(self.num_classes) == target).astype(np.float32)
        elif output is None:
            return self.position_index(target)

    def ml_model(self, model):        
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit,
                       predictors=model.predict,
                       load_fn=self.load_fn,
                       save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)


class SKLP(ClassifModel):

    def ml_model(self, model):        
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit,
                       predictors=model.predict_proba,
                       load_fn=self.load_fn,
                       save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)


class XGB(ClassifModel):
    def ml_model(self, model, bst=None):
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


class LGB(ClassifModel):
    def ml_model(self, model, bst=None):
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

    @staticmethod
    def array2dmatrix(data):
        import lightgbm as lgb
        return lgb.Dataset(data.to_ndarray())


class TFL(ClassifModel):

    def reformat_labels(self, labels):
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        return (np.arange(self.num_classes) == labels[:,None]).astype(np.float)

    def load_fn(self, path):
        model = self.prepare_model()
        self.model = MLModel(fit_fn=model.fit, 
                            predictors=model.predict,
                            load_fn=self.load_fn,
                            save_fn=model.save)

    def predict(self, data, output=None, transform=True, chunks_size=1):
        with tf.Graph().as_default():
            return super(TFL, self).predict(data, output=output)

    def train(self, batch_size=10, num_steps=1000, n_splits=None):
        with tf.Graph().as_default():
            self.model = self.prepare_model()
            self.model.fit(self.dataset.train_data, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.validation_data, self.dataset.validation_labels),
                show_metric=True, 
                batch_size=batch_size,
                run_id="tfl_model")


class Keras(ClassifModel):
    def __init__(self, **kwargs):
        super(Keras, self).__init__(**kwargs)
        self.ext = "ckpt"

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path)
        self.model = self.ml_model(model)

    def ml_model(self, model):
        return MLModel(fit_fn=model.fit, 
                        predictors=model.predict,
                        load_fn=self.load_fn,
                        save_fn=model.save)

    def reformat_labels(self, labels):
        self.target_dim = self.num_classes
        return (np.arange(self.num_classes) == labels[:, None]).astype(np.float)

    def train_kfolds(self, batch_size=10, num_steps=100, n_splits=None):
        from sklearn.model_selection import StratifiedKFold
        self.model = self.prepare_model_k()
        cv = StratifiedKFold(n_splits=n_splits)
        
        with self.train_ds:
            labels = self.position_index(self.train_ds.labels[:])
            for k, (train, test) in enumerate(cv.split(self.train_ds.data, labels), 1):
                train = list(train)
                test = list(test)
                self.model.fit(self.train_ds.data[train], 
                    self.train_ds.labels[train],
                    epochs=num_steps,
                    batch_size=batch_size,
                    shuffle="batch",
                    validation_data=(self.train_ds.data[test], self.train_ds.labels[test]))
                print("fold ", k)

    def train(self, batch_size=0, num_steps=0, n_splits=None):
        if n_splits is not None:
            self.train_kfolds(batch_size=batch_size, num_steps=num_steps, n_splits=n_splits)
        else:
            self.model = self.prepare_model()
            with self.train_ds, self.validation_ds:
                self.model.fit(self.train_ds.data, 
                    self.train_ds.labels,
                    epochs=num_steps,
                    batch_size=batch_size,
                    shuffle="batch",
                    validation_data=(self.validation_ds.data, self.validation_ds.labels))

