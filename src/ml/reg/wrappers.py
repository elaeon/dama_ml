import numpy as np
import tensorflow as tf
import logging

from sklearn.preprocessing import LabelEncoder
from ml.utils.config import get_settings
from ml.models import BaseModel, MLModel
from ml.ds import DataLabel, Data
from ml.clf import measures as metrics
from ml.layers import IterLayer
from tqdm import tqdm


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


class RegModel(BaseModel):
    def __init__(self, **params):
        self.labels_dim = 1
        super(RegModel, self).__init__(**params)

    def scores(self, measures=None):
        if measures is None or isinstance(measures, str):
            measures = metrics.Measure.make_metrics(measures="msle", name=self.model_name)
        with self.test_ds:
            test_data = self.test_ds.data[:]
            test_labels = self.test_ds.labels[:]

        predictions = np.asarray(list(tqdm(
            self.predict(test_data, raw=measures.has_uncertain(), transform=False, chunks_size=0), 
            total=test_labels.shape[0])))
        measures.set_data(predictions, test_labels, None)
        log.info("Getting scores")
        return measures.to_list()

    def confusion_matrix(self):
        with self.test_ds:
            test_data = self.test_ds.data[:]
            test_labes = self.test_ds.labels[:]
        predictions = self.predict(test_data, raw=False, transform=False, 
                                chunks_size=0)
        measure = metrics.Measure(np.asarray(list(tqdm(predictions, 
                        total=test_labels.shape[0]))),
                        test_labels, 
                        labels2classes=None,
                        name=self.__class__.__name__)
        measure.add(metrics.confusion_matrix, greater_is_better=None, uncertain=False)
        return measure.to_list()

    def reformat_all(self, dataset):
        log.info("Reformating {}...".format(self.cls_name()))
        dl_train = DataLabel(
            dataset_path=settings["dataset_model_path"],
            apply_transforms=not dataset._applied_transforms,
            compression_level=9,
            transforms=dataset.transforms,
            rewrite=True)
        dl_test = DataLabel(
            dataset_path=settings["dataset_model_path"],
            apply_transforms=not dataset._applied_transforms,
            compression_level=9,
            transforms=dataset.transforms,
            rewrite=True)
        dl_validation = DataLabel(
            dataset_path=settings["dataset_model_path"],
            apply_transforms=not dataset._applied_transforms,
            compression_level=9,
            transforms=dataset.transforms,
            rewrite=True)

        train_data, validation_data, test_data, train_labels, validation_labels, test_labels = dataset.cv()
        with dl_train:
            dl_train.build_dataset(train_data, train_labels, chunks_size=30000)
            dl_train.apply_transforms = True
            dl_train._applied_transforms = dataset._applied_transforms
        with dl_test:
            dl_test.build_dataset(test_data, test_labels, chunks_size=30000)
            dl_test.apply_transforms = True
            dl_test._applied_transforms = dataset._applied_transforms
        with dl_validation:
            dl_validation.build_dataset(validation_data, validation_labels, chunks_size=30000)
            dl_validation.apply_transforms = True
            dl_validation._applied_transforms = dataset._applied_transforms

        return dl_train, dl_test, dl_validation

    def _predict(self, data, raw=False):
        prediction = self.model.predict(data)
        if not isinstance(prediction, np.ndarray):
            prediction = np.asarray(prediction, dtype=np.float)
        return prediction


class SKLP(RegModel):
    def __init__(self, *args, **kwargs):
        super(SKLP, self).__init__(*args, **kwargs)

    def ml_model(self, model):        
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit, 
                            predictors=[model.predict],
                            load_fn=self.load_fn,
                            save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        from sklearn.externals import joblib
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)


class LGB(RegModel):
    def ml_model(self, model, model_2=None):
        return MLModel(fit_fn=model.train, 
                            predictors=[model_2.predict],
                            load_fn=self.load_fn,
                            save_fn=model_2.save_model)

    def load_fn(self, path):
        import lightgbm as lgb
        bst = lgb.Booster(model_file=path)
        self.model = self.ml_model(lgb, model_2=bst)

    def array2dmatrix(self, data):
        import lightgbm as lgb
        return lgb.Dataset(data)
