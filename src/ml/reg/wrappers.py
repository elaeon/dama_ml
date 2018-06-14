import numpy as np
import tensorflow as tf
import logging

from sklearn.preprocessing import LabelEncoder
from ml.utils.config import get_settings
from ml.models import SupervicedModel, MLModel
from ml.data.ds import DataLabel, Data
from ml import measures as metrics
from tqdm import tqdm


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


class RegModel(SupervicedModel):
    def __init__(self, **params):
        self.labels_dim = 1
        super(RegModel, self).__init__(**params)

    def load(self, model_version="1"):
        self.model_version = model_version
        self.test_ds = self.get_dataset()
        self.get_train_validation_ds()
        self.load_model()

    def scores(self, measures=None, chunks_size=2000):
        if measures is None or isinstance(measures, str):
            measures = metrics.Measure.make_metrics(measures="msle", name=self.model_name)
        with self.test_ds:
            test_data = self.test_ds.data[:]
            test_labels = self.test_ds.labels[:]
            length = test_labels.shape[0]

        for output in measures.outputs():
            predictions = self.predict(test_data, output=output, 
                transform=False, chunks_size=chunks_size).to_memory(length)
            measures.set_data(predictions, test_labels, output=output)
        return measures.to_list()

    #def confusion_matrix(self):
    #    with self.test_ds:
    #        test_data = self.test_ds.data[:]
    #        test_labes = self.test_ds.labels[:]
    #    predictions = self.predict(test_data, raw=False, transform=False, 
    #                            chunks_size=0)
    #    measure = metrics.Measure(np.asarray(list(tqdm(predictions, 
    #                    total=test_labels.shape[0]))),
    #                    test_labels, 
    #                    labels2classes=None,
    #                    name=self.__class__.__name__)
    #    measure.add(metrics.confusion_matrix, greater_is_better=None, uncertain=False)
    #    return measure.to_list()

    def convert_labels(self, labels, output=None):
        for chunk in labels:
            yield chunk

    def reformat_all(self, dataset, train_size=.7, valid_size=.1, unbalanced=None, chunks_size=30000):
        log.info("Reformating {}...".format(self.cls_name()))
        dl_train = DataLabel(
            dataset_path=settings["dataset_model_path"],
            compression_level=3,
            clean=True)
        dl_train.transforms = dataset.transforms
        dl_test = DataLabel(
            dataset_path=settings["dataset_model_path"],
            compression_level=3,
            clean=True)
        dl_test.transforms = dataset.transforms
        dl_validation = DataLabel(
            dataset_path=settings["dataset_model_path"],
            compression_level=3,
            clean=True)
        dl_validation.transforms = dataset.transforms

        train_data, validation_data, test_data, train_labels, validation_labels, test_labels = dataset.cv(
             train_size=train_size, valid_size=valid_size, unbalanced=unbalanced)
        with dl_train:
            dl_train.from_data(train_data, train_labels, chunks_size=chunks_size,
                transform=False)
            dl_train.columns = dataset.columns
        with dl_test:
            dl_test.from_data(test_data, test_labels, chunks_size=chunks_size,
                transform=False)
            dl_test.columns = dataset.columns
        with dl_validation:
            dl_validation.from_data(validation_data, validation_labels, 
                chunks_size=chunks_size, transform=False)
            dl_validation.columns = dataset.columns

        return dl_train, dl_test, dl_validation

    def _predict(self, data, output=None):
        prediction = self.model.predict(data)
        return self.convert_labels(prediction, output=output)


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
