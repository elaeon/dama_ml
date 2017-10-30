import numpy as np
import tensorflow as tf
import logging

from sklearn.preprocessing import LabelEncoder
from ml.utils.config import get_settings
from ml.models import MLModel, DataDrive
from ml.ds import DataSetBuilder, Data
from ml.clf import measures as metrics
from ml.layers import IterLayer

settings = get_settings("ml")
logging.basicConfig()
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(console)
#np.random.seed(133)



class BaseClassif(DataDrive):
    def __init__(self, model_name=None, dataset=None, check_point_path=None, 
                model_version=None, dataset_train_limit=None, 
                autoload=True, group_name=None, rewrite=False, metrics=None):
        self.model = None
        self.le = LabelEncoder()
        self.dataset_train_limit = dataset_train_limit
        self.base_labels = None
        self._original_dataset_md5 = None
        self.dataset = None
        self.dl = None
        self.ext = "ckpt.pkl"
        self.rewrite = rewrite
        self.metrics = metrics

        super(BaseClassif, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)
        if autoload is True:
            self.load_dataset(dataset)

    def scores(self, measures=None):
        from tqdm import tqdm
        if measures is None or isinstance(measures, str):
            measures = metrics.Measure.make_metrics(measures, name=self.model_name)
        predictions = np.asarray(list(tqdm(
            self.predict(self.dataset.test_data[:], raw=measures.has_uncertain(), 
                        transform=False, chunk_size=258), 
            total=self.dataset.test_labels.shape[0])))
        measures.set_data(predictions, self.dataset.test_labels[:], self.numerical_labels2classes)
        log.info("Getting scores")
        return measures.to_list()

    def confusion_matrix(self):
        from tqdm import tqdm
        predictions = self.predict(self.dataset.test_data[:], raw=False, 
            transform=False, chunk_size=258)
        measure = metrics.Measure(np.asarray(list(tqdm(predictions, 
                        total=self.dataset.test_labels.shape[0]))),
                        self.dataset.test_labels[:], 
                        labels2classes=self.numerical_labels2classes,
                        name=self.__class__.__name__)
        measure.add(metrics.confusion_matrix, greater_is_better=None, uncertain=False)
        return measure.to_list()

    def only_is(self, op):
        predictions = np.asarray(list(self.predict(self.dataset.test_data, raw=False, transform=False)))
        data = zip(*filter(
                        lambda x: op(x[1], x[2]), 
                        zip(self.dataset.test_data, 
                            self.numerical_labels2classes(predictions), 
                            self.numerical_labels2classes(self.dataset.test_labels[:]))))
        if len(data) > 0:
            return np.array(data[0]), data[1], data[2]

    def erroneous_clf(self):
        import operator
        return self.only_is(operator.ne)

    def correct_clf(self):
        import operator
        return self.only_is(operator.eq)

    def reformat(self, dataset, labels):
        dataset = self.transform_shape(dataset)
        return dataset, labels

    def transform_shape(self, data, size=None):
        if isinstance(data, IterLayer):
            return np.asarray(list(data))
        elif len(data.shape) > 2:
            if size is None:
                size = data.shape[0]
            return data[:].reshape(size, -1)
        else:
            return data

    def is_binary():
        return self.num_labels == 2

    def labels_encode(self, labels):
        self.le.fit(labels)
        self.num_labels = self.le.classes_.shape[0]
        self.base_labels = self.le.classes_

    def position_index(self, label):
        if isinstance(label, np.ndarray) or isinstance(label, list):
            return np.argmax(label)
        return label

    def convert_label(self, label, raw=False):
        if raw is True:
            return label
        elif raw is None:
            return self.position_index(label)
        else:
            return self.le.inverse_transform(self.position_index(label))

    def numerical_labels2classes(self, labels):
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            return self.le.inverse_transform(np.argmax(labels, axis=1))
        else:
            return self.le.inverse_transform(labels.astype('int'))

    def reformat_all(self, dataset):
        log.info("Reformating {}...".format(self.cls_name()))
        dsb = DataSetBuilder(
            name=dataset.name+"_"+self.model_name+"_"+self.model_version+"_"+self.cls_name(),
            dataset_path=settings["dataset_model_path"],
            apply_transforms=not dataset._applied_transforms,
            compression_level=9,
            dtype=dataset.dtype,
            transforms=dataset.transforms,
            ltype='int',
            validator='',
            chunks=1000,
            rewrite=self.rewrite)

        self.labels_encode(dataset.labels)
        log.info("Labels encode finished")
        if dsb.mode == "w":
            train_data, train_labels = self.reformat(dataset.train_data, 
                                        self.le.transform(dataset.train_labels))
            test_data, test_labels = self.reformat(dataset.test_data, 
                                        self.le.transform(dataset.test_labels))
            validation_data, validation_labels = self.reformat(dataset.validation_data, 
                                        self.le.transform(dataset.validation_labels))
            dsb.build_dataset(train_data, train_labels, test_data=test_data, 
                            test_labels=test_labels, validation_data=validation_data, 
                            validation_labels=validation_labels)
            dsb.apply_transforms = True
            dsb._applied_transforms = dataset._applied_transforms
        dsb.close_reader()
        return dsb

    def load_dataset(self, dataset):
        if dataset is None:
            self.dataset = self.get_dataset()
        else:
            self.set_dataset(dataset)
        self.num_features = self.dataset.num_features()

    def set_dataset(self, dataset, auto=True):
        self._original_dataset_md5 = dataset.md5
        if auto is True:
            self.dataset = self.reformat_all(dataset)
        else:
            self.dataset.destroy()
            self.dataset = dataset
            self.labels_encode(dataset.labels)

    def chunk_iter(self, data, chunk_size=1, transform_fn=None, uncertain=False):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            data = np.asarray(list(chunk))
            size = data.shape[0]
            for prediction in self._predict(transform_fn(data, size), raw=uncertain):
                yield prediction

    def predict(self, data, raw=False, transform=True, chunk_size=258):

        if self.model is None:
            self.load_model()

        if not isinstance(chunk_size, int):
            log.warning("The parameter chunk_size must be an integer.")            
            log.warning("Chunk size is set to 1")
            chunk_size = 258

        if isinstance(data, IterLayer):
            def iter_(fn):
                for x in data:
                    yield IterLayer(self._predict(fn(x), raw=raw))

            if transform is True:
                fn = lambda x: self.transform_shape(
                    self.dataset.processing(np.asarray(list(x)), 
                    base_data=self.dataset.train_data[:]))
            else:
                fn = list
            return IterLayer(iter_(fn))
        else:
            if transform is True and chunk_size > 0:
                fn = lambda x, s: self.transform_shape(
                    self.dataset.processing(x, base_data=self.dataset.train_data[:]), size=s)
                return IterLayer(self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw))
            elif transform is True and chunk_size == 0:
                data = self.transform_shape(self.dataset.processing(data, 
                    base_data=self.dataset.train_data[:]))
                return IterLayer(self._predict(data, raw=raw))
            elif transform is False and chunk_size > 0:
                fn = lambda x, s: self.transform_shape(x, size=s)
                return IterLayer(self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw))
            elif transform is False and chunk_size == 0:
                if len(data.shape) == 1:
                    data = self.transform_shape(data)
                return IterLayer(self._predict(data, raw=raw))

    def _pred_erros(self, predictions, test_data, test_labels, valid_size=.1):
        validation_labels_d = {}
        pred_index = []
        for index, (pred, label) in enumerate(zip(predictions, test_labels)):
            if pred[1] >= .5 and (label < .5 or label == 0):
                pred_index.append((index, label - pred[1]))
                validation_labels_d[index] = 1
            elif pred[1] < .5 and (label >= .5 or label == 1):
                pred_index.append((index, pred[1] - label))
                validation_labels_d[index] = 0

        pred_index = sorted(filter(lambda x: x[1] < 0, pred_index), 
            key=lambda x: x[1], reverse=False)
        pred_index = pred_index[:int(len(pred_index) * valid_size)]
        validation_data = np.ndarray(
            shape=(len(pred_index), test_data.shape[1]), dtype=np.float32)
        validation_labels = np.ndarray(shape=len(pred_index), dtype=np.float32)
        for i, (j, _) in enumerate(pred_index):
            validation_data[i] = test_data[j]
            validation_labels[i] = validation_labels_d[j]
        return validation_data, validation_labels, pred_index

    def _metadata(self):
        list_measure = self.scores(measures=self.metrics)
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "md5": self.dataset.md5,
                "original_ds_md5": self._original_dataset_md5, #not reformated dataset
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "score": list_measure.measures_to_dict(),
                "base_labels": list(self.base_labels),
                "dl": self.dl.name if self.dl is not None else None}
        
    def get_dataset(self):
        from ml.ds import DataSetBuilder, Data
        meta = self.load_meta()
        try:
            dataset = DataSetBuilder(name=meta["dataset_name"], dataset_path=meta["dataset_path"],
                apply_transforms=False)
            self._original_dataset_md5 = meta["original_ds_md5"]
            self.labels_encode(meta["base_labels"])
            self.dl = Data.original_ds(name=meta["dl"], dataset_path=settings["dataset_model_path"])
        except KeyError:
            raise Exception, "No metadata found"
        else:
            self.group_name = meta.get('group_name', None)
            if meta.get('md5', None) != dataset.md5:
                log.warning("The dataset md5 is not equal to the model '{}'".format(
                    self.__class__.__name__))
            return dataset

    def preload_model(self):
        self.model = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_fn,
                            save_fn=None)

    def save_model(self):
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.save('{}.{}'.format(path, self.ext))
            self.save_meta()

    def load_model(self):        
        self.preload_model()
        if self.check_point_path is not None:
            path = self.get_model_path()
            self.model.load('{}.{}'.format(path, self.ext))

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            if not isinstance(prediction, np.ndarray):
                prediction = np.asarray(prediction)
            yield self.convert_label(prediction, raw=raw)

    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, 
                    model_params={}):
        from sklearn.model_selection import StratifiedKFold
        model = self.prepare_model_k(obj_fn=obj_fn)
        cv = StratifiedKFold(n_splits=n_splits)
        data = self.dataset.data_validation
        labels = self.dataset.data_validation_labels
        for k, (train, test) in enumerate(cv.split(data, labels), 1):
            model.fit(data[train], labels[train])
            print("fold ", k)
        return model

    def train(self, batch_size=0, num_steps=0, n_splits=None, obj_fn=None, model_params={}):
        log.info("Training")
        if n_splits is not None:
            self.model = self.train_kfolds(batch_size=batch_size, num_steps=num_steps, 
                            n_splits=n_splits, obj_fn=obj_fn, model_params=model_params)
        else:
            self.model = self.prepare_model(obj_fn=obj_fn, **model_params)
        log.info("Saving model")
        self.save_model()

    def scores2table(self):
        from ml.clf.measures import ListMeasure
        return ListMeasure.dict_to_measures(self.load_meta().get("score", None))


class SKL(BaseClassif):
    def convert_label(self, label, raw=False):
        if raw is True:
            return (np.arange(self.num_labels) == label).astype(np.float32)
        elif raw is None:
            return self.position_index(label)
        else:
            return self.le.inverse_transform(self.position_index(label))

    def ml_model(self, model):        
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit, 
                            predictors=[model.predict],
                            load_fn=self.load_fn,
                            save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)


class SKLP(BaseClassif):
    def __init__(self, *args, **kwargs):
        super(SKLP, self).__init__(*args, **kwargs)

    def ml_model(self, model):        
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit, 
                            predictors=[model.predict_proba],
                            load_fn=self.load_fn,
                            save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        from sklearn.externals import joblib
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)


class XGB(BaseClassif):
    def ml_model(self, model, model_2=None):
        return MLModel(fit_fn=model.train, 
                            predictors=[model_2.predict],
                            load_fn=self.load_fn,
                            save_fn=model_2.save_model,
                            transform_data=self.array2dmatrix)

    def load_fn(self, path):
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(path)
        self.model = self.ml_model(xgb, model_2=booster)

    def array2dmatrix(self, data):
        import xgboost as xgb
        return xgb.DMatrix(data)


class TFL(BaseClassif):

    def reformat(self, data, labels):
        data = self.transform_shape(data)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return data, labels_m

    def load_fn(self, path):
        model = self.prepare_model()
        self.model = MLModel(fit_fn=model.fit, 
                            predictors=[model.predict],
                            load_fn=self.load_fn,
                            save_fn=model.save)

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        with tf.Graph().as_default():
            return super(TFL, self).predict(data, raw=raw, transform=transform, chunk_size=chunk_size)

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
            self.save_model()


class Keras(BaseClassif):
    def __init__(self, **kwargs):
        super(Keras, self).__init__(**kwargs)
        self.ext = "ckpt"

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path)
        self.model = self.ml_model(model)

    def ml_model(self, model):
        return MLModel(fit_fn=model.fit, 
                        predictors=[model.predict],
                        load_fn=self.load_fn,
                        save_fn=model.save)

    def reformat(self, data, labels):
        data = self.transform_shape(data)
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return data, labels_m

    def train_kfolds(self, batch_size=10, num_steps=100, n_splits=None):
        from sklearn.model_selection import StratifiedKFold
        self.model = self.prepare_model_k()
        cv = StratifiedKFold(n_splits=n_splits)
        
        if self.dl is None:
            log.warning("The dataset dl is not set in the path, rebuilding...".format(
                self.__class__.__name__))
            dl = self.dataset.desfragment(dataset_path=settings["dataset_model_path"])
        else:
            dl = self.dl

        for k, (train, test) in enumerate(cv.split(dl.data, self.numerical_labels2classes(dl.labels)), 1):
            self.model.fit(dl.data.value[train], 
                dl.labels.value[train],
                nb_epoch=num_steps,
                batch_size=batch_size,
                shuffle="batch",
                validation_data=(dl.data.value[test], dl.labels.value[test]))
            print("fold ", k)

    def train(self, batch_size=0, num_steps=0, n_splits=None):
        if n_splits is not None:
            self.train_kfolds(batch_size=batch_size, num_steps=num_steps, n_splits=n_splits)
        else:
            self.model = self.prepare_model()
            self.model.fit(self.dataset.train_data, 
                self.dataset.train_labels,
                nb_epoch=num_steps,
                batch_size=batch_size,
                shuffle="batch",
                validation_data=(self.dataset.validation_data, self.dataset.validation_labels))
        self.save_model()

