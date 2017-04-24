import os
import numpy as np
import tensorflow as tf
import logging

from sklearn.preprocessing import LabelEncoder
from ml.utils.config import get_settings
from ml.models import MLModel
from ml.ds import DataSetBuilder, Data
from ml.clf.measures import ListMeasure


settings = get_settings("ml")
logging.basicConfig()
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(console)
#np.random.seed(133)


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

    def _metadata(self):
        pass

    def save_meta(self):
        from ml.ds import save_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", self._metadata())

    def edit_meta(self, key, value):
        from ml.ds import save_metadata
        meta = self.load_meta()
        meta[key] = value
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", meta)

    def load_meta(self):
        from ml.ds import load_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            return load_metadata(path+".xmeta")

    def get_model_path(self):
        model_name_v = self.get_model_name_v()
        path = os.path.join(self.check_point_path, self.__class__.__name__, model_name_v)
        return os.path.join(path, model_name_v)

    @classmethod
    def read_meta(self, data_name, path):        
        from ml.ds import load_metadata
        if data_name is not None:
            return load_metadata(path+".xmeta").get(data_name, None)
        return load_metadata(path+".xmeta")

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

    def destroy(self):
        """remove the dataset associated to the model and his checkpoints"""
        from ml.utils.files import rm
        self.dataset.destroy()
        #if hasattr(self, 'dl') and self.dl is not None:
        self.dl.destroy()
        rm(self.get_model_path()+"."+self.ext)
        rm(self.get_model_path()+".xmeta")

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)


class BaseClassif(DataDrive):
    def __init__(self, model_name=None, dataset=None, check_point_path=None, 
                model_version=None, dataset_train_limit=None, 
                autoload=True, group_name=None):
        self.model = None
        self.le = LabelEncoder()
        self.dataset_train_limit = dataset_train_limit
        self.base_labels = None
        self._original_dataset_md5 = None
        self.dataset = None
        self.dl = None
        self.ext = "ckpt.pkl"
        super(BaseClassif, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)
        if autoload is True:
            self.load_dataset(dataset)

    def scores(self, measures=None):
        list_measure = ListMeasure()
        list_measure.calc_scores(self.__class__.__name__, 
                                self.predict, 
                                self.dataset.test_data, 
                                self.dataset.test_labels[:], 
                                labels2classes_fn=self.numerical_labels2classes,
                                measures=measures)
        return list_measure

    def confusion_matrix(self):
        list_measure = ListMeasure()
        predictions = self.predict(self.dataset.test_data, raw=False, transform=False)
        measure = Measure(np.asarray(list(predictions)), self.dataset.test_labels, 
                        self.labels2classes_fn)
        list_measure.add_measure("CLF", self.__class__.__name__)
        list_measure.add_measure("CM", measure.confusion_matrix(base_labels=self.base_labels))
        return list_measure

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
        if size is None:
            size = data.shape[0]
        return data[:].reshape(size, -1)#.astype(np.float32)

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
            name=dataset.name+"_"+self.model_name+"_"+self.cls_name(),
            dataset_path=settings["dataset_model_path"],
            apply_transforms=True,
            compression_level=9,
            dtype=dataset.dtype,
            transforms=dataset.transforms,
            ltype='int',
            validator='',
            chunks=1000,
            rewrite=False)

        self.labels_encode(dataset.labels)
        log.info("Labels encode finished")
        if dsb.mode == "w":
            dsb._applied_transforms = dataset.apply_transforms
            train_data, train_labels = self.reformat(dataset.train_data, 
                                        self.le.transform(dataset.train_labels))
            test_data, test_labels = self.reformat(dataset.test_data, 
                                        self.le.transform(dataset.test_labels))
            validation_data, validation_labels = self.reformat(dataset.validation_data, 
                                        self.le.transform(dataset.validation_labels))
            dsb.build_dataset(train_data, train_labels, test_data, test_labels,
                            validation_data, validation_labels)
        
        dsb.close_reader()
        return dsb

    def load_dataset(self, dataset):
        if dataset is None:
            self.dataset = self.get_dataset()
        else:
            self.set_dataset(dataset)
        self.num_features = self.dataset.num_features()

    def set_dataset(self, dataset, auto=True):
        self._original_dataset_md5 = dataset.md5()
        if auto is True:
            self.dataset = self.reformat_all(dataset)
            if self.dataset.mode == "w":
                self.dl = self.dataset.desfragment(dataset_path=settings["dataset_model_path"])
                self.edit_meta("dl", self.dl.name)
            else:
                meta = self.load_meta()
                self.dl = Data.original_ds(name=meta["dl"], dataset_path=settings["dataset_model_path"])
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

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        from ml.layers import IterLayer

        if self.model is None:
            self.load_model()

        if not isinstance(chunk_size, int):
            log.warning("The parameter chunk_size must be an integer.")            
            log.warning("Chunk size is set to 1")
            chunk_size = 1

        if isinstance(data, IterLayer):
            def iter_(fn):
                for x in data:
                    yield IterLayer(self._predict(fn(x), raw=raw))

            if transform is True:
                fn = lambda x: self.transform_shape(self.dataset.processing(list(x), initial=False))
            else:
                fn = list
            return IterLayer(iter_(fn))
        else:
            if transform is True and chunk_size > 0:
                fn = lambda x, s: self.transform_shape(
                    self.dataset.processing(x, initial=False), size=s)
                return IterLayer(self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw))
            elif transform is True and chunk_size == 0:
                data = self.transform_shape(self.dataset.processing(data, initial=False))
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
        list_measure = self.scores()
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "md5": self.dataset.md5(),
                "original_ds_md5": self._original_dataset_md5, #not reformated dataset
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "score": list_measure.measures_to_dict(),
                "base_labels": list(self.base_labels),
                "dl": self.dl.name}
        
    def get_dataset(self):
        from ml.ds import DataSetBuilder, Data
        meta = self.load_meta()
        dataset = DataSetBuilder(name=meta["dataset_name"], dataset_path=meta["dataset_path"],
            apply_transforms=False)
        self._original_dataset_md5 = meta["original_ds_md5"]
        self.labels_encode(meta["base_labels"])
        self.group_name = meta.get('group_name', None)
        self.dl = Data.original_ds(name=meta["dl"], dataset_path=settings["dataset_model_path"])
        if meta.get('md5', None) != dataset.md5():
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

    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2):
        from sklearn.model_selection import StratifiedKFold
        self.model = self.prepare_model_k()
        cv = StratifiedKFold(n_splits=n_splits)
        
        if self.dl is None:
            log.warning("The dataset dl is not set in the path, rebuilding...".format(
                self.__class__.__name__))
            dl = self.dataset.desfragment(dataset_path=settings["dataset_model_path"])
        else:
            dl = self.dl

        for k, (train, test) in enumerate(cv.split(dl.data, dl.labels), 1):
            self.model.fit(dl.data.value[train], dl.labels.value[train])
            print("fold ", k)

    def train(self, batch_size=0, num_steps=0, n_splits=None):
        if n_splits is not None:
            self.train_kfolds(batch_size=batch_size, num_steps=num_steps, n_splits=n_splits)
        else:
            self.model = self.prepare_model()
        self.save_model()


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
    def __init__(self):
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

    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2):
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

