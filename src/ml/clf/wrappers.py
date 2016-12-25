import os
import numpy as np
import tensorflow as tf
import logging

from sklearn.preprocessing import LabelEncoder
from ml.utils.config import get_settings
from ml.models import MLModel
settings = get_settings("ml")


logging.basicConfig()
log = logging.getLogger(__name__)
#np.random.seed(133)

class Measure(object):
    def __init__(self, predictions, labels, labels2classes_fn):
        self.labels = labels2classes_fn(labels)
        self.predictions = predictions
        self.average = "macro"
        self.labels2classes = labels2classes_fn

    def accuracy(self):
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.labels, self.labels2classes(self.predictions))

    #false positives
    def precision(self):
        from sklearn.metrics import precision_score
        return precision_score(self.labels, self.labels2classes(self.predictions), 
            average=self.average, pos_label=None)

    #false negatives
    def recall(self):
        from sklearn.metrics import recall_score
        return recall_score(self.labels, self.labels2classes(self.predictions), 
            average=self.average, pos_label=None)

    #weighted avg presicion and recall
    def f1(self):
        from sklearn.metrics import f1_score
        return f1_score(self.labels, self.labels2classes(self.predictions), 
            average=self.average, pos_label=None)

    def auc(self):
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(self.labels, self.labels2classes(self.predictions), 
                average=self.average)
        except ValueError:
            return None

    def confusion_matrix(self, base_labels=None):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.labels, self.transform(self.predictions), labels=base_labels)
        return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    def logloss(self):
        from sklearn.metrics import log_loss
        return log_loss(self.labels, self.predictions)


class ListMeasure(object):
    def __init__(self, headers=None, measures=None):
        if headers is None:
            headers = []
        if measures is None:
            measures = [[]]

        self.headers = headers
        self.measures = measures

    def add_measure(self, name, value, i=0):
        self.headers.append(name)
        self.measures[i].append(value)

    def get_measure(self, name):
        try:
            i = self.headers.index(name)
        except IndexError:
            i = 0

        for measure in self.measures:
            yield measure[i]

    def measures_to_dict(self):
        return {header: measures for header, measures in zip(self.headers[1:], self.measures[0][1:])}
            
    def print_scores(self, order_column="f1"):
        from ml.utils.order import order_table_print
        order_table_print(self.headers, self.measures, order_column)

    def print_matrix(self, labels):
        from tabulate import tabulate
        for name, measure in self.measures:
            print("******")
            print(name)
            print("******")
            print(tabulate(np.c_[labels.T, measure], list(labels)))

    def __add__(self, other):
        for hs, ho in zip(self.headers, other.headers):
            if hs != ho:
                raise Exception

        diff_len = abs(len(self.headers) - len(other.headers)) + 1
        if len(self.headers) < len(other.headers):
            headers = other.headers
            this_measures = [m +  ([None] * diff_len) for m in self.measures]
            other_measures = other.measures
        elif len(self.headers) > len(other.headers):
            headers = self.headers
            this_measures = self.measures
            other_measures = [m + ([None] * diff_len) for m in other.measures]
        else:
            headers = self.headers
            this_measures = self.measures
            other_measures = other.measures

        list_measure = ListMeasure(
            headers=headers, measures=this_measures+other_measures)
        return list_measure

    def calc_scores(self, name, predict, data, labels, labels2classes_fn=None, measures=None):
        if measures is None:
            measures = ["accuracy", "presicion", "recall", "f1", "auc", "logloss"]
        elif isinstance(measures, str):
            measures = measures.split(",")
        else:
            measures = ["logloss"]
        uncertain = "logloss" in measures
        predictions = np.asarray(list(
            predict(data, raw=uncertain, transform=False, chunk_size=258)))
        measure = Measure(predictions, labels, labels2classes_fn)
        self.add_measure("CLF", name)

        measure_class = []
        for measure_name in measures:
            measure_name = measure_name.strip()
            if hasattr(measure, measure_name):
                measure_class.append((measure_name, measure))

        for measure_name, measure in measure_class:
            self.add_measure(measure_name, getattr(measure, measure_name)())


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

    def save_meta(self, score=None):
        from ml.ds import save_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", self._metadata(score=score))

    def load_meta(self):
        from ml.ds import load_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            return load_metadata(path+".xmeta")

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


class BaseClassif(DataDrive):
    def __init__(self, model_name=None, dataset=None, check_point_path=None, 
                model_version=None, dataset_train_limit=None, info=True, 
                auto_load=True, group_name=None):
        self.model = None
        self.le = LabelEncoder()
        self.dataset_train_limit = dataset_train_limit
        self.print_info = info
        self.base_labels = None
        self._original_dataset_md5 = None
        super(BaseClassif, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)
        if auto_load is True:
            self.load_dataset(dataset)

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def module_cls_name(cls):
        return "{}.{}".format(cls.__module__, cls.__name__)

    def scores(self, measures=None):
        list_measure = ListMeasure()
        list_measure.calc_scores(self.__class__.__name__, 
                                self.predict, 
                                self.dataset.test_data, 
                                self.dataset.test_labels, 
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
                            self.numerical_labels2classes(self.dataset.test_labels))))
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
        return data.reshape(size, -1).astype(np.float32)

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
        else:
            return self.le.inverse_transform(self.position_index(label))

    def numerical_labels2classes(self, labels):
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            return self.le.inverse_transform(np.argmax(labels, axis=1))
        else:
            return self.le.inverse_transform(labels)

    def reformat_all(self):
        all_ds = np.concatenate((self.dataset.train_labels, 
            self.dataset.valid_labels, self.dataset.test_labels), axis=0)
        self.labels_encode(all_ds)
        self.dataset.train_data, self.dataset.train_labels = self.reformat(
            self.dataset.train_data, self.le.transform(self.dataset.train_labels))
        self.dataset.test_data, self.dataset.test_labels = self.reformat(
            self.dataset.test_data, self.le.transform(self.dataset.test_labels))
        self.num_features = self.dataset.test_data.shape[-1]

        if len(self.dataset.valid_labels) > 0:
            self.dataset.valid_data, self.dataset.valid_labels = self.reformat(
                self.dataset.valid_data, self.le.transform(self.dataset.valid_labels))

        if self.dataset_train_limit is not None:
            if self.dataset.train_data.shape[0] > self.dataset_train_limit:
                self.dataset.train_data = self.dataset.train_data[:self.dataset_train_limit]
                self.dataset.train_labels = self.dataset.train_labels[:self.dataset_train_limit]

            if self.dataset.valid_data.shape[0] > self.dataset_train_limit:
                self.dataset.valid_data = self.dataset.valid_data[:self.dataset_train_limit]
                self.dataset.valid_labels = self.dataset.valid_labels[:self.dataset_train_limit]

    def load_dataset(self, dataset):
        if dataset is None:
            self.set_dataset(self.get_dataset())
        else:
            self.set_dataset(dataset.copy())

    def set_dataset(self, dataset):
        self.dataset = dataset
        self._original_dataset_md5 = self.dataset.md5()
        self.reformat_all()

    def set_dataset_from_raw(self, train_data, test_data, valid_data, 
                            train_labels, test_labels, valid_labels, save=False, 
                            dataset_name=None):
        from ml.ds import DataSetBuilder
        data = {}
        data["train_dataset"] = train_data
        data["test_dataset"] = test_data
        data["valid_dataset"] = valid_data
        data["train_labels"] = self.numerical_labels2classes(train_labels)
        data["test_labels"] = self.numerical_labels2classes(test_labels)
        data["valid_labels"] = self.numerical_labels2classes(valid_labels)
        data['transforms'] = self.dataset.transforms.get_all_transforms()
        if self.dataset.processing_class is not None:
            data['preprocessing_class'] = self.dataset.processing_class.module_cls_name()
        else:
            data['preprocessing_class'] = None
        data["md5"] = None
        dataset_name = self.dataset.name if dataset_name is None else dataset_name
        dataset = DataSetBuilder.from_raw_to_ds(
            dataset_name,
            self.dataset.dataset_path,
            data,
            save=save)
        self.set_dataset(dataset)
        
    def chunk_iter(self, data, chunk_size=1, transform_fn=None, uncertain=False):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            data = np.asarray(list(chunk))
            size = data.shape[0]
            for prediction in self._predict(transform_fn(data, size), raw=uncertain):
                yield prediction

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        from ml.utils.seq import grouper_chunk
        if self.model is None:
            self.load_model()

        if not isinstance(chunk_size, int):
            log.warning("The parameter chunk_size must be an integer.")            
            log.warning("Chunk size is set to 1")
            chunk_size = 1

        if transform is True and chunk_size > 0:
            fn = lambda x, s: self.transform_shape(
                self.dataset.processing(x, init=False), size=s)
            return self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw)
        elif transform is True and chunk_size == 0:
            data = self.transform_shape(self.dataset.processing(data, init=False))
            return self._predict(data, raw=raw)
        elif transform is False and chunk_size > 0:
            fn = lambda x, s: self.transform_shape(x, size=s)
            return self.chunk_iter(data, chunk_size, transform_fn=fn, uncertain=raw)
        elif transform is False:
            return self._predict(data, raw=raw)
            

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

    def _metadata(self, score=None):
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "md5": self._original_dataset_md5, #not reformated dataset
                "group_name": self.group_name,
                "model_module": self.module_cls_name(),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "score": score}
        
    def get_dataset(self):
        from ml.ds import DataSetBuilder
        meta = self.load_meta()
        dataset = DataSetBuilder.load_dataset(
            meta["dataset_name"],
            dataset_path=meta["dataset_path"])
        self.group_name = meta.get('group_name', None)
        if meta.get('md5', None) != dataset.md5():
            log.warning("The dataset md5 is not equal to the model '{}'".format(
                self.__class__.__name__))
        return dataset

    def to_model(self):
        pass


class SKL(BaseClassif):
    def convert_label(self, label, raw=False):
        if raw is True:
            return (np.arange(self.num_labels) == label).astype(np.float32)
        else:
            return self.le.inverse_transform(self.position_index(label))

    def train(self, batch_size=0, num_steps=0):
        self.prepare_model()
        self.save_model()

    def save_model(self):
        from sklearn.externals import joblib
        if self.check_point_path is not None:
            path = self.make_model_file()
            joblib.dump(self.model, '{}.pkl'.format(path))
            list_measure = self.scores()
            self.save_meta(score=list_measure.measures_to_dict())

    def load_model(self):
        from sklearn.externals import joblib
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model = joblib.load('{}.pkl'.format(path))

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            yield self.convert_label(prediction, raw=raw)


class SKLP(SKL):
    def __init__(self, *args, **kwargs):
        super(SKLP, self).__init__(*args, **kwargs)

    def convert_label(self, label, raw=False):
        if raw is True:
            return label
        else:
            return self.le.inverse_transform(self.position_index(label))

    def _predict(self, data, raw=False):
        for prediction in self.model.predict_proba(data):
            yield self.convert_label(prediction, raw=raw)


class TFL(BaseClassif):

    def reformat(self, data, labels):
        data = self.transform_shape(data)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return data, labels_m

    def save_model(self):
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.save('{}.ckpt'.format(path))
            list_measure = self.scores()
            self.save_meta(score=list_measure.measures_to_dict())

    def load_model(self):
        self.prepare_model()
        if self.check_point_path is not None:
            path = self.make_model_file()
            #print("+++++++", path)
            self.model.load('{}.ckpt'.format(path))

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        with tf.Graph().as_default():
            return super(TFL, self).predict(data, raw=raw, transform=transform, chunk_size=chunk_size)

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            yield self.convert_label(np.asarray(prediction), raw=raw)


class Keras(BaseClassif):
    def load_fn(self, path):
        from keras.models import load_model
        net_model = load_model(path)
        self.model = MLModel(fit_fn=net_model.fit, 
                            predictors=[net_model.predict],
                            load_fn=self.load_fn,
                            save_fn=net_model.save)

    def preload_model(self):
        self.model = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_fn,
                            save_fn=None)

    def reformat(self, data, labels):
        data = self.transform_shape(data)
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return data, labels_m

    def save_model(self):
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.save('{}.ckpt'.format(path))
            list_measure = self.scores()
            self.save_meta(score=list_measure.measures_to_dict())

    def load_model(self):        
        self.preload_model()
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.load('{}.ckpt'.format(path))

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        return super(Keras, self).predict(data, raw=raw, transform=transform, chunk_size=chunk_size)

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            yield self.convert_label(np.asarray(prediction), raw=raw)
