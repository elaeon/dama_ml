import os
import numpy as np
import tensorflow as tf
import logging

from sklearn.preprocessing import LabelEncoder

logging.basicConfig()
log = logging.getLogger(__name__)
#np.random.seed(133)

#Discrete measure
class DMeasure(object):
    def __init__(self, predictions, labels):
        if len(predictions.shape) > 1:
            self.transform = lambda x: np.argmax(x, 1)
        else:
            self.transform = lambda x: x
        
        self.labels = labels
        self.predictions = self.transform(predictions)
        self.average = "macro"

    def accuracy(self):
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.labels, self.predictions)

    #false positives
    def precision(self):
        from sklearn.metrics import precision_score
        return precision_score(self.labels, self.predictions, 
            average=self.average, pos_label=None)

    #false negatives
    def recall(self):
        from sklearn.metrics import recall_score
        return recall_score(self.labels, self.predictions, 
            average=self.average, pos_label=None)

    #weighted avg presicion and recall
    def f1(self):
        from sklearn.metrics import f1_score
        return f1_score(self.labels, self.predictions, 
            average=self.average, pos_label=None)

    def auc(self):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(self.labels, self.predictions, 
            average=self.average)

    def confusion_matrix(self, base_labels=None):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.labels, self.predictions, labels=base_labels)
        return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#Continuos measure
class CMeasure(object):
    def __init__(self, predictions, labels):
        self.labels = labels
        self.predictions = predictions

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

    def print_scores(self, order_column="f1"):
        from utils.order import order_table_print
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


class Grid(object):
    def __init__(self, classifs, model_name=None, dataset=None, 
            check_point_path=None, model_version=None):
        self.model = None
        self.model_name = model_name
        self.check_point_path = check_point_path
        self.model_version = model_version
        self.dataset = dataset
        self.classifs = classifs
        self.params = {}
        self.classifs_reader = None
        
    def load_models(self):
        for index, classif in enumerate(self.classifs):
            yield self.load_model(classif, info=False)
    
    def load_model(self, model, info=True):
        return model(dataset=self.dataset, 
                model_name=self.model_name, 
                model_version=self.model_version, 
                check_point_path=self.check_point_path,
                info=info,
                **self.get_params(model.cls_name_simple()))

    def train(self, batch_size=128, num_steps=1):
        for classif in self.load_models():
            classif.train(batch_size=batch_size, num_steps=num_steps)
    
    def scores(self):
        from operator import add
        list_measure = reduce(add, (classif.calc_scores() for classif in self.load_models()))
        return list_measure

    def confusion_matrix(self):
        from operator import add
        list_measure = reduce(add, (classif.confusion_matrix() for classif in self.load_models()))
        self.classifs_reader = self.load_models()
        classif = self.classifs_reader.next()
        list_measure.print_matrix(classif.base_labels)

    def add_params(self, model_cls, **params):
        self.params.setdefault(model_cls, params)
        self.params[model_cls].update(params)

    def get_params(self, model_cls):
        return self.params.get(model_cls, {})

    # select the best clf from the list of models
    def best_predictor(self, measure_name="logloss", operator=None):
        list_measure = self.scores()
        column_measures = list_measure.get_measure(measure_name)
        base_value = column_measures.next()
        best = 0
        for index, value in enumerate(column_measures, 1):
            if base_value is None:
                base_value = value
            elif value is not None and operator(value, base_value):
                base_value = value
                best = index
        
        return self.load_model(self.classifs[best], info=True)


class BaseClassif(object):
    def __init__(self, model_name=None, dataset=None, 
            check_point_path=None, model_version=None,
            dataset_train_limit=None,
            info=True):
        self.model = None
        self.model_name = model_name
        self.le = LabelEncoder()
        self.check_point_path = check_point_path
        self.model_version = model_version
        self.has_uncertain = False
        self.dataset_train_limit = dataset_train_limit
        self.print_info = info
        self.base_labels = None
        self._original_dataset_md5 = None
        self.load_dataset(dataset)

    @classmethod
    def cls_name(cls):
        return cls.__name__

    @classmethod
    def cls_name_simple(cls):
        try:
            return cls.__name__.split(".").pop()
        except IndexError:
            return cls.__name__

    def calc_scores(self, measures=None):
        list_measure = ListMeasure()
        predictions = self.predict(self.dataset.test_data, raw=False, transform=False)
        measure = DMeasure(np.asarray(list(predictions)), 
            np.asarray([self.convert_label(label, raw=False)
            for label in self.dataset.test_labels]))
        list_measure.add_measure("CLF", self.__class__.__name__)

        if measures is None:
            measures = ["accuracy", "presicion", "recall", "f1", "auc", "logloss"]
        elif isinstance(measures, str):
            measures = measures.split(",")
        else:
            measures = []

        measure_class = []
        for measure_name in measures:
            measure_name = measure_name.strip()
            if hasattr(measure, measure_name):
                measure_class.append((measure_name, measure))

        if self.has_uncertain and "logloss" in measures:
            predictions = self.predict(self.dataset.test_data, raw=True, transform=False)
            cmeasure = CMeasure(np.asarray(list(predictions)), self.dataset.test_labels)
            measure_class.append(("logloss", cmeasure))

        for measure_name, measure in measure_class:
            list_measure.add_measure(measure_name, getattr(measure, measure_name)())

        return list_measure

    def confusion_matrix(self):
        list_measure = ListMeasure()
        predictions = self.predict(self.dataset.test_data, raw=False, transform=False)
        measure = DMeasure(np.asarray(list(predictions)), 
            np.asarray([self.convert_label(label, raw=False)
            for label in self.dataset.test_labels]))
        list_measure.add_measure("CLF", self.__class__.__name__)
        list_measure.add_measure("CM", measure.confusion_matrix(base_labels=self.base_labels))
        return list_measure

    def scores(self, measures=None, order_column="f1"):
        list_measure = self.calc_scores(measures=measures)
        list_measure.print_scores(order_column=order_column)

    def only_is(self, op):
        predictions = list(self.predict(self.dataset.test_data, raw=False, transform=False))
        labels = [self.convert_label(label) for label in self.dataset.test_labels]
        data = zip(*filter(lambda x: op(x[1], x[2]), 
            zip(self.dataset.test_data, predictions, labels)))
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

    def transform_shape(self, data):
        return data.reshape(data.shape[0], -1).astype(np.float32)

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
            self.dataset = self.get_dataset()
        else:
            self.dataset = dataset.copy()
            self.model_name = self.dataset.name
        self._original_dataset_md5 = self.dataset.md5()
        self.reformat_all()

    def predict(self, data, raw=False, transform=True):
        if self.model is None:
            self.load_model()

        if transform is True:
            ndata = [self.dataset.processing(datum, 'global') for datum in data]
            data = self.transform_shape(np.asarray(ndata))
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

    def rebuild_validation_from_errors(self, dataset, valid_size=.1, test_data_labels=None):
        valid_size = dataset.valid_size if valid_size is None else valid_size
        if dataset.is_binary():
            if dataset.test_folder_path is not None or test_data_labels is None:
                test_data, test_labels = dataset.test_data, dataset.test_labels
                predictions = self.predict(self.dataset.test_data, raw=True, transform=False)
            else:
                test_data, test_labels = test_data_labels
                predictions = self.predict(test_data, raw=True, transform=True)
                test_data = dataset.processing(test_data, 'global')

            ndataset = dataset.copy()
            ndataset.train_data = np.concatenate(
                (dataset.train_data, dataset.valid_data), axis=0)
            ndataset.train_labels = np.concatenate(
                (dataset.train_labels, dataset.valid_labels), axis=0)
            ndataset.valid_data, ndataset.valid_labels, pred_index = self._pred_erros(
                predictions, test_data, test_labels, valid_size=valid_size)
            indexes = sorted(np.array([index for index, _ in pred_index]))            
            ndataset.test_data = np.delete(test_data, indexes, axis=0)
            ndataset.test_labels = np.delete(test_labels, indexes)
            ndataset.info()
            return ndataset
        else:
            raise Exception

    def retrain(self, dataset, batch_size=10, num_steps=1000):
        self.dataset = dataset
        self.reformat_all()
        self.train(batch_size=batch_size, num_steps=num_steps)

    def train2steps(self, dataset, valid_size=.1, batch_size=10, num_steps=1000, test_data_labels=None):
        self.train(batch_size=batch_size, num_steps=num_steps)
        dataset_v = self.rebuild_validation_from_errors(dataset, 
            valid_size=valid_size, test_data_labels=test_data_labels)
        self.retrain(dataset_v, batch_size=batch_size, num_steps=num_steps)

    def get_model_name_v(self):
        if self.model_version is None:
            import datetime
            id_ = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
        else:
            id_ = self.model_version
        return "{}.{}".format(self.model_name, id_)

    def make_model_file(self, check=True):
        model_name_v = self.get_model_name_v()
        check_point = os.path.join(self.check_point_path, self.__class__.__name__)
        if check is True:
            if not os.path.exists(check_point):
                os.makedirs(check_point)

            destination = os.path.join(check_point, model_name_v)
            if not os.path.exists(destination):
                os.makedirs(destination)
            
        return os.path.join(check_point, model_name_v, model_name_v)

    def _metadata(self):
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "md5": self._original_dataset_md5, #not reformated dataset
                "transforms": self.dataset.transforms.get_all_transforms(),
                "preprocessing_class": self.dataset.processing_class.module_cls_name()}

    def save_meta(self):
        from ml.ds import save_metadata
        if self.check_point_path is not None:
            path = self.make_model_file()
            save_metadata(path+".xmeta", self._metadata())

    def load_meta(self):
        from ml.ds import load_metadata
        if self.check_point_path is not None:
            path = self.make_model_file(check=False)
            return load_metadata(path+".xmeta")
        
    def get_dataset(self):
        from ml.ds import DataSetBuilder
        meta = self.load_meta()
        dataset = DataSetBuilder.load_dataset(
            meta["dataset_name"],
            dataset_path=meta["dataset_path"],
            info=self.print_info)
        if meta.get('md5', None) != dataset.md5():
            log.warning("The dataset md5 is not equal to the model '{}'".format(
                self.__class__.__name__))
        elif meta.get('transforms', None) != dataset.transforms.get_all_transforms():
            log.warning(
                "The filters in the dataset are distinct to training model '{}'".format(self.__class__.__name__))
        return dataset


class SKL(BaseClassif):
    def train(self, batch_size=0, num_steps=0):
        self.prepare_model()
        self.save_model()

    def save_model(self):
        from sklearn.externals import joblib
        if self.check_point_path is not None:
            path = self.make_model_file()
            joblib.dump(self.model, '{}.pkl'.format(path))
            self.save_meta()

    def load_model(self):
        from sklearn.externals import joblib
        if self.check_point_path is not None:
            path = self.make_model_file(check=False)
            self.model = joblib.load('{}.pkl'.format(path))

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            yield self.convert_label(prediction)


class SKLP(SKL):
    def __init__(self, *args, **kwargs):
        super(SKLP, self).__init__(*args, **kwargs) 
        self.has_uncertain = True

    def _predict(self, data, raw=False):
        for prediction in self.model.predict_proba(data):
            yield self.convert_label(prediction, raw=raw)


class TFL(BaseClassif):
    def __init__(self, **kwargs):
        super(TFL, self).__init__(**kwargs)
        self.has_uncertain = True

    def reformat(self, data, labels):
        data = self.transform_shape(data)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return data, labels_m

    def save_model(self):
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.save('{}.ckpt'.format(path))
            self.save_meta()

    def load_model(self):
        import tflearn
        self.prepare_model()
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.model.load('{}.ckpt'.format(path))

    def predict(self, data, raw=False, transform=True):
        with tf.Graph().as_default():
            if self.model is None:
                self.load_model()

            if transform is True:
                data = [self.dataset.processing(datum, 'global') for datum in data]
                data = self.transform_shape(np.asarray(data))

            return self._predict(data, raw=raw)

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            yield self.convert_label(prediction, raw=raw)

