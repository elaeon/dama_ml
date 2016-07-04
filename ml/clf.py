import os
import numpy as np
import tensorflow as tf

from sklearn import preprocessing

CHECK_POINT_PATH = "/home/sc/data/face_recog/"
#np.random.seed(133)
    
class Measure(object):
    def __init__(self, predictions, labels):
        if len(predictions.shape) > 1:
            self.transform = lambda x: np.argmax(x, 1)
        else:
            self.transform = lambda x: x
        
        self.labels = labels
        self.predictions = predictions
        self.average = "macro"

    def accuracy(self):
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.labels, self.transform(self.predictions))

    #false positives
    def precision(self):
        from sklearn.metrics import precision_score
        return precision_score(self.labels, self.transform(self.predictions), 
            average=self.average, pos_label=None)

    #false negatives
    def recall(self):
        from sklearn.metrics import recall_score
        return recall_score(self.labels, self.transform(self.predictions), 
            average=self.average, pos_label=None)

    #weighted avg presicion and recall
    def f1(self):
        from sklearn.metrics import f1_score
        return f1_score(self.labels, self.transform(self.predictions), 
            average=self.average, pos_label=None)

    def logloss(self):
        from sklearn.metrics import log_loss
        return log_loss(self.labels, self.predictions)

    def print_all(self):
        print("#############")
        print("Accuracy: {}%".format(self.accuracy()*100))
        print("Precision: {}%".format(self.precision()*100))
        print("Recall: {}%".format(self.recall()*100))
        print("F1: {}%".format(self.f1()*100))
        print("#############")

class BaseClassif(object):
    def __init__(self, dataset, check_point_path=CHECK_POINT_PATH, pprint=True):
        self.model = None
        self.pprint = pprint
        self.le = preprocessing.LabelEncoder()
        self.load_dataset(dataset)
        self.check_point_path = check_point_path
        self.check_point = check_point_path + self.__class__.__name__ + "/"

    def detector_test_dataset(self, raw=False):
        predictions = self.predict(self.dataset.test_dataset, raw=raw)
        measure = Measure(np.asarray(list(predictions)), np.asarray([self.convert_label(label) 
            for label in self.dataset.test_labels]))
        return self.__class__.__name__, measure

    def print_score(self):
        dt = ClassifTest(logloss=True)
        dt.classif_test(self, "f1")

    def only_is(self, op):
        predictions = list(self.predict(self.dataset.test_dataset, raw=False))
        labels = [self.convert_label(label) for label in self.dataset.test_labels]
        data = zip(*filter(lambda x: op(x[1], x[2]), zip(self.dataset.test_dataset, predictions, labels)))
        return np.array(data[0]), data[1], data[2]

    def erroneous_clf(self):
        import operator
        return self.only_is(operator.ne)

    def correct_clf(self):
        import operator
        return self.only_is(operator.eq)

    def reformat(self, dataset, labels):
        dataset = self.transform_img(dataset)
        return dataset, labels

    def transform_img(self, img):
        return img.reshape(img.shape[0], -1).astype(np.float32)

    def labels_encode(self, labels):
        self.le.fit(labels)
        self.num_labels = self.le.classes_.shape[0]

    def position_index(self, label):
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
        self.dataset.train_dataset, self.dataset.train_labels = self.reformat(
            self.dataset.train_dataset, self.le.transform(self.dataset.train_labels))
        self.dataset.test_dataset, self.dataset.test_labels = self.reformat(
            self.dataset.test_dataset, self.le.transform(self.dataset.test_labels))
        self.num_features = self.dataset.test_dataset.shape[-1]

        if len(self.dataset.valid_labels) > 0:
            self.dataset.valid_dataset, self.dataset.valid_labels = self.reformat(
                self.dataset.valid_dataset, self.le.transform(self.dataset.valid_labels))

        if self.pprint:
            print('RF-Training set', self.dataset.train_dataset.shape, self.dataset.train_labels.shape)
            if self.dataset.valid_dataset.shape[0] > 0:
                print('RF-Validation set', self.dataset.valid_dataset.shape,
                    self.dataset.valid_labels.shape)
            print('RF-Test set', self.dataset.test_dataset.shape, self.dataset.test_labels.shape)

    def accuracy(self, predictions, labels):
        measure = Measure(predictions, labels)
        if self.pprint:
            measure.print_all()
        return measure.accuracy()

    def load_dataset(self, dataset):
        from ml.ds import DataSetBuilder
        self.dataset = DataSetBuilder(dataset.name)
        self.dataset.image_size = dataset.image_size
        self.dataset.train_dataset = dataset.train_dataset
        self.dataset.train_labels = dataset.train_labels
        self.dataset.valid_dataset = dataset.valid_dataset
        self.dataset.valid_labels = dataset.valid_labels
        self.dataset.test_dataset = dataset.test_dataset
        self.dataset.test_labels = dataset.test_labels
        self.reformat_all()

    def predict(self, data, raw=False):
        return self._predict(data, raw=raw)


class SKL(BaseClassif):
    def train(self, batch_size=0, num_steps=0):
        self.prepare_model()
        predictions = self.model.predict(self.dataset.test_dataset)
        score = self.accuracy(predictions, self.dataset.test_labels)
        self.save_model()
        return score

    def save_model(self):
        from sklearn.externals import joblib
        if not os.path.exists(self.check_point):
            os.makedirs(self.check_point)
        if not os.path.exists(self.check_point + self.dataset.name + "/"):
            os.makedirs(self.check_point + self.dataset.name + "/")
        joblib.dump(self.model, '{}.pkl'.format(
            self.check_point+self.dataset.name+"/"+self.dataset.name))

    def load_model(self):
        from sklearn.externals import joblib
        self.model = joblib.load('{}.pkl'.format(
            self.check_point+self.dataset.name+"/"+self.dataset.name))

    def _predict(self, data, raw=False):
        if self.model is None:
            self.load_model()

        if isinstance(data, list):
            data = np.asarray(data)

        data = preprocessing.scale(data)
        for prediction in self.model.predict(self.transform_img(data)):
            yield self.convert_label(prediction)


class SVC(SKL):
    def prepare_model(self):
        from sklearn import svm
        reg = svm.LinearSVC(C=1, max_iter=1000)
        reg = reg.fit(self.dataset.train_dataset, self.dataset.train_labels)
        self.model = reg


class SKLP(SKL):
    def position_index(self, label):
        return np.argmax(label)

    def _predict(self, data, raw=False):
        if self.model is None:
            self.load_model()

        if isinstance(data, list):
            data = np.asarray(data)

        data = preprocessing.scale(data)
        for prediction in self.model.predict_proba(self.transform_img(data)):
            yield self.convert_label(prediction, raw=raw)


class ClassifTest(object):
    def __init__(self, logloss=False):
        self.logloss = logloss
        self.headers = ["CLF", "Precision", "Recall", "F1"]
        if logloss is True:
            self.headers = self.headers + ["logloss"]

    def dataset_test(self, classifs, dataset, order_column):
        from utils.order import order_table_print
        table = []
        print("DATASET", dataset.name)
        for classif_name in classifs:
            classif = classifs[classif_name]["name"]
            params = classifs[classif_name]["params"]
            clf = classif(dataset, **params)
            name_clf, measure = clf.detector_test_dataset(raw=self.logloss)
            if self.logloss:
                table.append((name_clf, measure.precision(), measure.recall(), 
                    measure.f1(), measure.logloss()))
            else:
                table.append((name_clf, measure.precision(), measure.recall(), measure.f1()))
        order_table_print(self.headers, table, order_column)

    def classif_test(self, clf, order_column):
        from utils.order import order_table_print
        table = []
        name_clf, measure = clf.detector_test_dataset(raw=self.logloss)
        if self.logloss:
            table = [(name_clf, measure.precision(), measure.recall(), 
                measure.f1(), measure.logloss())]
        else:
            table = [(name_clf, measure.precision(), measure.recall(), measure.f1())]
        order_table_print(self.headers, table, order_column)
