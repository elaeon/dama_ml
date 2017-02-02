import numpy as np
import logging

logging.basicConfig()
log = logging.getLogger(__name__)
#np.random.seed(133)


def natural_order(reverse):
    def view(fn):
        fn.reverse = reverse
        return fn
    return view


class Measure(object):
    """
    For measure the results of the predictors, distincts measures are defined in this class
    
    :type predictions: array
    :param predictions: array of predictions

    :type labels: array
    :param labels: array of correct labels of type float for compare with the predictions

    :type labels2classes_fn: function
    :param labels2classes_fn: function for transform the labels to classes
    """

    def __init__(self, predictions, labels, labels2classes_fn):
        self.labels = labels2classes_fn(labels)
        self.predictions = predictions
        self.average = "macro"
        self.labels2classes = labels2classes_fn

    @natural_order(True)
    def accuracy(self):
        """
        measure for correct predictions, true positives and true negatives.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.labels, self.labels2classes(self.predictions))

    @natural_order(True)
    def precision(self):
        """
        measure for false positives predictions.
        """
        from sklearn.metrics import precision_score
        return precision_score(self.labels, self.labels2classes(self.predictions), 
            average=self.average, pos_label=None)

    @natural_order(True)
    def recall(self):
        """
        measure from false negatives predictions.
        """
        from sklearn.metrics import recall_score
        return recall_score(self.labels, self.labels2classes(self.predictions), 
            average=self.average, pos_label=None)

    @natural_order(True)
    def f1(self):
        """
        weighted average presicion and recall
        """
        from sklearn.metrics import f1_score
        return f1_score(self.labels, self.labels2classes(self.predictions), 
            average=self.average, pos_label=None)

    @natural_order(True)
    def auc(self):
        """
        area under the curve of the reciver operating characteristic, measure for 
        true positives rate and false positive rate
        """
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

    @natural_order(False)
    def logloss(self):
        """
        accuracy by penalising false classifications
        """
        from sklearn.metrics import log_loss
        return log_loss(self.labels, self.predictions)


class ListMeasure(object):
    """
    Class for save distincts measures

    :type headers: list
    :param headers: lists of headers

    :type measures: list
    :param measures: list of values

    list_measure = ListMeasure(headers=["classif", "f1"], measures=[["test", 0.5], ["test2", 0.6]])
    """

    def __init__(self, headers=None, measures=None, order=None):
        if headers is None:
            headers = []
        if measures is None:
            measures = [[]]
        if order is None:
            order = [False for _ in headers]

        self.headers = headers
        self.measures = measures
        self.order = order

    def add_measure(self, name, value, i=0, reverse=False):
        """
        :type name: string
        :param name: column name

        :type value: float
        :param value: value to add
        """
        self.headers.append(name)
        try:
            self.measures[i].append(value)
        except IndexError:
            self.measures.append([])
            self.measures[len(self.measures) - 1].append(value)
        self.order.append(reverse)

    def get_measure(self, name):
        """
        :type name: string
        :param name: by name of the column you can get his values. 
        """
        return self.measures_to_dict().get(name, None)

    def measures_to_dict(self):
        """
        convert the matrix to a dictionary
        """
        from collections import defaultdict
        measures = defaultdict(dict)
        for i, header in enumerate(self.headers, 0):
            measures[header] = {"values": [], "reverse": self.order[i]}
            for measure in self.measures:
                measures[header]["values"].append(measure[i])
        return measures
             
    def print_scores(self, order_column=None):
        """
        :type order_column: string
        :param order_column: order the matrix by the order_column name that you pass
        
        :type reverse: bool
        :param reverse: if False the order is ASC else DESC

        print the matrix
        """
        from ml.utils.order import order_table_print
        self.drop_empty_columns()
        order_table_print(self.headers, self.measures, order_column, natural_order=self.order)

    def empty_columns(self):
        """
        return a set of indexes of empty columns
        """
        empty_cols = {}
        for row in self.measures:
            for i, col in enumerate(row):
                if col is None or col == '':
                    empty_cols.setdefault(i, 0)
                    empty_cols[i] += 1

        return set([col for col, counter in empty_cols.items() if counter == len(self.measures)])        

    def drop_empty_columns(self):
        """
        drop empty columns
        """
        empty_columns = self.empty_columns()
        for counter, c in enumerate(empty_columns):
            del self.headers[c-counter]
            del self.order[c-counter]
            for row in self.measures:
                del row[c-counter]

    def print_matrix(self, labels):
        from tabulate import tabulate
        for name, measure in enumerate(self.measures):
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
            order = other.order
        elif len(self.headers) > len(other.headers):
            headers = self.headers
            this_measures = self.measures
            other_measures = [m + ([None] * diff_len) for m in other.measures]
            order = self.order
        else:
            headers = self.headers
            this_measures = self.measures
            other_measures = other.measures
            order = self.order

        list_measure = ListMeasure(
            headers=headers, 
            measures=this_measures+other_measures,
            order=order)
        return list_measure

    def calc_scores(self, name, predict, data, labels, labels2classes_fn=None, measures=None):
        if measures is None:
            measures = ["accuracy", "precision", "recall", "f1", "auc", "logloss"]
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
            fn = getattr(measure, measure_name)
            self.add_measure(measure_name, fn(), reverse=fn.reverse)
