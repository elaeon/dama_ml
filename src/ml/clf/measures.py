import numpy as np
import logging

logging.basicConfig()
log = logging.getLogger(__name__)
#np.random.seed(133)


def greater_is_better_fn(reverse, uncertain):
    def view(fn):
        fn.reverse = reverse
        fn.uncertain = uncertain
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

    def __init__(self, predictions, labels, labels2classes=None, name=None):
        if labels2classes is None:
            self.labels2classes = lambda x: x 
        else:
            self.labels2classes = labels2classes
        self.labels = labels
        self.predictions = predictions
        self.name = name
        self.measures = []

    def add(self, measure, greater_is_better=True, uncertain=False):
        self.measures.append(greater_is_better_fn(greater_is_better, uncertain)(measure))

    def scores(self):
        if self.has_discrete():
            labels = self.labels2classes(self.labels)
            predictions_c = self.labels2classes(self.predictions)

        for measure in self.measures:
            if measure.uncertain:
                yield measure(self.labels, self.predictions)
            else:
                yield measure(labels, predictions_c)

    def has_uncertain(self):
        for measure in self.measures:
            if measure.uncertain is True:
                return True
        return False

    def has_discrete(self):
        for measure in self.measures:
            if measure.uncertain is False:
                return True
        return False

    def to_list(self):
        list_measure = ListMeasure(headers=[""]+[fn.__name__ for fn in self.measures],
                    order=[True]+[fn.reverse for fn in self.measures],
                    measures=[[self.name] + list(self.scores())])
        return list_measure


def accuracy(labels, predictions):
    """
    measure for correct predictions, true positives and true negatives.
    """
    from sklearn.metrics import accuracy_score
    return accuracy_score(labels, predictions)


def precision(labels, predictions):
    """
    measure for false positives predictions.
    """
    from sklearn.metrics import precision_score
    return precision_score(labels, predictions, average="macro", pos_label=None)


def recall(labels, predictions):
    """
    measure from false negatives predictions.
    """
    from sklearn.metrics import recall_score
    return recall_score(labels, predictions, average="macro", pos_label=None)


def f1(labels, predictions):
    """
    weighted average presicion and recall
    """
    from sklearn.metrics import f1_score
    return f1_score(labels, predictions, average="macro", pos_label=None)


def auc(labels, predictions):
    """
    area under the curve of the reciver operating characteristic, measure for 
    true positives rate and false positive rate
    """
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, predictions, average="macro")
    except ValueError:
        return None


def confusion_matrix(labels, predictions, base_labels=None):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions, labels=base_labels)
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


def logloss(labels, predictions):
    """
    accuracy by penalising false classifications
    """
    from sklearn.metrics import log_loss
    return log_loss(labels, predictions)


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
        for _, measure in enumerate(self.measures):
            print("******")
            print(measure[0])
            print("******")
            print(tabulate(np.c_[labels.T, measure[1]], list(labels)))

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
