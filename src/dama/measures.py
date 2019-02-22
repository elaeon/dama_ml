import numpy as np
from dama.utils.order import order_table
from dama.utils.logger import log_config


log = log_config(__name__)


def greater_is_better_fn(reverse, output):
    def view(fn):
        fn.reverse = reverse
        fn.output = output
        return fn
    return view


class MeasureBase(object):
    __slots__ = ['name', 'measures']

    def __init__(self, name: str = None):
        self.name = name
        self.measures = []

    def __iter__(self) -> iter:
        return iter(self.measures)

    def scores(self):
        return []

    def add(self, measure, greater_is_better: bool = True, output=None) -> None:
        self.measures.append(greater_is_better_fn(greater_is_better, output)(measure))

    def outputs(self):
        groups = {}
        for measure in self.measures:
            groups[str(measure.output)] = measure.output
        return groups.values()

    def to_list(self) -> 'ListMeasure':
        list_measure = ListMeasure(headers=[""]+[fn.__name__ for fn in self.measures],
                                   order=[True]+[fn.reverse for fn in self.measures],
                                   measures=[[self.name] + list(self.scores())])
        return list_measure

    @staticmethod
    def make_metrics(measure_cls, measures: str = None, discrete: bool = True):
        if measures is None and discrete is True:
            measure_cls.add(accuracy, greater_is_better=True, output='discrete')
            measure_cls.add(precision, greater_is_better=True, output='discrete')
            measure_cls.add(recall, greater_is_better=True, output='discrete')
            measure_cls.add(f1, greater_is_better=True, output='discrete')
            measure_cls.add(auc, greater_is_better=True, output='discrete')
            measure_cls.add(logloss, greater_is_better=False, output='n_dim')
        elif measures is None and discrete is False:
            measure_cls.add(mse, greater_is_better=False, output='n_dim')
            measure_cls.add(msle, greater_is_better=False, output='n_dim')
            measure_cls.add(gini_normalized, greater_is_better=True, output='n_dim')
        elif isinstance(measures, str):
            import sys
            m = sys.modules['ml.measures']
            if hasattr(m, measures) and measures != 'logloss':
                measure_cls.add(getattr(m, measures), greater_is_better=True,
                                output='discrete')
            elif measures == 'logloss':
                measure_cls.add(getattr(m, measures), greater_is_better=False,
                                output='n_dim')
        return measure_cls


class Measure(MeasureBase):
    __slots__ = ['score', 'name', 'measures']

    def __init__(self, name: str = None):
        super(Measure, self).__init__(name=name)
        self.score = {}

    def update(self, predictions, target) -> None:
        for measure_fn in self:
            self.update_fn(predictions, target, measure_fn)

    def update_fn(self, predictions, target, measure_fn) -> None:
        self.score[measure_fn.__name__] = measure_fn(target, predictions)

    def scores(self):
        for measure in self.measures:
            yield self.score[measure.__name__]

    def make_metrics(self, measures=None, discrete: bool = True) -> 'Measure':
        return MeasureBase.make_metrics(self, measures=measures, discrete=discrete)


class MeasureBatch(MeasureBase):
    __slots__ = ['score', 'name', 'measures', 'batch_size']

    def __init__(self, name: str = None, batch_size: int = 0):
        super(MeasureBatch, self).__init__(name=name)
        self.score = {}
        self.batch_size = batch_size

    def update(self, predictions, target) -> None:
        for measure_fn in self:
            self.update_fn(predictions, target, measure_fn)

    def update_fn(self, predictions, target, measure_fn) -> None:
        log.debug("Set measure {}".format(measure_fn.__name__))
        target_ = target.batch.to_ndarray()
        predictions_ = predictions.batch.to_ndarray()
        try:
            self.score[measure_fn.__name__][0] += measure_fn(target_, predictions_) * \
                                                  (len(predictions_) / self.batch_size)
            self.score[measure_fn.__name__][1] += (len(predictions_) / self.batch_size)
        except KeyError:
            self.score[measure_fn.__name__] = [measure_fn(target_, predictions_), 1]

    def scores(self):
        for measure in self.measures:
            value, size = self.score[measure.__name__]
            yield value / size

    def make_metrics(self, measures=None, discrete: bool = True) -> 'MeasureBatch':
        return MeasureBase.make_metrics(self, measures=measures, discrete=discrete)


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


def mse(labels, predictions):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(labels, predictions)


def msle(labels, predicitions):
    from sklearn.metrics import mean_squared_log_error
    return mean_squared_log_error(labels, predicitions)


def gini(actual, pred):
    assert (len(actual) == len(pred))
    actual = np.asarray(actual, dtype=np.float)
    n = actual.shape[0]
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    gini_sum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return gini_sum / n


def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:, 1]  # just pick class 1 if is a binary array
    return gini(a, p) / gini(a, a)


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

    def add_measure(self, name: str, value, i: int = 0, reverse: bool = False):
        self.headers.append(name)
        try:
            self.measures[i].append(value)
        except IndexError:
            self.measures.append([])
            self.measures[len(self.measures) - 1].append(value)
        self.order.append(reverse)

    def get_measure(self, name: str):
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
             
    @classmethod
    def dict_to_measures(cls, data_dict):
        headers = data_dict.keys()
        measures = [[v["values"][0] for k, v in data_dict.items()]]
        order = [v["reverse"] for k, v in data_dict.items()]
        return ListMeasure(headers=headers, measures=measures, order=order)

    def to_tabulate(self, order_column: str = None, limit=None):
        self.drop_empty_columns()
        return order_table(self.headers, self.measures, order_column,
                           natural_order=self.order, limit=limit)

    def __str__(self):
        return self.to_tabulate()

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
                raise Exception("Could not add new headers to the table")

        if len(self.headers) == 0:
            headers = other.headers
            this_measures = []
            other_measures = other.measures
            order = other.order
        elif len(other.headers) == 0:
            headers = self.headers
            this_measures = self.measures
            other_measures = []
            order = self.order
        else:
            diff_len = abs(len(self.headers) - len(other.headers))
            if len(self.headers) < len(other.headers):
                headers = other.headers
                this_measures = [m + ([""] * diff_len) for m in self.measures]
                other_measures = other.measures
                order = other.order
            elif len(self.headers) > len(other.headers):
                headers = self.headers
                this_measures = self.measures
                other_measures = [m + ([""] * diff_len) for m in other.measures]
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
