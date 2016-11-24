from itertools import izip
from operator import mul
import numpy as np

def pearsoncc(X, Y):
    """ Compute Pearson Correlation Coefficient. """
    X = (X - X.mean(0)) / X.std(0)
    Y = (Y - Y.mean(0)) / Y.std(0)
    return (X * Y).mean()

def le(x, y):
    return x - y

def ge(x, y):
    return y - x

def geometric_mean(predictions, total):
    for row_prediction in izip(*predictions):
        mul_predictions = reduce(mul, row_prediction)
        yield np.power(mul_predictions, (1. / total))

def arithmetic_mean(predictions, total):
    for row_prediction in izip(*predictions):
        sum_prediction = sum(row_prediction)
        yield sum_prediction / float(total)

def discrete_weight(predictions, weights):
    for row_prediction in izip(*predictions):
        counter = {}
        for w, prediction in izip(weights, row_prediction):
            counter.setdefault(prediction, 0)
            counter[prediction] += w
        yield max(counter.items(), key=lambda x:x[1])[0]

def logb(x, b):
    return np.log(x) / np.log(b)

def humanize_bytesize(size):
    if size == 0:
        return '0B'
    size_name = ('B', "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(logb(size, 1024)))
    p = np.power(1024, i)
    s = round(size / float(p), 2)
    return '{} {}'.format(s, size_name[i])
