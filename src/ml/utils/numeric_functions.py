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
    """
    :type x: float
    :param x: number to transform

    :type b: float
    :param b: base to transform

    transform natural log to log base b
    """
    return np.log(x) / np.log(b)

def humanize_bytesize(size):
    """
    :type size: int
    :param size: number of bytes

    transforms an integer to a human readeable of the bytes.
    1024 -> 1.0 KB
    """
    if size == 0:
        return '0B'
    size_name = ('B', "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(logb(size, 1024)))
    p = np.power(1024, i)
    s = round(size / float(p), 2)
    return '{} {}'.format(s, size_name[i])

def expand_matrix_row(matrix, max_size, actual_size):
    """
    :type matrix: array
    :param matrix: NxM matrix

    :type max_size: int
    :param max_size: max size of new rows to add

    :type actual_size: int
    :param actual_size: number of rows to ignore

    add rows of zeros to the end of the matrix
    """
    return np.append(
        matrix, 
        np.zeros((max_size - actual_size, matrix.shape[1]), dtype=matrix.dtype),
        axis=0)

def expand_matrix_col(matrix, max_size, actual_size):
    """
    add columns of zeros to the right of the matrix
    """
    return np.append(
        matrix, 
        np.zeros((matrix.shape[0], max_size - actual_size), dtype=matrix.dtype),
        axis=1)


def expand_rows_cols(X, n_rows=2, n_cols=2):
    if len(X.shape) == 2:
        X = np.hstack((X, np.zeros((X.shape[0], n_cols))))
        X = np.vstack((X, np.zeros((n_rows, X.shape[1]))))
    return X
