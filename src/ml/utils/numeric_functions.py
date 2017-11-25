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

    add rows with zeros to the end of the matrix
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
    """
    :type n_rows: int
    :param n_rows: number of rows to add

    :type n_cols: int
    :param n_cols: number of columns to add

    add an expecific number of rows and columns of zeros to the array X
    """
    if len(X.shape) == 2:
        X = np.hstack((X, np.zeros((X.shape[0], n_cols))))
        X = np.vstack((X, np.zeros((n_rows, X.shape[1]))))
    return X


def is_binary(array, include_null=True):
    if include_null is False:
        return np.count_nonzero((array != 0) & (array != 1)) == 0
    else:
        return np.all((array==0)|(array==1)|(np.isnan(array)))


def is_integer(array):
    mask = np.isnan(array)
    if any(mask):
        return all(np.equal(np.mod(array[~mask], 1), 0))
    else:
        return all(np.equal(np.mod(array, 1), 0))


def is_integer_if(array, card_size=4):
    return is_integer(array) and np.unique(array).size >= card_size


def index_if_type_row(array, fn, **params):
    return [i for i, row in enumerate(array) if fn(row, **params)]


def index_if_type_col(array, fn, **params):
    return [i for i, col in enumerate(array.T) if fn(col, **params)]


def unique_size(array):
    mask = np.isnan(array)
    array[mask] = -1
    return np.unique(array).size


def data_type(size, total_size):
    types = {
        "bin": "boolean",
        "nbn": "nan boolean",
        "ord": "ordinal",
        "car": "cardinal",
        "den": "dense"
    }
    critery = [
        ("bin", size == 2),
        ("nbn", size == 3),
        ("ord", size > 3 and total_size*.0001 > size),
        ("den", True)
    ]

    for name, value in critery:
        if value is True:
            return types[name]


def gini(actual, pred):
    assert(len(actual) == len(pred))
    actual = np.asarray(actual, dtype=np.float)
    n = actual.shape[0]
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 

def gini_normalized(a, p):
    if p.ndim == 2:
        p = p[:,1] #just pick class 1 if is a binary array
    return gini(a, p) / gini(a, a)


def missing(column):
    return (np.count_nonzero(np.isnan(column)) / float(column.size)) * 100


def zeros(column):
    return ((column.size - np.count_nonzero(column)) / float(column.size)) * 100
