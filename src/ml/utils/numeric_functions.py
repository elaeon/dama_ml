from itertools import izip
from operator import mul
import heapq as hq
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


def data_type(usize, total_size):
    from ml import fmtypes
    
    critery = [
        (fmtypes.BOOLEAN, usize == 2),
        (fmtypes.NANBOOLEAN, usize == 3),
        (fmtypes.ORDINAL, usize > 3 and total_size*.0001 > usize or total_size <= 1000),
        (fmtypes.DENSE, True)
    ]

    for fmtype, value in critery:
        if value is True:
            return fmtype


def features2rows(data):
    """
    :type labels: bool

    transforms a matrix of dim (n, m) to a matrix of dim (n*m, 2) where
    each row has the form [feature_column, feature_data]
    e.g
    [['a', 'b'], ['c', 'd'], ['e', 'f']] => [[0, 'a'], [0, 'c'], [0, 'e'], [1, 'b'], [1, 'd'], [1, 'f']]
    """
    ndata = np.empty((data.shape[0] * data.shape[1], 2), dtype=data.dtype)
    index = 0
    for ci, column in enumerate(data.T):
        for value in column:
            ndata[index] = np.asarray([ci, value])
            index += 1 
    return ndata


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


def features_fmtype(fmtypes, fmtype):
    from collections import defaultdict
    map_col = defaultdict(list)
    for ci, c_fmtype in enumerate(fmtypes):
        map_col[c_fmtype].append(ci)
    return map_col[fmtype.id]


def swap_noise(x, y, p=.15, cols=[1]):
    indexes_o = [int(round(i, 0)) for i in np.random.uniform(0, x.size-1, size=int(round(x.size * p, 0)))]
    indexes_d = [int(round(i, 0)) for i in np.random.uniform(0, x.size-1, size=int(round(x.size * p, 0)))]

    M = np.c_[x, y]
    vo = M[indexes_o, cols]
    vd = M[indexes_d, cols]
    for s0, s1, nv0, nv1 in zip(indexes_o, indexes_d, vo, vd):
        M[s1, cols] = nv0
        M[s0, cols] = nv1

    return M


def max_type(items):
    sizeof = [np.dtype(type(e)).num for e in items]
    types = [type(e) for e in items]
    if len(sizeof) == 0:
        return None
    v = max(zip(sizeof, types), key=lambda x: x[0])
    return v[1]


def wsr(stream, k):
    heap = []
    hkey = lambda w: -np.random.exponential(1./w)

    for item, weight in stream:
        if len(heap) < k:
            hq.heappush(heap, (hkey(weight), item))
        elif hkey(weight) > heap[0][0]:
            hq.heapreplace(heap, (hkey(weight), item))

    while len(heap) > 0:
        yield hq.heappop(heap)[1]


def wsrj(stream, k):
    reservoir = []
    hkey = lambda w: np.random.exponential(1./w)

    for item, weight in stream:
        if len(reservoir) < k:
            hq.heappush(reservoir, (hkey(weight), item))
        else:
            break

    if len(reservoir) == 0:
        return

    while True:
        T_w = reservoir[0][0]
        r = np.random.uniform(0, 1)
        X_w = np.log(r) / np.log(T_w)
        w_c = 0
        for item, weight in stream:
            if 0 < X_w - w_c <= weight:
                r2 = np.random.uniform(T_w, 1)
                k_i = r2**(1./weight)
                hq.heapreplace(reservoir, (k_i, item))                
                break
            w_c += weight
        
        if all(False for _ in stream):
            break

    while len(reservoir) > 0:
        yield hq.heappop(reservoir)[1]


def filter_sample(stream, label, col_index):
    if col_index is None:
        for e in stream:
            if e == label:
                yield e
    else:
        for row in stream:
            if row[col_index] == label:
                yield row


def downsample(stream, sampling, col_index, size):
    from ml.layers import IterLayer
    import numpy as np

    if col_index is not None:
        u_values, counter = np.unique(stream[:size, col_index], return_counts=True)
    else:
        u_values, counter = np.unique(stream[:size], return_counts=True)
    counter = dict(zip(u_values, counter))
    if len(counter) == 0:
        return IterLayer([], shape=(0,))

    sampling_n = {}
    for y, k in sampling.items():
        if 0 <= k <= 1:
            v = counter[y] * k
        elif 1 < k < counter[y]:
             v = k % counter[y]
        else:
            v = counter[y] % k
        sampling_n[y] = int(round(v, 0))

    iterators = []
    for y, k in sampling_n.items():
        iterators.append(IterLayer(filter_sample(stream[:size], y, col_index)).sample(k))
    return IterLayer.concat_n(iterators)
