import numpy as np
from sklearn.preprocessing import LabelEncoder
import time


def unique_dtypes(dtypes) -> np.ndarray:
    return np.unique([dtype.name for _, dtype in dtypes])


def labels2num(labels):
    le = LabelEncoder()
    le.fit(labels)
    return le


def isnamedtupleinstance(x):
    f = getattr(x, '_fields', None)
    shape = getattr(x, 'shape', None)
    return f is not None and shape is None  # x.__bases__[0] == tuple


def time2str(date):
    return time.strftime("%a, %d %b %Y %H:%M", time.gmtime(date))
