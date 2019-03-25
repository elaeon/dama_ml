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


def str2slice(str_slice: str) -> slice:
    if str_slice is not None:
        elems = str_slice.split(":")
        stop = None
        if len(elems) > 1:
            try:
                start = int(elems[0])
            except ValueError:
                start = 0
            if elems[1] != '':
                stop = int(elems[1])
        else:
            start = int(elems[0])
        page = slice(start, stop)
    else:
        page = slice(None, None)
    return page


def libsvm_row(labels, data):
    for label, row in zip(labels, data):
        row = [str(i)+':'+str(x) for i, x in enumerate(row, 1) if x > 0]
        if len(row) > 0:
            row.insert(0, str(label))
            yield row


def to_libsvm(data, target, save_to=None):
        """
        tranforms a dataset to libsvm format
        """
        le = LabelEncoder()
        target_t = le.fit_transform(data.data[target].to_ndarray())
        groups = [group for group in data.groups if group != target]
        with open(save_to, 'w') as f:
            for row in libsvm_row(target_t, data.data[groups].to_ndarray()):
                f.write(" ".join(row))
                f.write("\n")


def filter_dtypes(group: str, dtypes: np.dtype) -> np.dtype:
    return np.dtype([(group, dtypes.fields[group][0])])


def merge_dtype_list(dtype_list: list) -> np.dtype:
    dtypes = []
    for dtype in dtype_list:
        for name in dtype.names:
            d, _ = dtype.fields[name]
            dtypes.append((name, d))
    return np.dtype(dtypes)
