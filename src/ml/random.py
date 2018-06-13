import numpy as np
from ml.data.it import Iterator
from ml.utils.numeric_functions import filter_sample
from ml import ittools


def sampling_size(sampling, stream):
    if isinstance(sampling, Iterator):
        counter = stream.unique()
    else:
        u_values, counter = np.unique(stream, return_counts=True)
        counter = dict(zip(u_values, counter))

    if len(counter) == 0:
        return {}

    sampling_n = {}
    for y, k in sampling.items():
        unique_v = counter.get(y, 0)
        if 0 <= k <= 1:
            v = unique_v * k
        elif 1 < k < unique_v:
             v = k % unique_v
        else:
            v = unique_v % k
        sampling_n[y] = int(round(v, 0))

    return sampling_n


def downsample(stream, sampling, col_index, size, exact=False):
    iterators = []
    for y, k in sampling.items():
        iterators.append(Iterator(filter_sample(stream[:size], y, col_index)).sample(k))
    return ittools.concat(iterators)
