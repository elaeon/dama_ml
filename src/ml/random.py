import numpy as np
from ml.data.it import Iterator


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
