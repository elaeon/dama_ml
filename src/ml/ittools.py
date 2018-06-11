from itertools import izip, chain
import operator
import logging

from ml.layers import IterLayer
from ml.utils.config import get_settings


settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.addHandler(handler)
log.setLevel(int(settings["loglevel"]))


def avg(iters, method="arithmetic"):
    size = len(iters)
    dtype = iters[0].dtype
    chunks_size = iters[0].chunks_size
    if method == "arithmetic":
        iter_ = (sum(x) / float(size) for x in izip(*iters))
    else:
        iter_ = (reduce(operator.mul, x)**(1. / size) for x in izip(*iters))
    return IterLayer(iter_, dtype=dtype, chunks_size=chunks_size)


def max_counter(iters_b, weights=None):
    def merge(labels, weights):
        if weights is None:
            return ((label, 1) for label in labels)
        else:
            values = {}
            for label, w in izip(labels, weights):
                values.setdefault(label, 0)
                values[label] += w
            return values.items()

    iters = iter(iters_b)
    base_iter = next(iters)
    iter_ = (max(merge(x, weights), key=lambda x: x[1])[0] for x in izip(base_iter, *iters))
    return IterLayer(iter_, dtype=base_iter.dtype, chunks_size=base_iter.chunks_size)


def concat(iters):
    if len(iters) > 1:
        base_iter = iters[0]
        length = sum(it.length for it in iters if it.length is not None)
        it = IterLayer(chain(*iters), chunks_size=base_iter.chunks_size)
        it.set_length(None if length == 0 else length)
        return it
    elif len(iters) == 1:
        return iters[0]
