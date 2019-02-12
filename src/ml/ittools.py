from functools import reduce
import operator
import logging

from ml.data.it import Iterator
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
    dtypes = iters[0].dtypes
    batch_size = getattr(iters[0], 'batch_size', 0)
    if method == "arithmetic":
        iter_ = (sum(x) / float(size) for x in zip(*iters))
    else:
        iter_ = (reduce(operator.mul, x)**(1. / size) for x in zip(*iters))
    return Iterator(iter_, dtypes=dtypes).batchs(batch_size=batch_size)


def max_counter(iters_b, weights=None):
    def merge(labels, weights):
        if weights is None:
            return ((label, 1) for label in labels)
        else:
            values = {}
            for label, w in zip(labels, weights):
                values.setdefault(label, 0)
                values[label] += w
            return values.items()

    iters = iter(iters_b)
    base_iter = next(iters)
    batch_size = getattr(base_iter, 'batch_size', 0)
    iter_ = (max(merge(x, weights), key=lambda x: x[1])[0] for x in zip(base_iter, *iters))
    return Iterator(iter_, dtypes=base_iter.dtypes).batchs(batch_size=batch_size)

