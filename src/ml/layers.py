from itertools import izip, imap
import operator
import collections
import itertools
import types


def choice(operator):
    def inner(fn):
        def view(x, y):
            if isinstance(x, IterLayer) and isinstance(y, IterLayer):
                return x.stream_operation(operator, y)
            elif isinstance(y, collections.Iterable):
                return x.stream_operation(operator, IterLayer(y))
            else:
                return x.scalar_operation(operator, y)
        return view
    return inner


class IterLayer(object):
    def __init__(self, fn_iter, shape=None):
        self.shape = shape
        if isinstance(fn_iter, types.GeneratorType):
            self.fn_iter = fn_iter
        elif isinstance(fn_iter, IterLayer):
            self.fn_iter = fn_iter.fn_iter
        else:
            self.fn_iter = (e for e in fn_iter)

    def scalar_operation(self, operator, scalar):
        iter_ = imap(lambda x: operator(x, scalar), self)
        return IterLayer(iter_)

    def stream_operation(self, operator, stream):
        iter_ = imap(lambda x: operator(x[0], x[1]), izip(self, stream))
        return IterLayer(iter_)

    @choice(operator.add)
    def __add__(self, x):
        return

    @choice(operator.sub)
    def __sub__(self, x):
        return

    @choice(operator.mul)
    def __mul__(self, x):
        return

    @choice(operator.div)
    def __div__(self, x):
        return

    @choice(pow)
    def __pow__(self, x):
        return

    __rmul__ = __mul__
    __radd__ = __add__

    @choice(operator.iadd)
    def __iadd__(self, x):
        return

    @choice(operator.isub)
    def __isub__(self, x):
        return

    @choice(operator.imul)
    def __imul__(self, x):
        return

    @choice(operator.idiv)
    def __idiv__(self, x):
        return

    @choice(operator.ipow)
    def __ipow__(self, x):
        return

    @classmethod
    def avg(self, iters, size, method="arithmetic"):
        if method == "arithmetic":
            iter_ = (sum(x) / float(size) for x in izip(*iters))
        else:
            iter_ = (reduce(operator.mul, x)**(1. / size) for x in izip(*iters)) 
        return IterLayer(iter_)

    @classmethod
    def max_counter(self, iters, weights=None):
        def merge(labels, weights):
            if weights is None:
                return ((label, 1) for label in labels)
            else:
                values = {}
                for label, w in izip(labels, weights):
                    values.setdefault(label, 0)
                    values[label] += w
                return values.items()

        iter_ = (max(merge(x, weights), key=lambda x: x[1])[0] for x in izip(*iters))
        return IterLayer(iter_)

    @classmethod
    def concat_n(self, iters):
        return IterLayer(itertools.chain(*iters))

    def concat_elems(self, data):
        iter_ = (list(itertools.chain(x0, x1)) for x0, x1 in izip(self, data))
        return IterLayer(iter_)

    def compose(self, fn, *args, **kwargs):
        iter_ = (fn(x, *args, **kwargs) for x in self)
        return IterLayer(iter_)

    def concat(self, iterlayer):
        return IterLayer(itertools.chain(self, iterlayer))

    def to_datamodelset(self, labels, features, size, ltype):
        from ml.ds import DataLabel
        from ml.utils.config import get_settings
        import numpy as np

        settings = get_settings("ml")

        data = np.zeros((size, features))
        label_m = np.empty(size, dtype=ltype)
        for i, y in enumerate(self):
            row_c = i % labels.shape[0]
            data[i] = y
            label_m[i] = labels[row_c]
        
        #fixme: add a dataset chunk writer
        dataset = DataLabel(dataset_path=settings["dataset_model_path"], ltype=ltype)
        with dataset:
            dataset.build_dataset(data, label_m)
        return dataset

    def to_narray(self, dtype=float):
        import numpy as np
        if self.shape is None:
            raise Exception("Data shape is None, IterLayer can't be converted to array")     
        smx_a = np.empty(self.shape, dtype=dtype)
        init = 0
        end = 0
        for i, row in enumerate(self):
            if len(row.shape) == 1:
                smx_a[i] = row
            else:
                end += row.shape[0]
                smx_a[init:end] = row
                init = end
        return smx_a

    def tee(self):
        it0, it1 = itertools.tee(self)
        return IterLayer(it0), IterLayer(it1)

    def __iter__(self):
        return self

    def next(self):
        try:
            return self.fn_iter.next()
        except StopIteration:
            if hasattr(self.fn_iter, 'close'):
                self.fn_iter.close()
            raise StopIteration
