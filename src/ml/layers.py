from itertools import izip, imap
import operator
import collections
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


class IterLayer:
    def __init__(self, fn_iter):
        if isinstance(fn_iter, types.GeneratorType):
            self.fn_iter = fn_iter
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

    def compose(self, fn, *args, **kwargs):
        iter_ = (fn(x, *args, **kwargs) for x in self)
        return IterLayer(iter_)

    def __iter__(self):
        return self.fn_iter
