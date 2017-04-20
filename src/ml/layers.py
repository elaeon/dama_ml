from itertools import izip, imap
import operator


def choice(operator):
    def inner(fn):
        def view(x, y):
            if type(x) == type(y):
                return x.stream_operation(operator, y)
            else:
                return x.scalar_operation(operator, y)
        return view
    return inner


class IterLayer:
    def __init__(self, fn_iter):
        self.fn_iter = fn_iter

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

    def __iter__(self):
        return self.fn_iter
