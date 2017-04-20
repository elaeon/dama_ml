
class IterLayer:
    def __init__(self, fn_iter):
        self.fn_iter = fn_iter

    def operation(self, operator, v):
        from itertools import imap
        return imap(lambda x: operator(x, v), self.fn_iter)

    def scalar_operation(self, operator, scalar):
        iter_layer = IterLayer(self.operation(operator, scalar))
        return iter_layer

    def __add__(self, scalar):
        from operator import add
        return self.scalar_operation(add, scalar)

    def __sub__(self, scalar):
        from operator import sub
        return self.scalar_operation(sub, scalar)

    def __mul__(self, scalar):
        from operator import mul
        return self.scalar_operation(mul, scalar)

    def __div__(self, scalar):
        from operator import div
        return self.scalar_operation(div, scalar)

    def __pow__(self, scalar):
        return self.scalar_operation(pow, scalar)

    def __iadd__(self, scalar):
        from operator import iadd
        return self.scalar_operation(iadd, scalar)

    def __isub__(self, scalar):
        from operator import isub
        return self.scalar_operation(isub, scalar)

    def __imul__(self, scalar):
        from operator import imul
        return self.scalar_operation(imul, scalar)

    def __idiv__(self, scalar):
        from operator import idiv
        return self.scalar_operation(idiv, scalar)

    def __ipow__(self, scalar):
        from operator import ipow
        return self.scalar_operation(ipow, scalar)

    def __iter__(self):
        return self.fn_iter
