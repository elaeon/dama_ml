from collections import MutableSet, OrderedDict
from collections import defaultdict
import weakref
import functools
import dask
import networkx as nx
import copy
import matplotlib.pyplot as plt
from ml.data.it import Iterator


class OrderedSet(MutableSet):
    def __init__(self, values=()):
        self._od = OrderedDict().fromkeys(values)

    def __len__(self):
        return len(self._od)

    def __iter__(self):
        return iter(self._od)

    def __contains__(self, value):
        return value in self._od

    def add(self, value):
        self._od[value] = None

    def discard(self, value):
        self._od.pop(value, None)


class OrderedWeakrefSet(weakref.WeakSet):
    def __init__(self, values=()):
        super(OrderedWeakrefSet, self).__init__()
        self.data = OrderedSet()
        for elem in values:
            self.add(elem)


def inc(x):
    return x + 1


def dec(x):
    return x - 1


def identity(x):
    return x


class StreamABC(object):
    def __init__(self, upstream=None):
        self.downstreams = OrderedWeakrefSet()

        if upstream is None:
            self.upstreams = []
        else:
            self.upstreams = [upstream]

        for upstream in self.upstreams:
            upstream.downstreams.add(self)

    @classmethod
    def register_api(cls, modifier=identity):
        def _(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)
            setattr(cls, func.__name__, modifier(wrapped))
            return func
        return _


class Stream(StreamABC):
    def __init__(self, seq, upstream=None):
        super(Stream, self).__init__(upstream=upstream)
        self.it = seq
        self.G = None

    def graph(self):
        self.G = nx.DiGraph()
        l = []
        self.maps(self.downstreams, l, None)
        self.G.add_edges_from(l)

    def _eval(self, it):
        results = []
        if self.G is None:
            self.graph()
        for x in it:
            leafs = []
            self.evaluate_graph_root(x, None, leafs)
            results.extend(leafs)
            self.clean_graph_eval(None)
        return results

    def compute(self):
        results = self._eval(self.it)
        return dask.compute(results)

    def evaluate_graph_root(self, x, root_node, leafs):
        for node in self.G.neighbors(root_node):
            self.evaluate_graph(node(x), leafs)
        return x

    def evaluate_graph(self, base_node, leafs):
        if len(list(self.G.neighbors(base_node))) == 0:
            nodes = list(self.G.predecessors(base_node))
            if len(nodes) > 1:
                base_node(*nodes)
            if base_node.completed:
                leafs.append(base_node.eval_task)

        for node in self.G.neighbors(base_node):
            nodes = list(self.G.predecessors(node))
            if len(nodes) > 1:
                node(*nodes)
            else:
                node(base_node)
            self.evaluate_graph(node, leafs)

    def clean_graph_eval(self, root):
        for node in self.G.neighbors(root):
            node.eval_task = None
            self.clean_graph_eval(node)

    def maps(self, downstreams, l, parent):
        for fn in downstreams:
            if len(fn.downstreams) > 0:
                if type(fn) == zip:
                    for fn_zip in fn.extra:
                        for map_zip_fn in fn.downstreams:
                            l.append((fn_zip, map_zip_fn))
                    self.maps(fn.downstreams, l, fn)
                else:
                    self.maps(fn.downstreams, l, fn)
            if type(fn) != zip and type(parent) != zip:
                l.append((parent, fn))

    def visualize_graph(self):
        if self.G is None:
            self.graph()
        nx.draw(self.G, with_labels=True, font_weight='bold')
        plt.show()

    def visualize_task_graph(self, **kwargs):
        r = self._eval([1])
        r[0].visualize(**kwargs)

    def __str__(self):
        return "MAIN"
        

@StreamABC.register_api()
class map(StreamABC):
    def __init__(self, upstream, func):
        self.task = dask.delayed(func)
        self.eval_task = None
        self.completed = False
        self.fn_name = func.__name__
        StreamABC.__init__(self, upstream)

    def __call__(self, *nodes):
        fn = []
        for node in nodes:
            if type(node) == map:
                if node.eval_task is None:
                    self.completed = False
                    return self
                fn.append(node.eval_task)
            else:
                fn.append(node)
        self.eval_task = self.task(*fn)
        self.completed = True
        return self

    def __str__(self):
        return self.fn_name


@StreamABC.register_api()
class zip(StreamABC):
    def __init__(self, upstream, *func):
        self.extra = func
        StreamABC.__init__(self, upstream)

    def __str__(self):
        return "zip"


def add(x, y):
    return x + y

it = Iterator([4])
s = Stream(it)
a = s.map(inc)
c = a.map(dec)
f = c.map(lambda x: x*4)
b = s.map(identity)
d = s.zip(c, b).map(add).map(str).map(float)
g = s.zip(d, f).map(add)

for r in s.compute():
    print("RESULT", r)
s.visualize_task_graph(filename="stream", format="svg")
#s.visualize_graph()

