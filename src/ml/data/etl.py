from collections import MutableSet, OrderedDict
import weakref
import functools
import dask
import networkx as nx
import matplotlib.pyplot as plt


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


def identity(x):
    return x


class PipelineABC(object):
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


class Pipeline(PipelineABC):
    def __init__(self, data, upstream=None):
        super(Pipeline, self).__init__(upstream=upstream)
        self.data = data
        self.G = None
        self.root = 'root'

    def graph(self):
        self.G = nx.DiGraph()
        nodes = []
        self.maps(self.downstreams, nodes, self.root)
        self.G.add_edges_from(nodes)

    def _eval(self, batch):
        if self.G is None:
            self.graph()
        leafs = []
        self.evaluate_graph_root(batch, self.root, leafs)
        self.clean_graph_eval(self.root)
        return leafs

    def compute_delays(self):
        for values in self.data:
            leafs = self._eval(values)
            yield dask.compute(leafs)[0]

    def evaluate_graph_root(self, x, root_node, leafs):
        for node in self.G.neighbors(root_node):
            self.evaluate_graph(node(x), leafs)

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
            if isinstance(node, PipelineABC):
                node.eval_task = None
            self.clean_graph_eval(node)

    def maps(self, downstreams, l, parent):
        for fn in downstreams:
            if type(fn) == zip:
                for fn_zip in fn.args:
                    for map_zip_fn in fn.downstreams:
                        l.append((fn_zip, map_zip_fn))
            self.maps(fn.downstreams, l, fn)
            
            if type(fn) != zip and type(parent) != zip:
                l.append((parent, fn))

    def visualize_graph(self):
        if self.G is None:
            self.graph()
        nx.draw(self.G, with_labels=True, font_weight='bold')
        plt.show()

    def visualize(self, **kwargs):
        graphs = self._eval(None)
        for i, graph in enumerate(graphs):
            if 'filename' not in kwargs:
                graph.visualize(filename='{}_{}'.format(i, graph.key), **kwargs)
            else:
                graph.visualize(**kwargs)

    def to_dask_graph(self):
        if self.G is None:
            self.graph()
        dask_graph = {}
        for node in self.G.neighbors(self.root):
            dask_graph[node.input_value] = self.data
            if node.with_values is None:
                dask_graph[node.key] = (node.func, node.input_value)
            else:
                dask_graph[node.key] = (node.func, node.with_values)
            self._walk_graph(node, dask_graph)
        return dask_graph

    def _walk_graph(self, base_node: PipelineABC, dask_graph: dict):
        if len(list(self.G.neighbors(base_node))) == 0:
            return

        for node in self.G.neighbors(base_node):
            if node.key in dask_graph:
                params = [param for param in dask_graph[node.key]]
                tuple_value = tuple(params + [base_node.key])
                dask_graph[node.key] = tuple_value
            else:
                if node.with_values is None:
                    dask_graph[node.key] = (node.func, base_node.key)
                else:
                    dask_graph[node.key] = (node.func, node.with_values)
            self.walk_graph(node, dask_graph)

    def __str__(self):
        return "MAIN"
        

@PipelineABC.register_api()
class map(PipelineABC):
    def __init__(self, upstream, func, with_values=None):
        self.task = dask.delayed(func)
        self.func = func
        self.eval_task = None
        self.completed = False
        self.fn_name = func.__name__
        self.with_values = with_values
        self.input_value = '{}_x'.format(self.fn_name)
        PipelineABC.__init__(self, upstream)

    def __call__(self, *nodes):
        items = []
        for node in nodes:
            if type(node) == map:
                if node.eval_task is None:
                    self.completed = False
                    return self
                items.append(node.eval_task)
            elif node is None:
                pass
            else:
                items.append(node)
        self.eval_task = self.task(*items)
        self.completed = True
        return self

    @property
    def key(self):
        return "{}-{}".format(self.fn_name, "fn")

    def __str__(self):
        return self.fn_name


@PipelineABC.register_api()
class zip(PipelineABC):
    def __init__(self, upstream, *func):
        self.args = func
        PipelineABC.__init__(self, upstream)

    def __str__(self):
        return "zip"

