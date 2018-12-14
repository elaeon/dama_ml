from collections import MutableSet, OrderedDict
# import weakref
import functools
import dask
import networkx as nx
import matplotlib.pyplot as plt
from pydoc import locate
import sys
import json
from dask import get as sync_get


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


#class OrderedWeakrefSet(weakref.WeakSet):
#    def __init__(self, values=()):
#        super(OrderedWeakrefSet, self).__init__()
#        self.data = OrderedSet()
#        for elem in values:
#            self.add(elem)


class OrderedWeakrefSet(OrderedSet):
    def __init__(self, values=()):
        super(OrderedWeakrefSet, self).__init__()


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
    def register_api(cls):
        def _(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)
            setattr(cls, func.__name__, wrapped)
            return func
        return _

    @property
    def key(self):
        return NotImplemented

    def node_ref(self):
        return NotImplemented

    @staticmethod
    def fn_name(fn):
        return "{}.{}".format(fn.__module__, fn.__name__)


class Pipeline(PipelineABC):
    def __init__(self, data=None, upstream=None):
        super(Pipeline, self).__init__(upstream=upstream)
        self.data = data
        self.di_graph = None
        self.root = 'root'

    def graph(self) -> nx.DiGraph:
        di_graph = nx.DiGraph()
        edges = self.edges(self.downstreams, self.root)
        di_graph.add_edges_from(edges)
        return di_graph

    def _eval(self, batch):
        if self.di_graph is None:
            self.di_graph = self.graph()
        leafs = []
        self.evaluate_graph_root(batch, self.root, leafs)
        self.clean_graph_eval(self.root)
        return leafs

    def evaluate_graph_root(self, x, root_node, leafs):
        for node in self.di_graph.neighbors(root_node):
            self.evaluate_graph(node(x), leafs)

    def evaluate_graph(self, base_node, leafs):
        if len(list(self.di_graph.neighbors(base_node))) == 0:
            nodes = list(self.di_graph.predecessors(base_node))
            if len(nodes) > 1:
                base_node(*nodes)
            if base_node.completed:
                leafs.append((base_node.eval_task, base_node.key))

        for node in self.di_graph.neighbors(base_node):
            nodes = list(self.di_graph.predecessors(node))
            if len(nodes) > 1:
                node(*nodes)
            else:
                node(base_node)
            self.evaluate_graph(node, leafs)

    def clean_graph_eval(self, root) -> None:
        for node in self.di_graph.neighbors(root):
            if isinstance(node, PipelineABC):
                node.eval_task = None
            self.clean_graph_eval(node)

    def edges(self, downstreams: OrderedWeakrefSet, parent) -> list:
        edges = []
        for fn in downstreams:
            if type(fn) == zip:
                for fn_zip in fn.args:
                    for map_zip_fn in fn.downstreams:
                        edges.append((fn_zip, map_zip_fn))
            edges.extend(self.edges(fn.downstreams, fn))
            
            if type(fn) != zip and type(parent) != zip:
                edges.append((parent, fn))
        return edges

    def visualize_graph(self) -> None:
        if self.di_graph is None:
            self.di_graph = self.graph()
        nx.draw(self.di_graph, with_labels=True, font_weight='bold')
        plt.show()

    def visualize(self, **kwargs) -> None:
        graphs = self._eval(None)
        for i, (graph, _) in enumerate(graphs):
            if 'filename' not in kwargs:
                graph.visualize(filename='{}_{}'.format(i, graph.key), **kwargs)
            else:
                graph.visualize(**kwargs)

    def compute(self, scheduler=None, **kwargs):
        if scheduler is None:
            scheduler = sync_get
        dask_graph = self.to_dask_graph()
        leafs = [key for _, key in self._eval(None)]
        return scheduler(dask_graph, leafs, **kwargs)

    def to_dask_graph(self) -> dict:
        if self.di_graph is None:
            self.di_graph = self.graph()
        dask_graph = {}
        for node in self.di_graph.neighbors(self.root):
            dask_graph[node.placeholder] = self.data
            if node.data is None:
                dask_graph[node.key] = (node.func, node.placeholder)
            else:
                dask_graph[node.key] = (node.func, node.data)
            self._walk(node, dask_graph)
        return dask_graph

    def _walk(self, base_node: PipelineABC, dask_graph: dict) -> None:
        if len(list(self.di_graph.neighbors(base_node))) == 0:
            return

        for node in self.di_graph.neighbors(base_node):
            if node.key in dask_graph:
                params = [param for param in dask_graph[node.key]]
                if base_node.key not in params[1:]:
                    tuple_value = tuple(params + [base_node.key])
                    dask_graph[node.key] = tuple_value
            else:
                if node.data is None:
                    dask_graph[node.key] = (node.func, base_node.key)
                else:
                    dask_graph[node.key] = (node.func, node.data)
            self._walk(node, dask_graph)

    def feed(self, data) -> 'Pipeline':
        self.data = data
        return self

    def to_json(self) -> str:
        placeholder = {"fn": ["placeholder"],  "input": [type(self.data).__name__, self.data]}
        text = [placeholder]
        text.extend(self._to_json(self.downstreams))
        return json.dumps(text)

    def _to_json(self, downstreams: OrderedWeakrefSet) -> list:
        row = []
        for fn in downstreams:
            row.append(fn.node_ref())
            row.extend(self._to_json(fn.downstreams))
        return row

    @classmethod
    def load(cls, json_text: str, path: str) -> 'Pipeline':
        sys.path.append(path)
        json_stc = json.loads(json_text)
        tree_map = {}
        for lvl in json_stc:
            if lvl["fn"][0] == "placeholder":
                if lvl["input"][0] == "int":
                    cast = int
                else:
                    cast = lambda x: x
                tree_map["placeholder"] = Pipeline(cast(lvl["input"][1]))
            elif lvl["fn"][0] == "map":
                fn = lvl["fn"][1]
                tree_map[lvl["id"]] = tree_map[lvl["input"]].map(locate(fn))
            elif lvl["fn"][0] == "zip":
                args = [tree_map[name] for name in lvl["args"]]
                tree_map[lvl["id"]] = tree_map[lvl["input"]].zip(*args)
        return tree_map["placeholder"]

    @property
    def key(self) -> str:
        return self.root

    def __str__(self) -> str:
        return "Pipeline"

    def __repr__(self) -> str:
        return self.key
        

@PipelineABC.register_api()
class map(PipelineABC):
    def __init__(self, upstream, func, data=None, **kwargs):
        self.task = dask.delayed(func)
        self.func = func
        self.eval_task = None
        self.completed = False
        self.data = data
        self.func_params = kwargs
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

    def get_root(self, node) -> Pipeline:
        if len(node.upstreams) > 0:
            parent = node.upstreams[-1]
            ant = self.get_root(parent)
            if ant.key == "root":
                return ant
        return node

    @property
    def key(self) -> str:
        return self.task.key

    @property
    def placeholder(self) -> str:
        return 'placeholder'.format(self.func.__name__)

    def compute(self, scheduler=None, **kwargs):
        if scheduler is None:
            scheduler = sync_get
        root_node = self.get_root(self)
        dask_graph = root_node.to_dask_graph()
        return scheduler(dask_graph, self.key, **kwargs)

    def node_ref(self):
        upstream = self.upstreams[-1]
        if upstream.key == "root":
            ref = self.placeholder
        else:
            ref = upstream.key
        return {"fn": [self.__class__.__name__, Pipeline.fn_name(self.func)],
                "input": ref, "id": self.key, "kwargs": self.func_params}

    def __str__(self):
        return "{} fn: {}".format(self.__class__.__name__, Pipeline.fn_name(self.func))

    def __repr__(self):
        return self.__str__()


@PipelineABC.register_api()
class zip(PipelineABC):
    def __init__(self, upstream, *func):
        self.args = func
        PipelineABC.__init__(self, upstream)

    @property
    def placeholder(self) -> str:
        return 'placeholder'

    def node_ref(self):
        upstream = self.upstreams[-1]
        if upstream.key == "root":
            ref = self.placeholder
        else:
            ref = upstream.key
        args = [arg.key for arg in self.args]
        return {"fn": [self.__class__.__name__], "input": ref, "id": self.key, "args": args}

    @property
    def key(self) -> str:
        return "zip"

    def __str__(self):
        return self.__class__.__name__

