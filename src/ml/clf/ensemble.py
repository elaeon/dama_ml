import numpy as np
from ml.models import DataDrive
from ml.clf.measures import ListMeasure, Measure
from ml.ds import Data
from ml.utils.config import get_settings
from ml.layers import IterLayer

from pydoc import locate
import inspect
import logging

settings = get_settings("ml")
log = logging.getLogger(__name__)
logFormatter = logging.Formatter("[%(name)s] - [%(levelname)s] %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(logFormatter)
log.setLevel(int(settings["loglevel"]))


class Grid(DataDrive):
    def __init__(self, clfs=[], model_name=None,
            check_point_path=None, meta_name="", 
            group_name=None, metrics=None):
        super(Grid, self).__init__(
            check_point_path=check_point_path,
            model_name=model_name,
            group_name=group_name)
        
        self.model = None        
        self.meta_name = meta_name
        self.fn_output = None
        self.metrics = metrics

        if len(clfs) > 0:
            self.classifs = self.transform_clfs(clfs, fn=lambda x: x)
        else:
            self.classifs = None

    @classmethod
    def read_meta(self, path):        
        from ml.ds import load_metadata
        return load_metadata(path+".xmeta")

    def classifs_to_string(self, classifs):
        classifs = [{"model_module": model.module_cls_name(), 
            "model_name": model.model_name, "model_version": model.model_version, 
            "check_point_path": model.check_point_path} 
            for model in classifs]
        return classifs

    def transform_clfs(self, classifs, fn=None):
        layer = []
        for classif_obj in classifs:
            #Case: load model from metadata, 
            if isinstance(classif_obj, dict):
                classif_obj["model_module"] = fn(classif_obj["model_module"])
                layer.append(self.load_model(**classif_obj))
            #Case: [model0, model1, ...]
            else:
                layer.append(classif_obj)
        return layer

    def load_model(self, model_module=None, model_name=None, model_version=None,
        check_point_path=None):
        if inspect.isclass(model_module):
            model = model_module( 
                model_name=model_name, 
                check_point_path=check_point_path,
                group_name=None)
            model.load(model_version=model_version)
        return model
    
    def all_clf_scores(self, measures=None):
        from operator import add
        try:
            return reduce(add, (classif.scores(measures=measures) 
                for classif in self.classifs if hasattr(classif, 'scores')))
        except TypeError:
            return ListMeasure()

    def scores(self, measures=None, all_clf=False):
        from tqdm import tqdm
        if measures is None or isinstance(measures, str):
            measures = Measure.make_metrics(measures, name=self.model_name)

        with self.classifs[0].test_ds as test_ds:
            test_labels = test_ds.labels[:]
        predictions = np.asarray(list(tqdm(self.predict_test(raw=measures.has_uncertain(), chunks_size=0),
            total=test_labels.shape[0])))
        measures.set_data(predictions, test_labels, self.numerical_labels2classes)
        list_measure = measures.to_list()
        if all_clf is True:
            return list_measure + self.all_clf_scores(measures=measures)
        else:
            return list_measure

    def numerical_labels2classes(self, labels):
        if not hasattr(self, 'le'):
            for classif in self.classifs:
                self.le = classif.le
                break

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            return self.le.inverse_transform(np.argmax(labels, axis=1))
        else:
            return self.le.inverse_transform(labels.astype('int'))

    def print_confusion_matrix(self, namespace=None):
        from operator import add
        list_measure = reduce(add, (classif.confusion_matrix() 
            for classif in self.load_models()))
        classifs_reader = self.load_models()
        classif = classifs_reader.next()
        list_measure.print_matrix(classif.base_labels)

    def ordered_best_predictors(self, measure="logloss"):
        list_measure = self.all_clf_scores(measures=measure)
        column_measures = list_measure.get_measure(measure)
        class DTuple:
            def __init__(self, counter, elem):
                self.elem = elem
                self.counter = counter
            
            def __sub__(self, other):
                if self.elem is None or other is None:
                    return 0
                return self.elem - other.elem

            def __str__(self):
                return "({}, {})".format(self.counter, self.elem)

            def __repr__(self):
                return self.__str__()

        def enum(seq, position=0):
            counter = 0
            for elem in seq:
                yield DTuple(counter, elem)
                counter += 1

        return sorted(enum(column_measures["values"]), reverse=column_measures["reverse"])

    def best_predictor_threshold(self, threshold=2, limit=3, measure="logloss"):
        best = self.ordered_best_predictors(measure=measure)
        base = best[0].elem
        return filter(lambda x: x[1] < threshold, 
            ((elem.counter, elem.elem/base) for elem in best if elem.elem is not None))[:limit]

    def best_predictor(self, measure="logloss"):
        best = self.ordered_best_predictors(measure=measure)[0].counter
        return best

    def predict_test(self, raw=False, chunks_size=258):
        def iter_():
            for classif in self.classifs:
                with classif.test_ds as test_ds:
                    test_data = test_ds.data[:]
                yield classif.predict(test_data, raw=raw, transform=False, 
                                        chunks_size=chunks_size)

        return self.output_layer(iter_)

    def predict(self, data, raw=False, transform=True, chunks_size=258):
        def iter_():
            for classif in self.classifs:
                yield classif.predict(data, raw=raw, transform=transform, 
                                    chunks_size=chunks_size)
        return self.output_layer(iter_)

    def output_layer(self, iter_):
        if self.fn_output == "stack":
            return IterLayer.concat_n(iter_())
        elif self.fn_output is None or self.fn_output == "avg":
            return IterLayer.avg(iter_(), len(self.classifs))
        elif self.fn_output == "bagging":
            return IterLayer.avg(iter_(), len(self.classifs), method="geometric")
        else:
            return self.fn_output(*iter_())

    def destroy(self):
        from ml.utils.files import rm
        rm(self.path_m+".xmeta")

    def _metadata(self, calc_scores=True):
        models = {"models": None}
        models["models"] = self.classifs_to_string(self.classifs)
        models["output"] = self.fn_output
        models["score"] = self.scores(self.metrics).measures_to_dict()
        return models

    def load_meta(self):
        from ml.ds import load_metadata
        if self.check_point_path is not None:
            metadata = {}
            self.path_m = self.make_model_file()
            return load_metadata(self.path_m+".xmeta")

    def save_meta(self):
        from ml.ds import save_metadata
        metadata = self._metadata()
        self.path_m = self.make_model_file()
        save_metadata(self.path_m+".xmeta", metadata)

    def load(self):
        from ml.ds import load_metadata
        metadata = self.load_meta()
        self.fn_output = metadata["output"]
        self.classifs = self.transform_clfs(metadata.get('models', {}), fn=locate)
        self.output(metadata["output"])

    def save(self):
        if self.check_point_path:
            self.save_meta()

    def output(self, fn):
        self.fn_output = fn

    def scores2table(self):
        from ml.clf.measures import ListMeasure
        return ListMeasure.dict_to_measures(self.load_meta().get("score", None))


class Boosting(Grid):
    def __init__(self, classifs, weights=None, election='best', num_max_clfs=1, 
            **kwargs):
        super(Boosting, self).__init__(classifs, meta_name="boosting", **kwargs)
        if len(classifs) > 0:
            self.weights = self.set_weights(0, self.classifs["0"], weights)
            self.election = election
            self.num_max_clfs = num_max_clfs
        else:
            meta = self.load_meta()
            self.weights = self.get_boosting_values(meta['weights'])
            self.election = meta["election"]
            self.model_name = meta["model_name"]

    def get_boosting_values(self, weights):
        return {index: w for index, w in enumerate(weights)}

    def avg_prediction(self, predictions, weights, uncertain=True):
        from ml.layers import IterLayer
        if uncertain is False:
            return IterLayer.max_counter(predictions, weights=weights)
        else:
            predictors = IterLayer(predictions) * weights
            return IterLayer.avg(predictors, sum(weights))

    def train(self, batch_size=128, num_steps=1, only_voting=False, calc_score=True):
        self.calc_score = calc_score
        if only_voting is False:
            for classif in self.load_models():
                log.info("Training [{}]".format(classif.__class__.__name__))
                classif.train(batch_size=batch_size, num_steps=num_steps)

        if self.election == "best":
            from ml.utils.numeric_functions import le
            best = self.best_predictor(operator=le)
            self.weights = self.set_weights(best, self.classifs[self.meta_name+".0"], self.weights.values())
            models = self.classifs
            models_index = range(len(self.classifs[self.meta_name+".0"]))
        elif self.election == "best-c":
            bests = self.correlation_between_models(sort=True)
            self.weights = self.set_weights(bests[0], self.classifs[self.meta_name+".0"], self.weights.values())
            key = self.meta_name + ".0"
            models = {key: []}
            for index in bests:
                models[key].append(self.classifs[self.meta_name+".0"][index])
            models_index = bests

        weights = [self.weights[index] for index in models_index]
        self.save_model(models, weights)
        self.reload()

    def _metadata(self):
        meta = super(Boosting, self)._metadata()
        meta["weights"] = self.clf_weights
        meta["election"] = self.election
        return meta

    def save_model(self, models, weights):
        self.clf_models_namespace = self.string_namespaces(models)
        self.clf_weights = weights
        self.weights = self.get_boosting_values(self.clf_weights)
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.save_meta()

    def set_weights(self, best, classifs, values):
        if values is None:
            values = [1]
        max_value = max(values)
        min_value = min(values)
        weights = {} 
        for c_index, clf in enumerate(classifs):
            if c_index == best:
                weights[c_index] = max_value
            else:
                weights[c_index] = min_value
        return weights

    def predict(self, data, raw=False, transform=True, chunks_size=1):
        models = (self.load_model(classif, namespace=self.meta_name+".0") 
                    for classif, _ in self.classifs[self.meta_name+".0"])
        weights = [w for i, w in sorted(self.weights.items(), key=lambda x:x[0])]
        predictions = (
            classif.predict(data, raw=raw, transform=transform, chunks_size=chunks_size)
            for classif in models)
        return self.avg_prediction(predictions, weights, uncertain=raw)

    def correlation_between_models(self, sort=False):
        from ml.utils.numeric_functions import pearsoncc
        from ml.utils.network import all_simple_paths_graph
        from itertools import combinations
        from ml.utils.order import order_from_ordered
        import networkx as nx

        best_predictors = []
        def predictions_fn():
            predictions = {}
            for index, _ in self.best_predictor_threshold(limit=self.num_max_clfs):
                clf_class, _ = self.classifs[self.meta_name+".0"][index]
                classif = self.load_model(clf_class, namespace=self.meta_name+".0")
                predictions[index] = np.asarray(list(classif.predict(
                    classif.dataset.test_data, raw=None, transform=False, chunks_size=258)))
                best_predictors.append(index)
            return predictions

        def correlations_fn(predictions):
            for clf_index1, clf_index2 in combinations(predictions.keys(), 2):
                correlation = pearsoncc(predictions[clf_index1], predictions[clf_index2])
                yield (clf_index1, clf_index2, correlation)

        FG = nx.Graph()
        FG.add_weighted_edges_from(correlations_fn(predictions_fn()))
        classif_weight = []
        for initial_node in FG.nodes():
            for path in all_simple_paths_graph(FG, initial_node, self.num_max_clfs-2):
                total_weight = sum(FG[v1][v2]["weight"] for v1, v2 in combinations(path, 2))
                #total_weight = sum(FG[v1][v2]["weight"] for v1, v2 in zip(path, path[1:]))
                classif_weight.append((total_weight/len(path), path))

        if len(classif_weight) == 0:
            return (0, [1])[1]
        else:
            relation_clf = min(classif_weight, key=lambda x:x[0])[1]
            if sort:
                return order_from_ordered(best_predictors, relation_clf)
            else:
                return relation_clf
