import numpy as np
from ml.clf.wrappers import DataDrive
from ml.clf.measures import ListMeasure, Measure
from ml.ds import DataSetBuilder, Data
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
    def __init__(self, clfs=[], model_name=None, dataset=None, 
            check_point_path=None, model_version=None, meta_name="", 
            group_name=None, metrics=None):
        super(Grid, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)
        
        self.model = None        
        self.meta_name = meta_name
        self.fn_output = None
        self.metrics = metrics

        if len(clfs) == 0:
            self.reload()
        else:
            self.classifs = self.transform_clfs(clfs, fn=lambda x: x)
            #self.dataset = dataset

    def reset_dataset(self, dataset):
        clf_layers = {}
        classifs = self.classifs
        self.classifs = self.transform_clfs(
            [(classif[0], dataset) for classif in classifs], 
            fn=lambda x: x)
        #self.dataset = dataset

    def reload(self):
        meta = self.load_meta()
        #self.dataset = Data.original_ds(meta["dataset_name"], meta["dataset_path"])
        self.classifs = self.transform_clfs(meta.get('models', {}), fn=locate)
        self.output(meta["output"])

    def classifs_to_string(self, classifs):
        classifs = [{"classif": model.module_cls_name(), 
            "model_name": model.model_name, "model_version": model.model_version, 
            "check_point_path": model.check_point_path} 
            for model in classifs]
        return classifs

    def transform_clfs(self, classifs, fn=None):
        layer = []
        for classif_obj in classifs:
            #Case: load model from metadata, 
            if isinstance(classif_obj, dict):
                classif_obj["classif"] = fn(classif_obj["classif"])
                layer.append(self.load_model(**classif_obj))
            #Case: [model0, model1, ...]
            else:
                layer.append(classif_obj)
        return layer

    #def load_models(self, autoload=True):
    #    for clf_params in self.classifs:
    #        yield self.load_model(**clf_params)

    def load_model(self, classif=None, model_name=None, model_version=None,
        check_point_path=None, dataset=None):
        if inspect.isclass(classif):
            model = classif(dataset=dataset, 
                    model_name=model_name, 
                    model_version=model_version, 
                    check_point_path=check_point_path,
                    group_name=self.group_name,
                    autoload=True)

        return model

    def train(self):
        self.save_model()
    
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

        with self.classifs[0].dataset as ds_test:
            test_labels = ds_test.test_labels[:]
        predictions = np.asarray(list(tqdm(self.predict_test(raw=measures.has_uncertain(), chunk_size=0),
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

    def predict_test(self, raw=False, chunk_size=258):
        def iter_():
            for classif in self.classifs:
                with classif.dataset as ds:
                    yield classif.predict(ds.test_data[:], raw=raw, transform=False, 
                                        chunk_size=chunk_size)

        return self.output_layer(iter_)

    def predict(self, data, raw=False, transform=True, chunk_size=258):
        def iter_():
            for classif in self.classifs:
                yield classif.predict(data, raw=raw, transform=transform, 
                                        chunk_size=chunk_size)
        return self.output_layer(iter_, data=data)

    def output_layer(self, iter_, data=None):
        if self.fn_output == "stack":
            return IterLayer.concat_n(iter_())
        elif self.fn_output is None or self.fn_output == "avg":
            return IterLayer.avg(iter_(), len(self.classifs))
        elif self.fn_output == "bagging":
            predictions = IterLayer.avg(iter_(), len(self.classifs), method="geometric")
            return predictions.concat_elems(data)
        else:
            return self.fn_output(*iter_())

    def destroy(self):
        from ml.utils.files import rm
        for clf in self.classifs:
            clf.destroy()
        rm(self.get_model_path()+".xmeta")

    def _metadata(self, calc_scores=True):
        if calc_scores:
            list_measure = self.scores(self.metrics)
        else:
            list_measure = ListMeasure()

        #if self.dataset is not None:
        #    dataset_path = self.dataset.dataset_path
        #    dataset_name = self.dataset.name
        #else:
        #    dataset_path = None
        #    dataset_name = None
        return {
                #"dataset_path": dataset_path,
                #"dataset_name": dataset_name,
                "models": self.classifs_to_string(self.classifs),
                "model_name": self.model_name,
                "output": self.fn_output,
                "score": list_measure.measures_to_dict()}

    def save_model(self):
        self.save_meta()

    def output(self, fn):
        self.fn_output = fn

    def scores2table(self):
        from ml.clf.measures import ListMeasure
        return ListMeasure.dict_to_measures(self.load_meta().get("score", None))


class EnsembleLayers(DataDrive):
    def __init__(self, raw_dataset=None, metrics=None, **kwargs):
        super(EnsembleLayers, self).__init__(**kwargs)

        self.metrics = metrics
        self.layers = []
        if raw_dataset is None:
            self.reload()
        else:
            self.dataset = raw_dataset

    def add(self, ensemble):
        self.layers.append(ensemble)

    def train(self, n_splits=None):
        from ml.clf.utils import add_params_to_params

        initial_layer = self.layers[0]
        #initial_layer.train(others_models_args=others_models_args[0], 
        #                    calc_scores=calc_scores)
        #with self.dataset:
        #y_submission = initial_layer.predict(
        #    self.dataset.data_validation, raw=True, 
        #    transform=not self.dataset._applied_transforms, chunk_size=258)
        y_submission = initial_layer.predict_test(raw=True, chunk_size=258)
        for classif in initial_layer.classifs:
            num_labels = classif.num_labels
            with classif.dataset as ds:
                nrows = ds.shape[0]
                test_labels = ds.test_labels[:]
            break

        if initial_layer.fn_output == "stack":
            deep = len(initial_layer.classifs)
        else:
            deep = 1

        size = nrows * deep
        dataset = y_submission.to_datamodelset(test_labels, num_labels, size, 'int')

        if len(self.layers) > 1:
            second_layer = self.layers[1]
            second_layer.reset_dataset(dataset)
            second_layer.train()

        #self.save_model()
        #self.reload()

    def numerical_labels2classes(self, labels):
        if not hasattr(self, 'le'):
            for classif in self.layers[-1].load_models():
                self.le = classif.le
                break

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            return self.le.inverse_transform(np.argmax(labels, axis=1))
        else:
            return self.le.inverse_transform(labels.astype('int'))

    def save_model(self):
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.save_meta()

    def scores(self, measures=None):
        from tqdm import tqdm
        if measures is None or isinstance(measures, str):
            measures = Measure.make_metrics(measures, name=self.model_name)
        with self.dataset:
            predictions = np.asarray(list(tqdm(
                self.predict(self.dataset.test_data[:], raw=measures.has_uncertain(), 
                            transform=not self.dataset._applied_transforms, chunk_size=258), 
                total=self.dataset.test_labels.shape[0])))
            measures.set_data(predictions, self.dataset.test_labels[:], self.numerical_labels2classes)
        return measures.to_list()

    def _metadata(self):
        list_measure = self.scores(self.metrics)
        clf_models_namespace = {}
        for i, layer in enumerate(self.layers, 1):
            clf_models_namespace["layer"+str(i)] = {
                "model": layer.module_cls_name(),
                "model_name": layer.model_name,
                "model_version": layer.model_version,
                "check_point_path": layer.check_point_path
            }
        return {
            "dataset_path": self.dataset.dataset_path,
            "dataset_name": self.dataset.name,
            "models": clf_models_namespace,
            "score": list_measure.measures_to_dict(),
            "num_layers": len(self.layers),
            "group_name": self.group_name}

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        stack = False
        for (raw_l, transform_l), layer in zip([(True, transform), (raw, False)], self.layers):
            if layer.fn_output is None:
                layer.output("stack")

            data = np.asarray(list(layer.predict(data, raw=raw_l, transform=transform_l, 
                chunk_size=chunk_size)))

        if self.layers[0].fn_output == "stack":
            size = len(self.layers[0].classifs["0"])
            batch = data.shape[0] / size
            def sub_matrix(size, batch):                
                i = 0
                step = batch
                while i < data.shape[0]:
                    yield data[i:step]
                    i = step
                    step += batch

            data = sum(sub_matrix(size, batch)) / float(size)
        return data
        
    def destroy(self):
        for grid in self.layers:
            grid.destroy()

    def clean(self):
        self.layers = []
        self.dataset = None

    def reload(self):
        self.clean()
        meta = self.load_meta()
        models = meta.get('models', {})
        for i in range(1, meta.get("num_layers", 0)+1): 
            key = "layer{}".format(i)     
            model_1 = locate(models[key]["model"])
            log.debug("Load {} {} {} {}".format(
                models[key]["model"], models[key]["model_name"], 
                models[key]["model_version"], models[key]["check_point_path"]))
            classif = model_1([], 
                model_name=models[key]["model_name"], 
                model_version=models[key]["model_version"],
                check_point_path=models[key]["check_point_path"])
            self.add(classif)
        self.dataset = Data.original_ds(meta["dataset_name"], meta["dataset_path"])
        return models

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

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        models = (self.load_model(classif, namespace=self.meta_name+".0") 
                    for classif, _ in self.classifs[self.meta_name+".0"])
        weights = [w for i, w in sorted(self.weights.items(), key=lambda x:x[0])]
        predictions = (
            classif.predict(data, raw=raw, transform=transform, chunk_size=chunk_size)
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
                    classif.dataset.test_data, raw=None, transform=False, chunk_size=258)))
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
