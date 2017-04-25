import numpy as np
from ml.clf.wrappers import DataDrive
from ml.clf.measures import ListMeasure
from ml.ds import DataSetBuilder, Data

from pydoc import locate


class Grid(DataDrive):
    def __init__(self, classifs, model_name=None, dataset=None, 
            check_point_path=None, model_version=None, meta_name="", 
            group_name=None):
        super(Grid, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name,
            group_name=group_name)
        
        self.model = None        
        self.meta_name = meta_name

        if len(classifs) == 0:
            self.reload()
        else:
            classifs = {"0": classifs}
            classifs_layers = self.rename_namespaces(classifs)
            self.individual_ds = self.check_layers(classifs_layers)
            if self.individual_ds != True:
                classifs_layers = self.load_namespaces(classifs_layers, dataset=dataset,
                    fn=lambda x: x)
            self.classifs = classifs_layers
        self.dataset = dataset
        self.params = {}

    def reset_dataset(self, dataset, individual_ds=False):
        self.individual_ds = individual_ds
        clf_layers = {}
        for layer, classifs in self.classifs.items():
            clf_layers[layer] = [(classif, dataset) for classif, _ in classifs]
        self.classifs = clf_layers
        self.dataset = dataset

    def active_network(self):
        if self.meta_name is None or self.meta_name == "":
            return "0"
        else:
            return ".".join([self.meta_name, "0"])

    def reload(self):
        meta = self.load_meta()
        self.individual_ds = meta.get('individual_ds', None)
        classifs_layers = meta.get('models', {})
        if self.individual_ds == True:
            for layer, classifs in classifs_layers.items():
                classifs_layers[layer] = [(classif, None) for classif in classifs]
        classifs_layers = self.load_namespaces(classifs_layers, fn=locate)
        self.classifs = classifs_layers

    def check_layers(self, classifs):
        try:
            for layer, classifs in classifs.items():
                for classif, dataset in classifs:
                    continue
            return True
        except TypeError:
            return False
        except ValueError:
            return False

    def string_namespaces(self, classifs):
        namespaces = {}
        for namespace, models in classifs.items():
            namespaces[namespace] = [model.module_cls_name() for model, dataset in models]
        return namespaces

    def load_namespaces(self, classifs_layers, dataset=None, fn=None):
        clf_layers = {}
        if self.individual_ds == False:
            for layer, classifs in classifs_layers.items():
                clf_layers[layer] = [(fn(classif), dataset) for classif in classifs]
        else:
            for layer, classifs in classifs_layers.items():
                clf_layers[layer] = [(fn(classif), dataset) for classif, dataset in classifs]
        return clf_layers

    def rename_namespaces(self, classifs):
        namespaces = {}
        for namespace, models in classifs.items():
            n_namespace = namespace if self.meta_name == "" else self.meta_name + "." + namespace                
            namespaces[n_namespace] = models                
        return namespaces

    def model_namespace2str(self, namespace):
        if namespace is None:
            return self.model_name
        else:
            return "{}.{}".format(self.model_name, namespace)

    def load_models(self, autoload=True):
        namespace = self.active_network()
        for index, (classif, dataset) in enumerate(self.classifs[namespace]):
            yield self.load_model(classif, dataset=dataset, 
                namespace=namespace, autoload=autoload, index=index)

    def load_model(self, model, dataset=None, namespace=None, autoload=True, index=0):
        namespace = self.model_namespace2str("" if namespace is None else namespace)
        return model(dataset=dataset, 
                model_name=namespace, 
                model_version=self.model_version, 
                check_point_path=self.check_point_path,
                group_name=self.group_name,
                autoload=autoload,
                **self.get_params(model.cls_name(), index))

    def train(self, others_models_args={}):
        default_params = others_models_args
        for classif in self.load_models():
            print("Training [{}]".format(classif.__class__.__name__))
            params = others_models_args.get(classif.cls_name(), [default_params]).pop(0)
            classif.train(**params)
        self.save_model()
    
    def all_clf_scores(self, measures=None):
        from operator import add
        try:
            return reduce(add, (classif.scores(measures=measures) 
                for classif in self.load_models() if hasattr(classif, 'scores')))
        except TypeError:
            return ListMeasure()

    def scores(self, measures=None, namespace=None):
        return self.all_clf_scores(measures=measures)

    def print_confusion_matrix(self, namespace=None):
        from operator import add
        list_measure = reduce(add, (classif.confusion_matrix() 
            for classif in self.load_models()))
        classifs_reader = self.load_models()
        classif = classifs_reader.next()
        list_measure.print_matrix(classif.base_labels)

    def add_params(self, model_cls, index, **params):
        self.params.setdefault(model_cls, {index: params})
        self.params[model_cls][index] = params

    def get_params(self, model_cls, index):
        params = self.params.get(model_cls, {})
        return params.get(index, {})

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

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        from ml.layers import IterLayer
        def iter_():
            for classif in self.load_models():
                yield classif.predict(data, raw=raw, transform=transform, chunk_size=chunk_size)
        return IterLayer(iter_())

    def destroy(self):
        from ml.utils.files import rm
        for clf in self.load_models():
            clf.destroy()
        rm(self.get_model_path()+".xmeta")

    def _metadata(self, score=None):
        if self.dataset is not None:
            dataset_path = self.dataset.dataset_path
            dataset_name = self.dataset.name
        else:
            dataset_path = None
            dataset_name = None
        return {
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "models": self.clf_models_namespace,
                "model_name": self.model_name,
                "individual_ds": self.individual_ds,
                "score": score}

    def save_model(self):
        self.clf_models_namespace = self.string_namespaces(self.classifs)
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.save_meta()


class Ensemble(Grid):
    def scores(self, measures=None, all_clf=True):
        list_measure = ListMeasure()
        list_measure.calc_scores(self.__class__.__name__, 
                                self.predict, 
                                self.dataset.test_data, 
                                self.dataset.test_labels[:],
                                labels2classes_fn=self.numerical_labels2classes, 
                                measures=measures)
        if all_clf is True:
            return list_measure + self.all_clf_scores(measures=measures)
        else:
            return list_measure

    def numerical_labels2classes(self, labels):
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            return self.le.inverse_transform(np.argmax(labels, axis=1))
        else:
            return self.le.inverse_transform(labels.astype('int'))


class EnsembleLayers(DataDrive):
    def __init__(self, dataset=None, **kwargs):
        super(EnsembleLayers, self).__init__(**kwargs)
        self.fn_output = None
        if dataset is None:
            self.reload()
        else:
            self.layers = []
            self.dataset = dataset

    def add(self, ensemble):
        self.layers.append(ensemble)

    def train(self, others_models_args):
        from ml.clf.utils import add_params_to_params

        initial_layer = self.layers[0]
        initial_layer.train(others_models_args=others_models_args[0])
        y_submission = initial_layer.predict(
            self.dataset.test_data, raw=True, transform=False, chunk_size=0)

        size = self.dataset.test_data.shape[0] * len(initial_layer.classifs[initial_layer.active_network()])
        for classif in initial_layer.load_models():
            num_labels = classif.num_labels
            break
        data = np.zeros((size, num_labels))
        labels = np.empty(size, dtype="|S1")
        i = 0
        for y in y_submission:
            for row, label in zip(y, self.dataset.test_labels):
                data[i] = row
                labels[i] = label
                i += 1
    
        #fixme: add a dataset chunk writer
        dataset = DataSetBuilder(dataset_path="/tmp/", rewrite=False)
        dataset.build_dataset(data, labels)
        second_layer = self.layers[1]
        second_layer.reset_dataset(dataset)

        others_models_args_c = add_params_to_params(second_layer.classifs, 
                                                    others_models_args,
                                                    n_splits=5)

        second_layer.train(others_models_args=others_models_args_c)

        self.clf_models_namespace = {
            "layer1": {
                "model": initial_layer.module_cls_name(),
                "model_name": initial_layer.model_name,
                "model_version": initial_layer.model_version,
                "check_point_path": initial_layer.check_point_path
            },
            "layer2": {
                "model": second_layer.module_cls_name(),
                "model_name": second_layer.model_name,
                "model_version": second_layer.model_version,
                "check_point_path": second_layer.check_point_path
            }
        }
        self.save_model()
        self.reload()
        dataset.destroy()

    def scores(self, measures=None):
        if self.fn_output is not None:
            list_measure = ListMeasure()
            list_measure.calc_scores(self.model_name, 
                                    self.predict, 
                                    self.dataset.test_data, 
                                    self.dataset.test_labels[:],
                                    labels2classes_fn=self.numerical_labels2classes, 
                                    measures=measures)
            return list_measure
        else:
            return ListMeasure()

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

    def _metadata(self):
        list_measure = self.scores()
        return {
            "dataset_path": self.dataset.dataset_path,
            "dataset_name": self.dataset.name,
            "models": self.clf_models_namespace,
            "output": self.fn_output,
            "score": list_measure.measures_to_dict()}

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        initial_layer = self.layers[0]
        y_submission = initial_layer.predict(data, raw=True, transform=True, 
            chunk_size=chunk_size)
        second_layer = self.layers[1]
        output = second_layer.predict(y_submission, raw=raw, transform=transform, 
            chunk_size=chunk_size)
        return list(self.fn_output(*output)).pop()

    def destroy(self):
        for layer in self.layers:
            layer.destroy()

    def reload(self):
        meta = self.load_meta()
        models = meta.get('models', {})

        model_1 = locate(models["layer1"]["model"])
        classif_1 = model_1([], 
            model_name=models["layer1"]["model_name"], 
            model_version=models["layer1"]["model_version"],
            check_point_path=models["layer1"]["check_point_path"])

        model_2 = locate(models["layer2"]["model"])
        classif_2 = model_2([],
            model_name=models["layer2"]["model_name"], 
            model_version=models["layer2"]["model_version"],
            check_point_path=models["layer2"]["check_point_path"])

        self.layers = []
        self.add(classif_1)
        self.add(classif_2)
        self.output(meta["output"])

        return models

    def output(self, fn):
        self.fn_output = fn


class Boosting(Ensemble):
    def __init__(self, classifs, weights=None, election='best', num_max_clfs=1, 
            **kwargs):
        super(Boosting, self).__init__(classifs, meta_name="boosting", **kwargs)
        if len(classifs) > 0:
            self.weights = self.set_weights(0, self.classifs[self.active_network()], weights)
            self.election = election
            self.num_max_clfs = num_max_clfs
        else:
            meta = self.load_meta()
            self.weights = self.get_boosting_values(meta['weights'])
            self.election = meta["election"]
            self.model_name = meta["model_name"]

        for classif in self.load_models():
            self.le = classif.le
            break

    def get_boosting_values(self, weights):
        return {index: w for index, w in enumerate(weights)}

    def avg_prediction(self, predictions, weights, uncertain=True):
        from ml.layers import IterLayer
        if uncertain is False:
            return IterLayer.max_counter(predictions, weights=weights)
        else:
            predictors = IterLayer(predictions) * weights
            return IterLayer.avg(predictors, sum(weights))

    def train(self, batch_size=128, num_steps=1, only_voting=False):
        if only_voting is False:
            for classif in self.load_models():
                print("Training [{}]".format(classif.__class__.__name__))
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

    def _metadata(self, score=None):
        list_measure = self.scores(all_clf=False)
        meta = super(Boosting, self)._metadata(score=list_measure.measures_to_dict())
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


class Stacking(Ensemble):
    def __init__(self, classifs, n_splits=2, **kwargs):
        super(Stacking, self).__init__(classifs, meta_name="stacking", **kwargs)
        self.dataset_blend = None
        self.n_splits = n_splits
        self.ext = "pkl"
        if len(classifs) == 0:
            meta = self.load_meta()
            self.model_name = meta["model_name"]

        for classif in self.load_models():
            self.le = classif.le
            self.num_labels = classif.num_labels
            break

    def load_blend(self):
        from sklearn.externals import joblib
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.dataset_blend = joblib.load('{}.pkl'.format(path))

    def _metadata(self):
        list_measure = self.scores(all_clf=False)
        return super(Stacking, self)._metadata(score=list_measure.measures_to_dict())

    def save_model(self, dataset_blend):
        from sklearn.externals import joblib
        self.clf_models_namespace = self.string_namespaces(self.classifs)
        if self.check_point_path is not None:
            path = self.make_model_file()            
            joblib.dump(dataset_blend, '{}.{}'.format(path, self.ext))
            self.save_meta()

    def train(self, batch_size=128, num_steps=1):
        num_classes = len(self.dataset.labels_info().keys())
        n_splits = self.n_splits if num_classes < self.n_splits else num_classes
        size = self.dataset.shape[0]
        namespace = self.active_network()
        dataset_blend_train = np.zeros((size, len(self.classifs[namespace]), num_classes))
        dataset_blend_labels = np.zeros((size, len(self.classifs[namespace]), 1))
        for j, clf in enumerate(self.load_models()):
            print("Training [{}]".format(clf.__class__.__name__))
            init_r = 0
            clf.train(batch_size=batch_size, num_steps=num_steps, n_splits=n_splits)
            y_submission = np.asarray(list(
                clf.predict(clf.dataset.test_data, raw=True, transform=False, chunk_size=0)))
            if len(clf.dataset.test_labels.shape) > 1:
                test_labels = np.argmax(clf.dataset.test_labels, axis=1)
            else:
                test_labels = clf.dataset.test_labels[:]
            r = init_r + clf.dataset.test_data.shape[0]
            dataset_blend_train[init_r:r, j] = y_submission
            dataset_blend_labels[init_r:r, j] = test_labels.reshape(-1, 1)
            init_r = r
        dataset_blend_train = dataset_blend_train.reshape(dataset_blend_train.shape[0], -1)
        dataset_blend_labels = dataset_blend_labels.reshape(dataset_blend_labels.shape[0], -1)
        self.save_model(np.append(dataset_blend_train, dataset_blend_labels, axis=1))
        self.reload()

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        from sklearn.linear_model import LogisticRegression
        from ml.layers import IterLayer

        if self.dataset_blend is None:
            self.load_blend()
        
        namespace = self.active_network()
        size = data.shape[0]
        dataset_blend_test = np.zeros((size, len(self.classifs[namespace]), self.num_labels))
        for j, clf in enumerate(self.load_models()):
            y_predict = np.asarray(list(
                clf.predict(data, raw=True, transform=transform, chunk_size=0)))
            dataset_blend_test[:, j] = y_predict

        clf = LogisticRegression()
        columns = len(self.classifs[namespace]) * self.num_labels
        clf.fit(self.dataset_blend[:,:columns], self.dataset_blend[:,columns])
        return IterLayer(clf.predict_proba(dataset_blend_test.reshape(dataset_blend_test.shape[0], -1)))

    def destroy(self):
        from ml.utils.files import rm
        for clf in self.load_models():
            clf.destroy()
        rm(self.get_model_path()+"."+self.ext)
        rm(self.get_model_path()+".xmeta")


class Bagging(Ensemble):
    def __init__(self, classif, classifs_rbm, **kwargs):
        super(Bagging, self).__init__(classifs_rbm, meta_name="bagging", **kwargs)
        if classif is None:
            meta = self.load_meta()
            self.model_name = meta["model_name"]
            classif = locate(meta["model_base"])
            self.classif = self.load_model(classif, namespace=self.meta_name)
        else:
            self.classif = self.load_model(classif, dataset=self.dataset, 
                                    namespace=self.meta_name)
        self.le = self.classif.le

    def _metadata(self, score=None):
        list_measure = self.scores(all_clf=False)
        meta = super(Bagging, self)._metadata(score=list_measure.measures_to_dict())
        meta["model_base"] = self.classif.module_cls_name()
        return meta

    def save_model(self):         
        self.clf_models_namespace = self.string_namespaces(self.classifs)
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.save_meta()

    def train(self, model_base_args={"batch_size": 128, "num_steps": 1}, 
            others_models_args={}):
        default_params = model_base_args
        for classif in self.load_models():
            print("Training [{}]".format(classif.__class__.__name__))
            params = others_models_args.get(classif.cls_name(), [default_params]).pop(0)
            classif.train(**params)
        
        print("Building features...")
        dsb = DataSetBuilder(name=self.classif.dataset.name+"."+self.meta_name, rewrite=True)
        dsb.build_dataset(
            self.prepare_data(self.classif.dataset.train_data, transform=False, chunk_size=256),
            self.classif.numerical_labels2classes(self.classif.dataset.train_labels[:]),
            test_data=self.prepare_data(self.classif.dataset.test_data, transform=False, 
                chunk_size=256),
            test_labels=self.classif.numerical_labels2classes(self.classif.dataset.test_labels[:]),
            validation_data=self.prepare_data(self.classif.dataset.validation_data, 
                transform=False, chunk_size=256),
            validation_labels=self.classif.numerical_labels2classes(
                self.classif.dataset.validation_labels[:]))
        self.classif.set_dataset(dsb, auto=False)
        print("Done...")
        self.classif.train(**model_base_args)
        self.save_model()
        self.reload()

    def prepare_data(self, data, transform=True, chunk_size=1):
        from ml.layers import IterLayer
        predictions = (
            classif.predict(data, raw=True, transform=transform, chunk_size=chunk_size)
            for classif in self.load_models())
        namespace = self.active_network()
        predictions = IterLayer.avg(predictions, len(self.classifs[namespace]), method="geometric")
        predictions = np.asarray(list(predictions))
        if len(data.shape) == 1:
            data = data[:].reshape(-1, 1)
        return np.append(data, predictions, axis=1)

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        data_model_base = self.prepare_data(data, transform=transform, chunk_size=chunk_size)
        return self.classif.predict(data_model_base, raw=raw, transform=False, chunk_size=chunk_size)

    def destroy(self):
        from ml.utils.files import rm
        self.classif.destroy()
        for clf in self.load_models():
            clf.destroy()
