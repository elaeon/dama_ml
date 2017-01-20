import numpy as np
from ml.clf.wrappers import DataDrive, ListMeasure
from ml.ds import DataSetBuilder


class Grid(DataDrive):
    def __init__(self, classifs, model_name=None, dataset=None, 
            check_point_path=None, model_version=None, meta_name=""):
        super(Grid, self).__init__(
            check_point_path=check_point_path,
            model_version=model_version,
            model_name=model_name)        
        self.model = None
        self.dataset = dataset        
        self.meta_name = meta_name
        self.classifs = self.rename_namespaces(classifs)
        self.params = {}

    def load_namespaces(self, classifs, fn):
        namespaces = {}
        for namespace, models in classifs.items():
            for model in models:        
                namespaces.setdefault(namespace, [])
                namespaces[namespace].append(fn(model))
        return namespaces

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

    def load_models(self, dataset=None, autoload=True):
        for namespace, classifs in self.classifs.items():
            for classif in classifs:
                yield self.load_model(classif, dataset=dataset, 
                    namespace=namespace, autoload=autoload)

    def load_model(self, model, dataset=None, namespace=None, autoload=True):
        namespace = self.model_namespace2str("" if namespace is None else namespace)
        return model(dataset=dataset, 
                model_name=namespace, 
                model_version=self.model_version, 
                check_point_path=self.check_point_path,
                autoload=autoload,
                **self.get_params(model.cls_name()))

    def train(self, batch_size=128, num_steps=1):
        for classif in self.load_models(self.dataset):
            print("Training [{}]".format(classif.__class__.__name__))
            classif.train(batch_size=batch_size, num_steps=num_steps)
    
    def all_clf_scores(self, measures=None):
        from operator import add
        return reduce(add, (classif.scores(measures=measures) 
            for classif in self.load_models(self.dataset)))

    def scores(self, measures=None, namespace=None):
        return self.all_clf_scores(measures=measures)

    def print_confusion_matrix(self, namespace=None):
        from operator import add
        list_measure = reduce(add, (classif.confusion_matrix() 
            for classif in self.load_models(self.dataset)))
        classifs_reader = self.load_models(self.dataset)
        classif = classifs_reader.next()
        list_measure.print_matrix(classif.base_labels)

    def add_params(self, model_cls, **params):
        self.params.setdefault(model_cls, params)
        self.params[model_cls].update(params)

    def get_params(self, model_cls):
        return self.params.get(model_cls, {})

    def ordered_best_predictors(self, measure="logloss", operator=None):
        from functools import cmp_to_key
        list_measure = self.all_clf_scores(measures=measure)
        #print("MEASURES", list_measure.measures)
        column_measures = list_measure.get_measure(measure)
        #print(column_measures)
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

        return sorted(enum(column_measures), key=cmp_to_key(operator))

    def best_predictor_threshold(self, threshold=2, limit=3, measure="logloss", operator=None):
        best = self.ordered_best_predictors(measure=measure, operator=operator)
        base = best[0].elem
        return filter(lambda x: x[1] < threshold, 
            ((elem.counter, elem.elem/base) for elem in best if elem.elem is not None))[:limit]

    def best_predictor(self, measure="logloss", operator=None):
        best = self.ordered_best_predictors(measure=measure, operator=operator)[0].counter
        #self.load_model(self.classifs[best], info=False)
        return best

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        for classif in self.load_models(self.dataset):
            yield classif.predict(data, raw=raw, transform=transform, chunk_size=chunk_size)


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


class Boosting(Ensemble):
    def __init__(self, classifs, weights=None, election='best', num_max_clfs=1, **kwargs):
        kwargs["meta_name"] = "boosting"
        super(Boosting, self).__init__(classifs, **kwargs)
        if len(classifs) > 0:
            self.weights = self.set_weights(0, classifs["0"], weights)
            self.classifs_p = None
            self.weights_p = None
            self.election = election
            self.num_max_clfs = num_max_clfs
        else:
            meta = self.load_meta()
            self.set_boosting_values(meta.get('models', {}), meta['weights'])
            self.classifs = self.classifs_p
            self.weights = self.weights_p
            self.election = meta["election"]
            self.model_name = meta["model_name"]
            self.dataset = DataSetBuilder.load_dataset(
                meta["dataset_name"], 
                dataset_path=meta["dataset_path"])

        for classif in self.load_models(self.dataset):
            self.le = classif.le
            break

    def set_boosting_values(self, models, weights):
        from pydoc import locate
        self.classifs_p = self.load_namespaces(models, lambda x: locate(x))
        self.weights_p = {index: w for index, w in enumerate(weights)}

    def avg_prediction(self, predictions, weights, uncertain=True):
        from itertools import izip
        if uncertain is False:
            from ml.utils.numeric_functions import discrete_weight
            return discrete_weight(predictions, weights)
        else:
            from ml.utils.numeric_functions import arithmetic_mean
            predictions_iter = ((prediction * w for prediction in row_prediction)
                for w, row_prediction in izip(weights, predictions))
            return arithmetic_mean(predictions_iter, float(sum(weights)))

    def train(self, batch_size=128, num_steps=1, only_voting=False):
        if only_voting is False:
            for classif in self.load_models(self.dataset):
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

    def _metadata(self, score=None):
        list_measure = self.scores(all_clf=False)
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "models": self.clf_models_namespace,
                "weights": self.clf_weights,
                "model_name": self.model_name,
                "md5": self.dataset.md5(),
                "election": self.election,
                "score": list_measure.measures_to_dict()}

    def save_model(self, models, weights):
        self.clf_models_namespace = self.load_namespaces(
            models, 
            lambda x: x.module_cls_name())
        self.clf_weights = weights
        self.set_boosting_values(self.clf_models_namespace, self.clf_weights)
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
                    for classif in self.classifs_p[self.meta_name+".0"])
        weights = [w for i, w in sorted(self.weights_p.items(), key=lambda x:x[0])]
        predictions = (
            classif.predict(data, raw=raw, transform=transform, chunk_size=chunk_size)
            for classif in models)
        return self.avg_prediction(predictions, weights, uncertain=raw)

    def correlation_between_models(self, sort=False):
        from ml.utils.numeric_functions import le, pearsoncc
        from ml.utils.network import all_simple_paths_graph
        from itertools import combinations
        from ml.utils.order import order_from_ordered
        import networkx as nx

        best_predictors = []
        def predictions_fn():
            predictions = {}
            for index, _ in self.best_predictor_threshold(operator=le, limit=self.num_max_clfs):
                classif = self.load_model(self.classifs[self.meta_name+".0"][index], 
                    namespace=self.meta_name+".0")
                predictions[index] = np.asarray(list(classif.predict(
                    classif.dataset.test_data, raw=False, transform=False, chunk_size=258)))
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
        kwargs["meta_name"] = "stacking"
        super(Stacking, self).__init__(classifs, **kwargs)
        self.dataset_blend = None
        self.n_splits = n_splits
        if len(classifs) == 0:
            meta = self.load_meta()
            from pydoc import locate
            self.classifs = self.load_namespaces(
                meta.get('models', {}), lambda x: locate(x))
            self.iterations = meta["iterations"]
            self.model_name = meta["model_name"]
            self.dataset = DataSetBuilder.load_dataset(
                meta["dataset_name"], 
                dataset_path=meta["dataset_path"])
        else:
            self.iterations = 0

        for classif in self.load_models(self.dataset):
            self.le = classif.le
            break

    def load_blend(self):
        from sklearn.externals import joblib
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.dataset_blend = joblib.load('{}.pkl'.format(path))

    def _metadata(self):
        list_measure = self.scores(all_clf=False)
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "models": self.clf_models_namespace,
                "model_name": self.model_name,
                "md5": self.dataset.md5(),
                "iterations": self.iterations,
                "score": list_measure.measures_to_dict()}

    def save_model(self, dataset_blend):
        from sklearn.externals import joblib
        self.clf_models_namespace = self.load_namespaces(
            self.classifs, 
            lambda x: x.module_cls_name())
        if self.check_point_path is not None:
            path = self.make_model_file()            
            joblib.dump(dataset_blend, '{}.pkl'.format(path))
            self.save_meta()

    def train(self, batch_size=128, num_steps=1):
        #from sklearn.model_selection import StratifiedKFold
        from ml.ds import DataSetBuilderFold
        num_classes = len(self.dataset.labels_info().keys())
        n_splits = self.n_splits if num_classes < self.n_splits else num_classes
        #skf = StratifiedKFold(n_splits=n_splits)
        #dl = self.dataset.desfragment()
        size = self.dataset.shape[0]
        dataset_blend_train = np.zeros((size, len(self.classifs[self.meta_name+".0"]), num_classes))
        dataset_blend_labels = np.zeros((size, len(self.classifs[self.meta_name+".0"]), 1))
        for j, clf in enumerate(self.load_models(autoload=False)):
            print("Training [{}]".format(clf.__class__.__name__))
            dsbf = DataSetBuilderFold(n_splits=n_splits)
            dsbf.build_dataset(dataset=self.dataset)
            #for i, (train, test) in enumerate(skf.split(dl.data, dl.labels)):
            for i, dataset in enumerate(dsbf.folds())
                #validation_index = int(train.shape[0] * .1)
                #validation = train[:validation_index]
                #train = train[validation_index:]
                print("Fold", i)
                clf.set_dataset(dataset)
                #clf.set_dataset(data[train], data[test], data[validation],
                #    labels[train], labels[test], labels[validation])
                clf.train(batch_size=batch_size, num_steps=num_steps)
                y_submission = np.asarray(list(
                    clf.predict(clf.dataset.test_data, raw=True, transform=False, chunk_size=0)))
                if len(clf.dataset.test_labels.shape) > 1:
                    test_labels = np.argmax(clf.dataset.test_labels, axis=1)
                else:
                    test_labels = clf.dataset.test_labels
                dataset_blend_train[test, j] = y_submission
                dataset_blend_labels[test, j] = test_labels.reshape(-1, 1)
        dl.destroy()
        self.iterations = i + 1
        dataset_blend_train = dataset_blend_train.reshape(dataset_blend_train.shape[0], -1)
        dataset_blend_labels = dataset_blend_labels.reshape(dataset_blend_labels.shape[0], -1)
        self.save_model(np.append(dataset_blend_train, dataset_blend_labels, axis=1))

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        from sklearn.linear_model import LogisticRegression
        if self.dataset_blend is None:
            self.load_blend()

        clf_b = self.load_models(self.dataset).next()
        size = data.shape[0]
        num_classes = clf_b.num_labels
        dataset_blend_test = np.zeros((size, len(self.classifs[self.meta_name+".0"]), num_classes))
        for j, clf in enumerate(self.load_models(self.dataset)):
            dataset_blend_test_j = np.zeros((size, self.iterations, num_classes))
            count = 0     
            while count < self.iterations:
                y_predict = np.asarray(list(
                    clf.predict(data, raw=True, transform=transform, chunk_size=0)))
                dataset_blend_test_j[:, count] = y_predict
                count += 1
            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

        clf = LogisticRegression()
        columns = len(self.classifs[self.meta_name+".0"]) * num_classes
        clf.fit(self.dataset_blend[:,:columns], self.dataset_blend[:,columns])
        return clf.predict_proba(dataset_blend_test.reshape(dataset_blend_test.shape[0], -1))


class Bagging(Ensemble):
    def __init__(self, classif, classifs_rbm, **kwargs):
        kwargs["meta_name"] = "bagging"
        super(Bagging, self).__init__(classifs_rbm, **kwargs)
        if classif is None:          
            from pydoc import locate
            meta = self.load_meta()
            self.classifs = self.load_namespaces(
                meta.get('models', {}), lambda x: locate(x))
            self.classif = locate(meta["model_base"])
            self.model_name = meta["model_name"]
            self.dataset = DataSetBuilder.load_dataset(
                meta["dataset_name"], 
                dataset_path=meta["dataset_path"])
        else:
            self.classif = classif

    def _metadata(self, score=None):
        list_measure = self.scores()
        return {"dataset_path": self.dataset.dataset_path,
                "dataset_name": self.dataset.name,
                "models": self.clf_models_namespace,
                "model_base": self.classif.module_cls_name(),
                "model_name": self.model_name,
                "md5": self.dataset.md5(),
                "score": list_measure.measures_to_dict()}

    def save_model(self):         
        self.clf_models_namespace = self.load_namespaces(
            self.classifs, 
            lambda x: x.module_cls_name())
        if self.check_point_path is not None:
            path = self.make_model_file()
            self.save_meta()

    def train(self, batch_size=128, num_steps=1):
        for classif in self.load_models(self.dataset):
            print("Training [{}]".format(classif.__class__.__name__))
            classif.train(batch_size=batch_size, num_steps=num_steps)
        model_base = self.load_model(self.classif, dataset=self.dataset, 
                                    info=False, namespace=self.meta_name)
        self.le = model_base.le
        print("Building features...")
        model_base.set_dataset_from_raw(
            self.prepare_data(model_base.dataset.train_data, transform=False, chunk_size=256), 
            self.prepare_data(model_base.dataset.test_data, transform=False, chunk_size=256), 
            self.prepare_data(model_base.dataset.valid_data, transform=False, chunk_size=256),
            model_base.numerical_labels2classes(model_base.dataset.train_labels), 
            model_base.numerical_labels2classes(model_base.dataset.test_labels), 
            model_base.numerical_labels2classes(model_base.dataset.valid_labels),
            save=True,
            dataset_name=model_base.dataset.name+"."+self.meta_name)
        print("Done...")
        model_base.train(batch_size=batch_size, num_steps=num_steps)
        self.save_model()

    def prepare_data(self, data, transform=True, chunk_size=1):
        from ml.utils.numeric_functions import geometric_mean
        predictions = (
            classif.predict(data, raw=True, transform=transform, chunk_size=chunk_size)
            for classif in self.load_models(self.dataset))
        predictions = np.asarray(list(geometric_mean(predictions, len(self.classifs[self.meta_name+".0"]))))
        return np.append(data, predictions, axis=1)

    def predict(self, data, raw=False, transform=True, chunk_size=1):
        model_base = self.load_model(self.classif, info=True, namespace=self.meta_name)
        self.le = model_base.le
        data_model_base = self.prepare_data(data, transform=transform, chunk_size=chunk_size)
        return model_base.predict(data_model_base, raw=raw, transform=False, chunk_size=chunk_size)
