import unittest
import numpy as np

from ml.ds import DataSetBuilder
from ml.clf.extended.w_sklearn import RandomForest


def sizes(seq):
    return [len(list(row)) for row in seq]


class TestSeq(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(10, 10)

    def test_grouper_chunk_3(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(3, self.X)
        self.assertEqual(sizes(seq), [3, 3, 3, 1])

    def test_grouper_chunk_2(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(2, self.X)
        self.assertEqual(sizes(seq), [2, 2, 2, 2, 2])

    def test_grouper_chunk_10(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(10, self.X)
        self.assertEqual(sizes(seq), [10])

    def test_grouper_chunk_1(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(1, self.X)
        self.assertEqual(sizes(seq), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_grouper_chunk_7(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(7, self.X)
        self.assertEqual(sizes(seq), [7, 3])


class TestSKL(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(100, 10)
        self.Y = self.X[:,0] > .5
        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='int', rewrite=True)
        self.dataset.build_dataset(self.X, self.Y)

    def tearDown(self):
        self.dataset.destroy()

    def test_load_meta(self):
        classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(num_steps=1)
        self.assertEqual(type(classif.load_meta()), type({}))
        classif.destroy()

    def test_empty_load(self):
        classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(num_steps=1)

        classif = RandomForest(
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.destroy()

    def test_scores(self):        
        classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(num_steps=1)
        classif.scores().print_scores()

    def test_new_scores(self):
        from ml.utils.numeric_functions import gini_normalized
        from ml.clf.measures import Measure
        metrics = Measure.make_metrics(None)
        metrics.add(gini_normalized, greater_is_better=True, uncertain=True)
        classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            metrics=metrics)
        classif.train(num_steps=1)
        scores_table = classif.scores2table()
        self.assertEqual(scores_table.headers, ['', 'f1', 'auc', 'recall', 'precision', 
            'logloss', 'gini_normalized', 'accuracy'])
        classif.destroy()

    def test_scores2table(self):
        classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(num_steps=1)
        table = classif.scores2table()
        table.print_scores()


class TestGpy(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.extended.w_gpy import SVGPC, GPC

        X = np.asarray([1, 0]*10)
        Y = X*1
        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='int', rewrite=True)
        self.dataset.build_dataset(X, Y)
        self.classif = SVGPC(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif2 = GPC(dataset=self.dataset, 
            model_name="test2", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif.train(num_steps=2, batch_size=128)
        self.classif2.train(num_steps=2, batch_size=128)

    def tearDown(self):
        self.dataset.destroy()
        self.classif.destroy()
        self.classif2.destroy()

    def test_load_meta(self):
        list(self.classif.predict(self.dataset.data[:2]))
        list(self.classif2.predict(self.dataset.data[:2]))
        self.assertEqual(type(self.classif.load_meta()), type({}))
        self.assertEqual(type(self.classif2.load_meta()), type({}))


class TestGrid(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.ensemble import Grid

        X = np.asarray([1, 0]*1000)
        Y = X*1
        self.others_models_args = {"RandomForest": [{"batch_size": 50, "num_steps": 100, "n_splits": 2}],
            "AdaBoost": [{"batch_size": 50, "num_steps": 100}]}
        dataset = DataSetBuilder("test", dataset_path="/tmp/", rewrite=False)
        dataset.build_dataset(X, Y)
        self.dataset = dataset

    def tearDown(self):
        self.dataset.destroy()

    def test_load_meta(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.ensemble import Grid

        classif = Grid([RandomForest, AdaBoost],
            dataset=self.dataset, 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(others_models_args=self.others_models_args)
        classif.output("avg")
        classif.scores().print_scores()
        self.assertEqual(type(classif.load_meta()), type({}))

        classif = Grid({}, 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in classif.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        classif.destroy()

    def test_load_meta2(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.ensemble import Grid

        classif = Grid([(RandomForest, self.dataset), (AdaBoost, self.dataset)], 
            dataset=self.dataset,
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(others_models_args=self.others_models_args)
        classif.output("avg")
        classif.scores().print_scores()
        self.assertEqual(type(classif.load_meta()), type({}))

        classif = Grid({}, 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in classif.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        classif.destroy()

    def test_grid_fn(self):
        from ml.clf.ensemble import Grid
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost

        classif = Grid([RandomForest, AdaBoost],
            dataset=self.dataset, 
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/")

        classif.train(others_models_args=self.others_models_args)
        classif.output(lambda x, y: (x + y) / 2)        
        classif.scores().print_scores()

    def test_compose_grid(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost, KNN
        from ml.clf.ensemble import Grid, EnsembleLayers

        classif_1 = Grid([RandomForest, AdaBoost],
            dataset=self.dataset, 
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/")
        #classif_1.output("avg")
        classif_2 = Grid([AdaBoost, KNN],
            dataset=None, 
            model_name="test_grid1", 
            model_version="1",
            check_point_path="/tmp/")
        classif_2.output(lambda x, y: (x**.25) * .85 * y**.35)

        ensemble = EnsembleLayers( 
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/",
            raw_dataset=self.dataset)
        ensemble.add(classif_1)
        ensemble.add(classif_2)

        ensemble.train([self.others_models_args])
        s = classif_1.scores()
        st = ensemble.scores()
        (s + st).print_scores()
        ensemble.destroy()

    def test_compose_grid_predict(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.extended.w_sklearn import LogisticRegression, KNN
        from ml.clf.ensemble import Grid, EnsembleLayers      
        from ml.processing import FitRobustScaler, Transforms

        transforms = Transforms()
        transforms.add(FitRobustScaler, type="column")
        ds0 = self.dataset.add_transforms(transforms, name="ds_test_0")

        classif_1 = Grid([(RandomForest, ds0), (KNN, self.dataset)],
            model_name="test_grid0",            
            check_point_path="/tmp/",
            model_version="1")

        classif_2 = Grid([AdaBoost, LogisticRegression],
            dataset=None, 
            model_name="test_grid1",            
            check_point_path="/tmp/", 
            model_version="1")
        classif_2.output(lambda x, y: (x + y) / 2)

        ensemble = EnsembleLayers( 
            model_name="test_ensemble_grid", 
            model_version="1",            
            check_point_path="/tmp/",
            raw_dataset=self.dataset)

        ensemble.add(classif_1)
        ensemble.add(classif_2)
        others_models_args = {"RandomForest": [{"n_splits": 2}]}
        ensemble.train([others_models_args])
        ensemble.scores().print_scores()

        ensemble = EnsembleLayers(
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in ensemble.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        ensemble.destroy()
        ds0.destroy()

    def test_ensemble_bagging(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost, KNN
        from ml.clf.extended.w_sklearn import LogisticRegression
        from ml.clf.ensemble import Grid, EnsembleLayers

        classif_1 = Grid([RandomForest, AdaBoost, KNN],
            dataset=self.dataset,
            model_name="test_grid0",            
            check_point_path="/tmp/",
            model_version="1")
        classif_1.output("bagging")

        classif_2 = Grid([LogisticRegression],
            dataset=None, 
            model_name="test_grid1",
            check_point_path="/tmp/", 
            model_version="1")
        classif_2.output(lambda x: x)

        ensemble = EnsembleLayers( 
            model_name="test_ensemble_grid", 
            model_version="1",            
            check_point_path="/tmp/",
            raw_dataset=self.dataset)

        ensemble.add(classif_1)
        ensemble.add(classif_2)
        others_models_args = {"RandomForest": [{"n_splits": 2}]}
        ensemble.train([others_models_args])
        ensemble.scores().print_scores()

        ensemble = EnsembleLayers(
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in ensemble.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        ensemble.destroy()


class TestBoosting(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.ensemble import Boosting
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.clf.extended.w_sklearn import ExtraTrees, AdaBoost

        X = np.asarray([1, 0]*1000)
        Y = X*1
        self.dataset = DataSetBuilder("test", dataset_path="/tmp/", rewrite=False)
        self.dataset.build_dataset(X, Y)

        classif = Boosting([
            ExtraTrees,
            RandomForest,
            AdaBoost],
            dataset=self.dataset, 
            model_name="test_boosting", 
            model_version="1",
            check_point_path="/tmp/",
            weights=[3, 1],
            election='best-c',
            num_max_clfs=5)
        classif.train(num_steps=1)

    def tearDown(self):
        self.dataset.destroy()

    def test_load_meta(self):
        from ml.clf.ensemble import Boosting
        
        classif = Boosting([], 
            model_name="test_boosting", 
            model_version="1",
            check_point_path="/tmp/")
        
        self.assertEqual(type(classif.load_meta()), type({}))

        for p in classif.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        classif.destroy()


class TestXgboost(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        X = np.asarray([1, 0]*10)
        Y = X*1
        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='int', rewrite=True)
        self.dataset.build_dataset(X, Y)
        try:
            from ml.clf.extended.w_xgboost import Xgboost
        
            classif = Xgboost(dataset=self.dataset, 
                model_name="test", 
                model_version="1",
                check_point_path="/tmp/")
            params={'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
            classif.train(num_steps=1, model_params=params)
        except ImportError:
            return
        finally:
            pass

    def tearDown(self):
        self.dataset.destroy()

    def test_predict(self):
        try:
            from ml.clf.extended.w_xgboost import Xgboost
        except ImportError:
            return
        classif = Xgboost(
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.predict(self.dataset.test_data[0:1])
        classif.destroy()


class TestKFold(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.extended.w_keras import FCNet
        X = np.asarray([1, 0]*10)
        Y = X*1
        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='int', rewrite=True)
        self.dataset.build_dataset(X, Y)
        classif = FCNet(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(num_steps=1, batch_size=128, n_splits=4)

    def tearDown(self):
        self.dataset.destroy()

    def test_predict(self):
        from ml.clf.extended.w_keras import FCNet
        classif = FCNet(
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        classif.predict(self.dataset.test_data[0:1])
        classif.destroy()


class TestIterLayers(unittest.TestCase):

    def predict(self, data):
        for e in data:
            yield e + 1 

    def chunks(self, data, chunk_size=2):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            for p in self.predict(chunk):
                yield p

    def multi_round(self, X, *args):
        return [round(x, *args) for x in X]

    def test_operations_scalar(self):
        from ml.layers import IterLayer

        data = np.zeros((20, 2))
        predictor = IterLayer(self.chunks(data))
        predictor += 1
        predictor -= 1
        predictor *= 1
        predictor /= 1.
        predictor **= 1
        self.assertItemsEqual(np.asarray(list(predictor)).reshape(-1), np.zeros((40,)) + 1)

    def test_operations_stream(self):
        from ml.layers import IterLayer

        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2))
        predictor_0 = IterLayer(self.chunks(data_0, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(data_1, chunk_size=2))

        predictor = predictor_0 + predictor_1
        self.assertItemsEqual(np.asarray(list(predictor)).reshape(-1), np.zeros((40,)) + 1)

    def test_operations_list(self):
        from ml.layers import IterLayer
        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2))
        w = [1, 2]
        predictor_0 = IterLayer(self.chunks(data_0, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(data_1, chunk_size=2))

        predictor = IterLayer([predictor_0, predictor_1])
        predictors = predictor * w
        predictors = np.asarray(list(predictors))
        self.assertItemsEqual(np.asarray(list(predictors[0])).reshape(-1), np.zeros((40)))
        self.assertItemsEqual(np.asarray(list(predictors[1])).reshape(-1), np.zeros((40,)) + 2)

    def test_operations(self):
        from ml.layers import IterLayer

        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2))
        data_2 = np.zeros((20, 2)) + 3
        predictor_0 = IterLayer(self.chunks(data_0, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(data_1, chunk_size=2))
        predictor_2 = IterLayer(self.chunks(data_2, chunk_size=3))

        predictor = ((predictor_0**.65) * (predictor_1**.35) * .85) + predictor_2 * .15
        self.assertItemsEqual(np.asarray(list(predictor)).reshape(-1), np.zeros((40,)) + .6)

    def test_avg(self):
        from ml.layers import IterLayer

        predictor_0 = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(np.zeros((20, 2)) + 2, chunk_size=3))
        predictor_2 = IterLayer(self.chunks(np.zeros((20, 2)) + 3, chunk_size=3))

        predictor_avg = IterLayer.avg([predictor_0, predictor_1, predictor_2], 3)
        self.assertItemsEqual(np.asarray(list(predictor_avg)).reshape(-1), np.zeros((40,)) + 3)

        predictor_0 = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(np.zeros((20, 2)) + 2, chunk_size=3))
        predictor_2 = IterLayer(self.chunks(np.zeros((20, 2)) + 3, chunk_size=3))

        predictor_avg = IterLayer.avg([predictor_0, predictor_1, predictor_2], 3, method="geometric")
        predictor_avg = predictor_avg.compose(self.multi_round, 2)
        self.assertItemsEqual(np.asarray(list(predictor_avg)).reshape(-1), np.zeros((40,)) + 2.88)

    def test_max_counter(self):
        from ml.layers import IterLayer

        predictor_0 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"])
        predictor_1 = IterLayer(["1", "2", "2", "1", "2", "0", "0", "0"])
        predictor_2 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"])
        predictor_avg = IterLayer.max_counter([predictor_0, predictor_1, predictor_2])
        self.assertEqual(list(predictor_avg), ['0', '1', '0', '1', '2', '0', '1', '2'])

        weights = [1.5, 2, 1]
        predictor_0 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"])
        predictor_1 = IterLayer(["1", "2", "2", "1", "2", "0", "0", "0"])
        predictor_2 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"])        
        predictor_avg = IterLayer.max_counter([predictor_0, predictor_1, predictor_2], weights=weights)
        self.assertEqual(list(predictor_avg), ['0', '1', '0', '1', '2', '0', '1', '2'])

    def test_custom_fn(self):
        from ml.layers import IterLayer

        predictor = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3))
        predictor = predictor.compose(self.multi_round, 2)
        self.assertItemsEqual(np.asarray(list(predictor)).reshape(-1), np.zeros((40,)) + 2)

    def test_concat_fn(self):
        from ml.layers import IterLayer

        l0 = ["0", "1", "0", "1", "2", "0", "1", "2"]
        l1 = ["1", "2", "2", "1", "2", "0", "0", "0"]
        predictor_0 = IterLayer(l0)
        predictor_1 = IterLayer(l1)
        predictor = predictor_0.concat(predictor_1)
        self.assertItemsEqual(list(predictor), l0 + l1)

    def test_concat_n(self):
        from ml.layers import IterLayer

        l0 = np.zeros((20, 2)) + 1
        l1 = np.zeros((20, 2)) + 2
        l2 = np.zeros((20, 2)) + 3
        fl = np.concatenate((l0.reshape(-1) + 1, l1.reshape(-1) + 1, l2.reshape(-1) + 1))
        predictor_0 = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(np.zeros((20, 2)) + 2, chunk_size=3))
        predictor_2 = IterLayer(self.chunks(np.zeros((20, 2)) + 3, chunk_size=3))

        predictor = IterLayer.concat_n([predictor_0, predictor_1, predictor_2])
        self.assertItemsEqual(np.asarray(list(predictor)).reshape(-1), fl)

    def test_append_data_to_iter(self):
        from ml.layers import IterLayer

        data = [[0, 1, 0], [2, 3, 0], [4, 5, 0], [5, 6, 0]]
        data_i = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        iter_layer = IterLayer((e for e in data_i))
        iter_ce = iter_layer.concat_elems(data)

        for i, e in enumerate(iter_ce):
            self.assertItemsEqual(list(e), data_i[i] + data[i])

    def test_append_iter_to_iter(self):
        from ml.layers import IterLayer

        data_i2 = [[0, 1, 0], [2, 3, 0], [4, 5, 0], [5, 6, 0]]
        data_i1 = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        iter_layer_1 = IterLayer((e for e in data_i1))
        iter_layer_2 = IterLayer((e for e in data_i2))
        iter_ce = iter_layer_1.concat_elems(iter_layer_2)

        for i, e in enumerate(iter_ce):
            self.assertItemsEqual(list(e), data_i1[i] + data_i2[i])


if __name__ == '__main__':
    unittest.main()
