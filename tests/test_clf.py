import unittest
import numpy as np


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
        from ml.ds import DataSetBuilder
        from ml.clf.extended.w_sklearn import RandomForest

        X = np.asarray([1, 0]*10)
        Y = X*1
        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            ltype='int', rewrite=True)
        self.dataset.build_dataset(X, Y)
        self.classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif.train(num_steps=1)

    def tearDown(self):
        self.dataset.destroy()
        self.classif.destroy()

    def test_only_is(self):
        self.assertEqual(self.classif.erroneous_clf(), None)
        c, _ , _ = self.classif.correct_clf()
        self.assertEqual(list(c.reshape(-1, 1)), [1, 0, 0, 1])

    def test_load_meta(self):
        self.assertEqual(type(self.classif.load_meta()), type({}))


class TestMLP(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.extended.w_tflearn import MLP

        X = np.asarray([1, 0]*1000)
        Y = X*1
        self.dataset = DataSetBuilder(name="test_mlp", dataset_path="/tmp/", 
            ltype='int', rewrite=True)
        self.dataset.build_dataset(X, Y)
        
        #self.classif = MLP(dataset=self.dataset, 
        #    model_name="test", 
        #    model_version="1",
        #    check_point_path="/tmp/")
        #self.classif.train(num_steps=2)

    def tearDown(self):
        self.dataset.destroy()

    def test_labels(self):
        pass
        #self.assertEqual(type(self.classif.load_meta()), type({}))


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
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        classif.train(others_models_args=self.others_models_args)
        classif.scores().print_scores()
        self.assertEqual(type(classif.load_meta()), type({}))

        classif = Grid({}, 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in classif.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        classif.destroy()

    def test_compose_grid(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost, KNN
        from ml.clf.extended.w_keras import FCNet
        from ml.clf.ensemble import Grid, EnsembleLayers

        classif_1 = Grid([RandomForest, AdaBoost],
            dataset=self.dataset, 
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/")

        classif_2 = Grid([AdaBoost, KNN],
            dataset=None, 
            model_name="test_grid1", 
            model_version="1",
            check_point_path="/tmp/")

        ensemble = EnsembleLayers( 
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/",
            dataset=self.dataset)
        ensemble.add(classif_1)
        ensemble.add(classif_2)

        ensemble.train([self.others_models_args])
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

        self.classif = Boosting([
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
        self.classif.train(num_steps=1)

    def tearDown(self):
        self.dataset.destroy()

    def test_load_meta(self):
        from ml.clf.ensemble import Boosting
        self.classif.scores().print_scores()
        self.assertEqual(type(self.classif.load_meta()), type({}))
        classif = Boosting([], 
            model_name="test_boosting", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in classif.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        classif.destroy()


class TestStacking(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.ensemble import Stacking
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.clf.extended.w_sklearn import ExtraTrees, AdaBoost

        X = np.asarray([1, 0]*1000)
        Y = X*1
        self.dataset = DataSetBuilder("test", dataset_path="/tmp/", rewrite=False)
        self.dataset.build_dataset(X, Y)

        self.classif = Stacking([
            ExtraTrees,
            RandomForest,
            AdaBoost],
            dataset=self.dataset, 
            model_name="test_stacking", 
            model_version="1",
            check_point_path="/tmp/",
            n_splits=3)
        self.classif.train(num_steps=1)

    def tearDown(self):
        self.dataset.destroy()

    def test_load_meta(self):
        from ml.clf.ensemble import Stacking
        self.classif.scores().print_scores()
        self.assertEqual(type(self.classif.load_meta()), type({}))

        classif = Stacking([], 
            model_name="test_stacking", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in classif.predict(self.dataset.data[:1], raw=True):
            print(list(p))

        classif.destroy()


class TestBagging(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.ensemble import Bagging
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.clf.extended.w_sklearn import ExtraTrees, AdaBoost, GradientBoost

        X = np.asarray([1, 0]*1000)
        Y = X*1
        self.dataset = DataSetBuilder("test", dataset_path="/tmp/", rewrite=False)
        self.dataset.build_dataset(X, Y)

        self.classif = Bagging(GradientBoost, [
            ExtraTrees,
            RandomForest,
            AdaBoost],
            dataset=self.dataset, 
            model_name="test_bagging", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif.train()

    def tearDown(self):
        self.dataset.destroy()

    def test_load_meta(self):
        from ml.clf.ensemble import Bagging
        self.classif.scores().print_scores()
        self.assertEqual(type(self.classif.load_meta()), type({}))

        classif = Bagging(None, [], 
            model_name="test_bagging", 
            model_version="1",
            check_point_path="/tmp/")
        
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
                check_point_path="/tmp/",
                params={'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'})
            classif.train(num_steps=1)
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


class TestLayers(unittest.TestCase):

    def predict(self, data):
        for e in data:
            yield e + 1 

    def chunks(self, data, chunk_size=2):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            for p in self.predict(chunk):
                yield p

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
        predictor = predictor - predictor
        print(list(predictor))


if __name__ == '__main__':
    unittest.main()
