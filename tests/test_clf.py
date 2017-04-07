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
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost

        X = np.asarray([1, 0]*1000)
        Y = X*1
        self.dataset = DataSetBuilder("test", dataset_path="/tmp/", rewrite=False)
        self.dataset.build_dataset(X, Y)
        others_models_args = {"RandomForest": [{"batch_size": 50, "num_steps": 100, "n_splits": 2}],
            "AdaBoost": [{"batch_size": 50, "num_steps": 100}]}

        self.classif = Grid({0: [RandomForest, AdaBoost]},
            dataset=self.dataset, 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif.train(others_models_args=others_models_args)

    def tearDown(self):
        self.dataset.destroy()
        self.classif.destroy()

    def test_load_meta(self):
        print(self.classif.load_meta())
        self.assertEqual(type(self.classif.load_meta()), type({}))
        self.classif.scores().print_scores()


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

        self.classif = Boosting({"0": [
            ExtraTrees,
            RandomForest,
            AdaBoost]},
            dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            weights=[3, 1],
            election='best-c',
            num_max_clfs=5)
        self.classif.train(num_steps=1)

    def tearDown(self):
        self.dataset.destroy()
        self.classif.destroy()

    def test_load_meta(self):
        self.assertEqual(type(self.classif.load_meta()), type({}))
        self.classif.scores().print_scores()


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

        self.classif = Stacking({"0": [
            ExtraTrees,
            RandomForest,
            AdaBoost]},
            dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            n_splits=3)
        self.classif.train(num_steps=1)

    def tearDown(self):
        self.dataset.destroy()
        self.classif.destroy()

    def test_load_meta(self):
        self.assertEqual(type(self.classif.load_meta()), type({}))
        self.classif.scores().print_scores()


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

        self.classif = Bagging(GradientBoost, {"0": [
            ExtraTrees,
            RandomForest,
            AdaBoost]},
            dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif.train()

    def tearDown(self):
        self.dataset.destroy()
        self.classif.destroy()

    def test_load_meta(self):
        from ml.clf.ensemble import Bagging
        self.assertEqual(type(self.classif.load_meta()), type({}))
        self.classif.scores().print_scores()


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

if __name__ == '__main__':
    unittest.main()
