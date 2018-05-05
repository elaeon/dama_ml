import unittest
import numpy as np

from ml.ds import DataLabel
from ml.clf.extended.w_sklearn import RandomForest
np.random.seed(0)


def mulp(row):
    return row * 2


class TestSKL(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(100, 10)
        self.Y = self.X[:,0] > .5
        self.dataset = DataLabel(name="test", dataset_path="/tmp/", rewrite=True)
        with self.dataset:
            self.dataset.build_dataset(self.X, self.Y)

    def tearDown(self):
        self.dataset.destroy()

    def test_load_meta(self):
        classif = RandomForest(
            model_name="test_model",
            check_point_path="/tmp/")
        classif.set_dataset(self.dataset)
        classif.train(num_steps=1)
        classif.save(model_version="1")
        meta = classif.load_meta()
        self.assertEqual(meta["model"]["original_dataset_path"], "/tmp/")
        self.assertEqual(meta["train"]["model_version"], "1")
        classif.destroy()

    def test_empty_load(self):
        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        classif.set_dataset(self.dataset)
        classif.train(num_steps=1)
        classif.save(model_version="1")

        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        classif.load(model_version="1")
        self.assertEqual(classif.original_dataset_path, "/tmp/")
        classif.destroy()

    def test_scores(self):        
        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        classif.set_dataset(self.dataset)
        classif.train(num_steps=1)
        classif.save(model_version="1")
        scores_table = classif.scores2table()
        classif.destroy()
        self.assertEqual(scores_table.headers, ['', 'f1', 'auc', 'recall', 'precision', 
            'logloss', 'accuracy'])
        self.assertEqual(scores_table.measures[0][5] < 1, True)

    def test_new_scores(self):
        from ml.utils.numeric_functions import gini_normalized
        from ml.clf.measures import Measure
        metrics = Measure.make_metrics(None)
        metrics.add(gini_normalized, greater_is_better=True, output='uncertain')
        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/",
            metrics=metrics)
        classif.set_dataset(self.dataset)
        classif.train(num_steps=1)
        classif.save(model_version="1")
        scores_table = classif.scores2table()
        self.assertEqual(scores_table.headers, ['', 'f1', 'auc', 'recall', 'precision', 
            'logloss', 'gini_normalized', 'accuracy'])
        classif.destroy()

    def test_predict(self):
        from ml.processing import Transforms
        t = Transforms()
        t.add(mulp)
        X = np.random.rand(100, 1)
        Y = X[:,0] > .5
        dataset = DataLabel(name="test_0", dataset_path="/tmp/", 
            rewrite=True, transforms=t, apply_transforms=True)
        with dataset:
            dataset.build_dataset(X, Y)
        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        classif.set_dataset(dataset)
        classif.train()
        classif.save(model_version="1")
        values = np.asarray([[1], [2], [.4], [.1], [0], [1]])
        self.assertItemsEqual(classif.predict(values).to_memory(), [True, True, False, False, False, True])
        self.assertEqual(len(classif.predict(values).to_memory()), 6)
        self.assertEqual(len(classif.predict(np.asarray(values), chunks_size=0).to_memory()), 6)
        dataset.destroy()
        classif.destroy()

    def test_load(self):
        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        classif.set_dataset(self.dataset)
        classif.train(num_steps=1)
        classif.save(model_version="1")

        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        classif.load(model_version="1")
        classif.train(num_steps=1)
        classif.save(model_version="2")

        classif = RandomForest( 
            model_name="test", 
            check_point_path="/tmp/")
        classif.load(model_version="2")
        with self.dataset:
            values = self.dataset.data[:6]
        self.assertEqual(len(classif.predict(values).to_memory()), 6)
        classif.destroy()

#class TestGpy(unittest.TestCase):
#    def setUp(self):
#        from ml.ds import DataSetBuilder
#        from ml.clf.extended.w_gpy import SVGPC, GPC

#        X = np.random.rand(100, 10)
#        Y = (X[:,0] > .5).astype(int)
#        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
#            ltype='int', rewrite=True)
#        self.dataset.build_dataset(X, Y)
#        self.classif = SVGPC(dataset=self.dataset, 
#            model_name="test", 
#            model_version="1",
#            check_point_path="/tmp/")
#        self.classif2 = GPC(dataset=self.dataset, 
#            model_name="test2", 
#            model_version="1",
#            check_point_path="/tmp/")
#        self.classif.train(num_steps=2, batch_size=128)
#        self.classif2.train(num_steps=2, batch_size=128)

#    def tearDown(self):
#        self.dataset.destroy()
#        self.classif.destroy()
#        self.classif2.destroy()

#    def test_load_meta(self):
#        list(self.classif.predict(self.dataset.data[:2]))
#        list(self.classif2.predict(self.dataset.data[:2]))
#        self.assertEqual(type(self.classif.load_meta()), type({}))
#        self.assertEqual(type(self.classif2.load_meta()), type({}))


class TestGrid(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataLabel
        from ml.clf.ensemble import Grid

        X = np.random.rand(100, 10)
        Y = (X[:,0] > .5).astype(int)
        original_dataset = DataLabel("test", dataset_path="/tmp/", rewrite=True)
        with original_dataset:
            original_dataset.build_dataset(X, Y)
        self.original_dataset = original_dataset

    def tearDown(self):
        self.original_dataset.destroy()

    def test_load_meta(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.ensemble import Grid

        dataset = DataLabel(name="testdl", dataset_path="/tmp/", rewrite=True)
        with self.original_dataset, dataset:
            dataset.build_dataset(self.original_dataset.data[:], 
                                self.original_dataset.labels[:])

        rf = RandomForest(model_name="test_rf")
        rf.set_dataset(dataset)
        ab = AdaBoost(model_name="test_ab")
        ab.set_dataset(dataset)
        rf.train()
        ab.train()
        rf.save(model_version="1")
        ab.save(model_version="1")

        classif = Grid([rf, ab], 
            model_name="test_grid", 
            check_point_path="/tmp/")
        classif.output("avg")
        classif.save()
        meta = classif.load_meta()
        self.assertItemsEqual(meta.keys(), classif._metadata(calc_scores=False).keys())

        classif = Grid( 
            model_name="test_grid", 
            check_point_path="/tmp/")
        
        classif.load()
        with self.original_dataset:
            data = self.original_dataset.data[:1]

        for p in classif.predict(data, output='uncertain', transform=True):
            self.assertEqual(p.shape, (1, 2))
        classif.destroy()

    def test_grid_gini_measure(self):
        from ml.clf.ensemble import Grid
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.utils.numeric_functions import gini_normalized
        from ml.clf.measures import Measure

        metrics = Measure.make_metrics(None)
        metrics.add(gini_normalized, greater_is_better=True, output='uncertain')

        rf = RandomForest(model_name="test_rf")
        rf.set_dataset(self.original_dataset)
        ab = AdaBoost(model_name="test_ab")
        ab.set_dataset(self.original_dataset)
        rf.train()
        rf.save(model_version="1")
        ab.train()
        ab.save(model_version="1")

        classif = Grid([rf, ab],
            model_name="test_grid0", 
            check_point_path="/tmp/",
            metrics=metrics)
        classif.output(lambda x, y: (x + y) / 2)
        classif.save()
        self.assertEqual(len(classif.scores2table().measures[0]), 8)
        classif.destroy()

    def test_grid_transform(self):
        from ml.clf.ensemble import Grid
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.processing import Transforms

        with self.original_dataset:
            features = self.original_dataset.num_features()
            transforms = Transforms()
            transforms.add(mulp)
            dataset = self.original_dataset.convert(name="test_dataset",
                transforms=transforms, apply_transforms=True, dataset_path="/tmp/")
        
        rf = RandomForest(model_name="test_rf")
        rf.set_dataset(dataset)
        ab = AdaBoost(model_name="test_ab")
        ab.set_dataset(self.original_dataset)
        rf.train()
        rf.save(model_version="1")
        ab.train()
        ab.save(model_version="1")

        classif = Grid([rf, ab],
            model_name="test_grid0", 
            check_point_path="/tmp/")
        classif.output(lambda x, y: (x + y) / 2)
        classif.save()   
        with self.original_dataset:
            data = self.original_dataset.data[:1]

        for p in classif.predict(data, output=None, transform=True):
            self.assertEqual(type(p), np.dtype('int'))
        dataset.destroy()
        classif.destroy()

    #def test_ensemble_bagging(self):
    #    from ml.clf.extended.w_sklearn import RandomForest, AdaBoost, KNN
    #    from ml.clf.extended.w_sklearn import LogisticRegression
    #    from ml.clf.ensemble import Grid, EnsembleLayers

    #    rf = RandomForest(model_name="test_rf", model_version="1", dataset=self.original_dataset)
    #    ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset)
    #    knn = KNN(model_name="test_knn", model_version="1", dataset=self.original_dataset)
    #    rf.train()
    #    ab.train()
    #    knn.train()

    #    classif_1 = Grid([rf, ab, knn],
    #        model_name="test_grid0",            
    #        check_point_path="/tmp/",
    #        model_version="1")
    #    classif_1.output("bagging")

    #    lr = LogisticRegression(model_name="test_lr", model_version="1", autoload=False)
    #    classif_2 = Grid([lr],
    #        model_name="test_grid1",
    #        check_point_path="/tmp/", 
    #        model_version="1")
    #    classif_2.output(lambda x: x)

    #    ensemble = EnsembleLayers( 
    #        model_name="test_ensemble_grid", 
    #        model_version="1",            
    #        check_point_path="/tmp/",
    #        raw_dataset=self.original_dataset)

    #    ensemble.add(classif_1)
    #    ensemble.add(classif_2)
    #    ensemble.train()
    #    ensemble.scores().print_scores()

    #    ensemble = EnsembleLayers(
    #        model_name="test_ensemble_grid", 
    #        model_version="1",
    #        check_point_path="/tmp/")
        
    #    with self.original_dataset:
    #        data = self.original_dataset.data[:10]

    #    predict = ensemble.predict(data, raw=True)
    #    self.assertEqual(predict.shape[1], 2)
    #    self.assertEqual(len(list(predict)), 10)
    #    ensemble.destroy()


#class TestBoosting(unittest.TestCase):
#    def setUp(self):
#        from ml.ds import DataSetBuilder
#        from ml.clf.ensemble import Boosting
#        from ml.clf.extended.w_sklearn import RandomForest
#        from ml.clf.extended.w_sklearn import ExtraTrees, AdaBoost

#        X = np.asarray([1, 0]*1000)
#        Y = X*1
#        self.dataset = DataSetBuilder("test", dataset_path="/tmp/", rewrite=False)
#        self.dataset.build_dataset(X, Y)

#        classif = Boosting([
#            ExtraTrees,
#            RandomForest,
#            AdaBoost],
#            dataset=self.dataset, 
#            model_name="test_boosting", 
#            model_version="1",
#            check_point_path="/tmp/",
#            weights=[3, 1],
#            election='best-c',
#            num_max_clfs=5)
#        classif.train(num_steps=1)

#    def tearDown(self):
#        self.dataset.destroy()

#    def test_load_meta(self):
#        from ml.clf.ensemble import Boosting
        
#        classif = Boosting([], 
#            model_name="test_boosting", 
#            model_version="1",
#            check_point_path="/tmp/")
        
#        self.assertEqual(type(classif.load_meta()), type({}))

#        for p in classif.predict(self.dataset.data[:1], raw=True):
#            print(list(p))

#        classif.destroy()


class TestXgboost(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataLabel
        X = np.random.rand(100, 10)
        Y = (X[:,0] > .5).astype(int)
        self.dataset = DataLabel(name="test", dataset_path="/tmp/", rewrite=True)
        with self.dataset:
            self.dataset.build_dataset(X, Y)
        try:
            from ml.clf.extended.w_xgboost import Xgboost
            classif = Xgboost(
                model_name="test", 
                check_point_path="/tmp/")
            classif.set_dataset(self.dataset)
            params={'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
            classif.train(num_steps=1, model_params=params)
            classif.save(model_version="1")
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
            check_point_path="/tmp/")
        classif.load(model_version="1")
        with self.dataset:
            predict = classif.predict(self.dataset.data, transform=False, output=None)
            self.assertEqual(len(list(predict)), 100)
        classif.destroy()


class TestLightGBM(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataLabel
        X = np.random.rand(100, 10)
        Y = (X[:,0] > .5).astype(int)
        self.dataset = DataLabel(name="test", dataset_path="/tmp/", rewrite=True)
        with self.dataset:
            self.dataset.build_dataset(X, Y)
        try:
            from ml.clf.extended.w_lgb import LightGBM
            classif = LightGBM(
                model_name="test", 
                check_point_path="/tmp/")
            classif.set_dataset(self.dataset)
            self.params={'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
        'objective': 'binary', 'seed': 99, "verbosity": 0, "learning_rate": 0.1, 
        'boosting_type':"gbdt", 'max_bin': 255, 'num_leaves': 25, 'max_depth': 50, 
        'metric': 'binary_logloss'}
            self.num_steps = 10
            self.model_version = "1"
            classif.train(num_steps=self.num_steps, model_params=self.params)
            classif.save(model_version=self.model_version)
        except ImportError:
            return
        finally:
            pass

    def tearDown(self):
        self.dataset.destroy()

    def test_train_params(self):
        try:
            from ml.clf.extended.w_lgb import LightGBM
        except ImportError:
            return

        classif = LightGBM(
            model_name="test",
            check_point_path="/tmp/")
        classif.load(model_version=self.model_version)
        meta = classif.load_meta()
        self.assertItemsEqual(meta["train"]["params"].keys(), self.params.keys())
        self.assertEqual(meta["train"]["model_version"], self.model_version)
        self.assertEqual(meta["train"]["num_steps"], self.num_steps)
        self.assertEqual(len(meta["train"]["score"].keys()) > 0, True)

    def test_predict(self):
        try:
            from ml.clf.extended.w_lgb import LightGBM
        except ImportError:
            return
        classif = LightGBM(
            model_name="test",
            check_point_path="/tmp/")
        classif.load(model_version=self.model_version)
        with self.dataset:
            predict = classif.predict(self.dataset.data, transform=False, output=None)
            self.assertEqual(len(list(predict)), 100)
        classif.destroy()


class TestKFold(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(1000, 2)
        self.Y = (self.X[:,0] > .5).astype(float)

    def tearDown(self):
        pass

    def test_predict(self):
        #from ml.clf.extended.w_keras import FCNet
        dataset = DataLabel(dataset_path="/tmp/", rewrite=True)
        with dataset:
            dataset.build_dataset(self.X, self.Y)
        classif = RandomForest(
            model_name="test",
            check_point_path="/tmp/")
        classif.set_dataset(dataset)
        classif.train(num_steps=1, batch_size=128, n_splits=4)
        classif.save(model_version="1")

        classif = RandomForest(
            model_name="test",
            check_point_path="/tmp/")
        classif.load(model_version="1")
        with dataset:
            predict = classif.predict(dataset.data)
            self.assertEqual(len(list(predict)), 1000)
        classif.destroy()
        dataset.destroy()


if __name__ == '__main__':
    unittest.main()
