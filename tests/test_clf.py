import unittest
import numpy as np

from ml.ds import DataLabel
from ml.clf.extended.w_sklearn import RandomForest
np.random.seed(0)


def mulp(row, fmtypes=None):
    return row * 2


class TestSKL(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(100, 10)
        self.Y = self.X[:,0] > .5
        self.dataset = DataLabel(name="test", dataset_path="/tmp/", 
            ltype='int', rewrite=True)
        with self.dataset:
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
        scores_table = classif.scores2table()
        classif.destroy()
        self.assertEqual(scores_table.headers, ['', 'f1', 'auc', 'recall', 'precision', 
            'logloss', 'accuracy'])

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
        original_dataset = DataLabel("test", dataset_path="/tmp/", rewrite=True, ltype='int')
        with original_dataset:
            original_dataset.build_dataset(X, Y)
        self.original_dataset = original_dataset

    def tearDown(self):
        self.original_dataset.destroy()

    def test_load_meta(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.ensemble import Grid

        dataset = DataLabel(name="testdl", dataset_path="/tmp/", ltype='int', rewrite=True)
        with self.original_dataset, dataset:
            dataset.build_dataset(self.original_dataset.data[:], 
                                self.original_dataset.labels[:])

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=dataset)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=dataset)
        rf.train()
        ab.train()

        classif = Grid([rf, ab],
            dataset=self.original_dataset, 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        classif.calc_scores = False
        classif.output("avg")
        classif.save_model()
        meta = classif.load_meta()
        self.assertItemsEqual(meta.keys(), classif._metadata(calc_scores=False).keys())

        classif = Grid( 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        with self.original_dataset:
            data = self.original_dataset.data[:1]

        for p in classif.predict(data, raw=True, transform=True):
            self.assertEqual(len(list(p)), 2)

        classif.destroy()

    def test_grid_gini_measure(self):
        from ml.clf.ensemble import Grid
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.utils.numeric_functions import gini_normalized
        from ml.clf.measures import Measure

        metrics = Measure.make_metrics(None)
        metrics.add(gini_normalized, greater_is_better=True, uncertain=True)

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=self.original_dataset)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset)
        rf.train()
        ab.train()
        classif = Grid([rf, ab],
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/",
            metrics=metrics)

        classif.output(lambda x, y: (x + y) / 2)
        classif.save_model()
        self.assertEqual(len(classif.scores2table().measures[0]), 8)
        classif.destroy()

    def test_grid_transform(self):
        from ml.clf.ensemble import Grid
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.processing import Transforms

        with self.original_dataset:
            features = self.original_dataset.num_features()
            transforms = Transforms()
            transforms.add(mulp, o_features=features)
            dataset = self.original_dataset.convert(name="test_dataset", ltype='int',
                transforms=transforms, apply_transforms=True, dataset_path="/tmp/")
        
        rf = RandomForest(model_name="test_rf", model_version="1", dataset=dataset)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset)
        rf.train()
        ab.train()

        classif = Grid([rf, ab],
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/")

        classif.output(lambda x, y: (x + y) / 2)
        classif.save_model()   
        with self.original_dataset:
            data = self.original_dataset.data[:1]

        for p in classif.predict(data, raw=True, transform=True):
            self.assertEqual(len(list(p)), 2)
        dataset.destroy()
        classif.destroy()

    def test_compose_grid(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost, KNN
        from ml.clf.ensemble import Grid, EnsembleLayers

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=self.original_dataset)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset)
        rf.train()
        ab.train()

        layer_1 = Grid([rf, ab], 
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/")

        knn = KNN(model_name="test_knn", model_version="1", autoload=False)
        ab2 = AdaBoost(model_name="test_ab_2", model_version="1", autoload=False)
        layer_2 = Grid([ab2, knn],
            model_name="test_grid1", 
            model_version="1",
            check_point_path="/tmp/")
        layer_2.output(lambda x, y: (x**.25) * .85 * y**.35)

        ensemble = EnsembleLayers( 
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/",
            raw_dataset=self.original_dataset)
        ensemble.add(layer_1)
        ensemble.add(layer_2)

        ensemble.train()
        t0 = layer_1.scores()
        t1 = ensemble.scores()
        all_ = (t0 + t1)
        all_.print_scores()
        self.assertEqual(len(all_.measures), 2)
        ensemble.destroy()

    def test_compose_grid_predict(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.extended.w_sklearn import LogisticRegression, KNN
        from ml.clf.ensemble import Grid, EnsembleLayers      
        from ml.processing import FitRobustScaler, Transforms

        transforms = Transforms()
        transforms.add(FitRobustScaler, type="column")
        with self.original_dataset:
            ds0 = self.original_dataset.add_transforms(transforms, name="ds_test_0")

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=ds0)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=ds0)
        rf.train()
        ab.train()

        classif_1 = Grid([rf, ab],
            model_name="test_grid0",            
            check_point_path="/tmp/",
            model_version="1")

        knn = KNN(model_name="test_knn", model_version="1", autoload=False)
        ab2 = AdaBoost(model_name="test_ab_2", model_version="1", autoload=False)
        classif_2 = Grid([knn, ab2],
            model_name="test_grid1",            
            check_point_path="/tmp/", 
            model_version="1")
        classif_2.output(lambda x, y: (x + y) / 2)

        ensemble = EnsembleLayers( 
            model_name="test_ensemble_grid", 
            model_version="1",            
            check_point_path="/tmp/",
            raw_dataset=self.original_dataset)

        ensemble.add(classif_1)
        ensemble.add(classif_2)
        ensemble.train()
        ensemble.scores().print_scores()

        ensemble = EnsembleLayers(
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        with self.original_dataset:
            data = self.original_dataset.data[:1]

        predict = ensemble.predict(data, raw=True)
        self.assertEqual(predict.shape[1], 2)

        ensemble.destroy()
        ds0.destroy()

    def test_ensemble_bagging(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost, KNN
        from ml.clf.extended.w_sklearn import LogisticRegression
        from ml.clf.ensemble import Grid, EnsembleLayers

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=self.original_dataset)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset)
        knn = KNN(model_name="test_knn", model_version="1", dataset=self.original_dataset)
        rf.train()
        ab.train()
        knn.train()

        classif_1 = Grid([rf, ab, knn],
            model_name="test_grid0",            
            check_point_path="/tmp/",
            model_version="1")
        classif_1.output("bagging")

        lr = LogisticRegression(model_name="test_lr", model_version="1", autoload=False)
        classif_2 = Grid([lr],
            model_name="test_grid1",
            check_point_path="/tmp/", 
            model_version="1")
        classif_2.output(lambda x: x)

        ensemble = EnsembleLayers( 
            model_name="test_ensemble_grid", 
            model_version="1",            
            check_point_path="/tmp/",
            raw_dataset=self.original_dataset)

        ensemble.add(classif_1)
        ensemble.add(classif_2)
        ensemble.train()
        ensemble.scores().print_scores()

        ensemble = EnsembleLayers(
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        with self.original_dataset:
            data = self.original_dataset.data[:10]

        predict = ensemble.predict(data, raw=True)
        self.assertEqual(predict.shape[1], 2)
        self.assertEqual(len(list(predict)), 10)
        ensemble.destroy()


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
        
            classif = Xgboost(dataset=self.dataset, 
                model_name="test", 
                model_version="1",
                check_point_path="/tmp/",
                dtype="int",
                ltype="int")
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
        with self.dataset:
            predict = classif.predict(self.dataset.data, transform=False, raw=False)
            self.assertEqual(len(list(predict)), 100)
        classif.destroy()


class TestKFold(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = (self.X[:,0] > .5).astype(float)

    def tearDown(self):
        pass

    def test_predict(self):
        from ml.clf.extended.w_keras import FCNet
        dataset = DataLabel(dataset_path="/tmp/", rewrite=True)
        with dataset:
            dataset.build_dataset(self.X, self.Y)
        classif = FCNet(dataset=dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            dtype="float",
            ltype="float")
        classif.train(num_steps=1, batch_size=128, n_splits=4)

        classif = FCNet(
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        with dataset:
            predict = classif.predict(dataset.data)
            self.assertEqual(len(list(predict)), 10)
        classif.destroy()
        dataset.destroy()


if __name__ == '__main__':
    unittest.main()
