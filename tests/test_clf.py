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
            check_point_path="/tmp/",
            rewrite=True)
        classif.train(num_steps=1)
        self.assertEqual(type(classif.load_meta()), type({}))
        classif.destroy()

    def test_empty_load(self):
        classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            rewrite=True)
        classif.train(num_steps=1)

        classif = RandomForest(
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            rewrite=False)
        classif.destroy()

    def test_scores(self):        
        classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            rewrite=True)
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
            metrics=metrics,
            rewrite=True)
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
        from ml.ds import DataSetBuilder
        from ml.clf.ensemble import Grid

        X = np.random.rand(100, 10)
        Y = (X[:,0] > .5).astype(int)
        original_dataset = DataSetBuilder("test", dataset_path="/tmp/", rewrite=True, ltype='int')
        with original_dataset:
            original_dataset.build_dataset(X, Y)
        self.original_dataset = original_dataset

    def tearDown(self):
        self.original_dataset.destroy()

    def test_load_meta(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.ensemble import Grid

        dataset = DataLabel(name="testdl", dataset_path="/tmp/", rewrite=True, ltype='int')
        with self.original_dataset, dataset:
            dataset.build_dataset(self.original_dataset.data_validation[:], 
                                self.original_dataset.data_validation_labels[:])

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=dataset, rewrite=True)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=dataset, rewrite=True)
        rf.train()
        ab.train()

        classif = Grid([rf, ab],
            dataset=self.original_dataset, 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        classif.calc_scores = False
        classif.output("avg")
        classif.train()
        meta = classif.load_meta()
        self.assertItemsEqual(meta.keys(), classif._metadata(calc_scores=False).keys())

        classif = Grid( 
            model_name="test_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        with self.original_dataset:
            data = self.original_dataset.data[:1]

        for p in classif.predict(data, raw=True, transform=False):
            self.assertEqual(len(list(p)), 2)

        classif.destroy()

    def test_grid_gini_measure(self):
        from ml.clf.ensemble import Grid
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.utils.numeric_functions import gini_normalized
        from ml.clf.measures import Measure

        metrics = Measure.make_metrics(None)
        metrics.add(gini_normalized, greater_is_better=True, uncertain=True)

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=self.original_dataset, rewrite=True)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset, rewrite=True)
        rf.train()
        ab.train()
        classif = Grid([rf, ab],
            #dataset=self.dataset,
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/",
            metrics=metrics)

        classif.output(lambda x, y: (x + y) / 2)
        classif.train()
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
        
        rf = RandomForest(model_name="test_rf", model_version="1", dataset=dataset, rewrite=True)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset, rewrite=True)
        rf.train()
        ab.train()

        classif = Grid([rf, ab],
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/")

        classif.output(lambda x, y: (x + y) / 2)
        classif.train()        
        with self.original_dataset:
            data = self.original_dataset.data[:1]

        for p in classif.predict(data, raw=True, transform=False):
            self.assertEqual(len(list(p)), 2)
        dataset.destroy()
        classif.destroy()

    def test_compose_grid(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost, KNN
        from ml.clf.ensemble import Grid, EnsembleLayers

        rf = RandomForest(model_name="test_rf", model_version="1", dataset=self.original_dataset, rewrite=True)
        ab = AdaBoost(model_name="test_ab", model_version="1", dataset=self.original_dataset, rewrite=True)
        rf.train()
        ab.train()

        layer_1 = Grid([rf, ab], 
            model_name="test_grid0", 
            model_version="1",
            check_point_path="/tmp/")

        knn = KNN(model_name="test_rf", model_version="1", autoload=False, rewrite=True)
        ab2 = AdaBoost(model_name="test_ab", model_version="1", autoload=False, rewrite=True)
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
        #t0 = layer_1.scores()
        #t1 = ensemble.scores()
        #all_ = (t0 + t1)
        #all_.print_scores()
        #self.assertEqual(len(all_.measures), 2)
        #ensemble.destroy()

    def test_compose_grid_predict(self):
        from ml.clf.extended.w_sklearn import RandomForest, AdaBoost
        from ml.clf.extended.w_sklearn import LogisticRegression, KNN
        from ml.clf.ensemble import Grid, EnsembleLayers      
        from ml.processing import FitRobustScaler, Transforms

        transforms = Transforms()
        transforms.add(FitRobustScaler, type="column")
        ds0 = self.dataset.add_transforms(transforms, name="ds_test_0")

        classif_1 = Grid([(RandomForest, ds0), (KNN, self.dataset)],
            dataset=self.dataset,
            model_name="test_grid0",            
            check_point_path="/tmp/",
            model_version="1")

        classif_2 = Grid([AdaBoost, LogisticRegression],
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

        classif_1 = Grid([(RandomForest, self.dataset), (AdaBoost, self.dataset), (KNN, self.dataset)],
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

        #classif_1.destroy()
        #classif_2.destroy()
        ensemble.add(classif_1)
        ensemble.add(classif_2)
        others_models_args = {"RandomForest": [{"n_splits": None}]}
        ensemble.train([others_models_args], calc_scores=False)
        ensemble.scores().print_scores()

        ensemble = EnsembleLayers(
            model_name="test_ensemble_grid", 
            model_version="1",
            check_point_path="/tmp/")
        
        for p in ensemble.predict(self.dataset.data[:1], raw=True):
            self.assertEqual(len(list(p)), 2)

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
        from ml.ds import DataSetBuilder
        X = np.asarray([1, 0]*10)
        Y = X*1
        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", 
            dtype='int', ltype='int', rewrite=True)
        with self.dataset:
            self.dataset.build_dataset(X, Y)
        try:
            from ml.clf.extended.w_xgboost import Xgboost
        
            classif = Xgboost(dataset=self.dataset, 
                model_name="test", 
                model_version="1",
                check_point_path="/tmp/",
                dtype="int",
                ltype="int",
                rewrite=True)
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
            classif.predict(self.dataset.test_data[0:1], transform=False)
        classif.destroy()


class TestKFold(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.extended.w_keras import FCNet
        X = np.random.rand(10, 2)
        Y = (X[:,0] > .5).astype(float)
        self.dataset = DataSetBuilder(dataset_path="/tmp/", 
            dtype='float', ltype='float', rewrite=True)
        with self.dataset:
            self.dataset.build_dataset(X, Y)
        classif = FCNet(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            dtype="float",
            ltype="float32",
            rewrite=True)
        classif.train(num_steps=1, batch_size=128, n_splits=4)

    def tearDown(self):
        self.dataset.destroy()

    def test_predict(self):
        from ml.clf.extended.w_keras import FCNet
        classif = FCNet(
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        with self.dataset:
            classif.predict(self.dataset.test_data[0:1])
        classif.destroy()


if __name__ == '__main__':
    unittest.main()
