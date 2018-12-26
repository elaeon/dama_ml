import unittest
import numpy as np

from ml.data.ds import Data
from ml.clf.extended.w_sklearn import RandomForest
from ml.data.etl import Pipeline
from ml.utils.numeric_functions import CV, gini_normalized
from ml.data.drivers import HDF5
from ml.measures import Measure


try:
    from ml.clf.extended.w_xgboost import Xgboost
except ImportError:
    from ml.clf.extended.w_sklearn import RandomForest as Xgboost


try:
    from ml.clf.extended.w_lgb import LightGBM
except ImportError:
    from ml.clf.extended.w_sklearn import RandomForest as LightGBM


def mulp(row):
    return row * 2


def multi_int(xm):
    try:
        return np.asarray([int(x) for x in xm])
    except TypeError:
        return xm


def to_data(cv, driver=None):
    x_train, x_validation, x_test, y_train, y_validation, y_test = cv
    x_train.rename_group("x", "train_x")
    y_train.rename_group("y", "train_y")
    x_test.rename_group("x", "test_x")
    y_test.rename_group("y", "test_y")
    x_validation.rename_group("x", "validation_x")
    y_validation.rename_group("y", "validation_y")
    stc = x_train + y_train + x_test + y_test + x_validation + y_validation
    cv_ds = Data(name="cv", driver=driver, clean=True)
    cv_ds.from_data(stc)
    return cv_ds


class TestSKL(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.rand(100, 10)
        self.Y = self.X[:, 0] > .5

    def tearDown(self):
        pass

    def test_load_meta(self):
        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data({"x": self.X, "y": self.Y})
        classif = RandomForest(
            model_name="test_model",
            check_point_path="/tmp/")
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        pipeline = Pipeline(dataset)
        a = pipeline.map(cv.apply).map(to_data, kwargs=dict(driver=HDF5()))
        b = a.map(classif.train, kwargs=dict(num_steps=1,
                  data_train_group="train_x", target_train_group='train_y',
                  data_test_group="test_x", target_test_group='test_y',
                  data_validation_group="validation_x", target_validation_group="validation_y"))
        b.compute()
        classif.save(model_version="1")
        meta = classif.load_meta()
        self.assertEqual(meta["model"]["hash"], "$sha1$b791e5ab2da782eeb47c811f1d7ec742f5ac0c1a")
        self.assertEqual(meta["train"]["model_version"], "1")
        self.assertEqual(meta["model"]["ds_basic_params"]["driver"], "ml.data.drivers.HDF5")
        classif.destroy()
        dataset.destroy()

    def test_empty_load(self):
        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data({"x": self.X, "y": self.Y})

        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        stc = cv.apply(dataset)
        ds = to_data(stc, driver=HDF5())
        classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
        classif.save(model_version="1")
        meta = classif.load_meta()
        ds_hash = meta["model"]["hash"]

        classif = RandomForest(
            model_name="test",
            check_point_path="/tmp/")
        classif.load(model_version="1")
        with classif.ds:
            self.assertEqual(classif.ds.hash, ds_hash)
        classif.destroy()
        dataset.destroy()

    def test_scores(self):
        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data({"x": self.X, "y": self.Y})

        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/")
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        stc = cv.apply(dataset)
        ds = to_data(stc, driver=HDF5())
        classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
        classif.save(model_version="1")
        scores_table = classif.scores2table()
        classif.destroy()
        dataset.destroy()
        self.assertCountEqual(list(scores_table.headers), ['', 'f1', 'auc', 'recall', 'precision',
                                                           'logloss', 'accuracy'])
        self.assertEqual(scores_table.measures[0][5] <= 1, True)

    def test_new_scores(self):
        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data({"x": self.X, "y": self.Y})

        metrics = Measure().make_metrics()
        metrics.add(gini_normalized, greater_is_better=True, output='uncertain')
        classif = RandomForest(
            model_name="test", 
            check_point_path="/tmp/",
            metrics=metrics)
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        stc = cv.apply(dataset)
        ds = to_data(stc, driver=HDF5())
        classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
        classif.save(model_version="1")
        scores_table = classif.scores2table()
        self.assertCountEqual(list(scores_table.headers), ['', 'f1', 'auc', 'recall', 'precision',
                                                           'logloss', 'gini_normalized', 'accuracy'])
        classif.destroy()
        dataset.destroy()

    def test_predict(self):
        x = np.random.rand(100)
        y = x > .5
        dataset = Data(name="test", dataset_path="/tmp", driver=HDF5(), clean=True)
        dataset.from_data({"x": x.reshape(-1, 1), "y": y})
        classif = RandomForest(
            model_name="test",
            check_point_path="/tmp/")
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with dataset:
            stc = cv.apply(dataset)
            ds = to_data(stc, driver=HDF5())
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y',
                          data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save(model_version="1")
        values = np.asarray([1, 2, .4, .1, 0, 1])
        ds = Data(name="test2", clean=True)
        ds.from_data(values)
        with ds:
            for pred in classif.predict(ds):
                self.assertCountEqual(pred, [True, True, False, False, False, True])
        dataset.destroy()
        classif.destroy()
        ds.destroy()

    def test_simple_predict(self):
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.utils.numeric_functions import gini_normalized
        from ml.measures import Measure

        dataset = Data(name="test", dataset_path="/tmp", driver=HDF5(), clean=True)
        dataset.from_data({"x": self.X, "y": self.Y})

        metrics = Measure()
        metrics.add(gini_normalized, greater_is_better=True, output='uncertain')

        rf = RandomForest(model_name="test_rf", metrics=metrics)
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with dataset:
            stc = cv.apply(dataset)
            ds = to_data(stc, driver=HDF5())
            rf.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                     data_test_group="test_x", target_test_group='test_y',
                     data_validation_group="validation_x", target_validation_group="validation_y")
            rf.save(model_version="1")

        with dataset:
            data = dataset["x"]
            predict_shape = rf.predict(data, output='uncertain', batch_size=0).shape
        self.assertEqual(predict_shape, (np.inf, 2))
        predict_shape = rf.predict(data, batch_size=10).shape
        self.assertEqual(predict_shape, (np.inf,))
        rf.destroy()
        dataset.destroy()

    def multi_output(self):
        # classif.output(lambda x, y: ((x + y) / 2).compose(multi_int))
        pass


class TestXgboost(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(100, 10)
        y = (x[:, 0] > .5).astype(int)
        self.dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        self.dataset.from_data({"x": x, "y": y})
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with self.dataset:
            stc = cv.apply(self.dataset)
            ds = to_data(stc, driver=HDF5())

        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        classif = Xgboost(
            model_name="test",
            check_point_path="/tmp/")
        classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
        classif.save(model_version="1")

    def tearDown(self):
        self.dataset.destroy()

    def test_predict(self):
        classif = Xgboost(
            model_name="test",
            check_point_path="/tmp/")
        classif.load(model_version="1")
        with self.dataset:
            predict = classif.predict(self.dataset["x"], batch_size=1)
            for pred in predict:
                self.assertEqual(pred[0], 1)
                break
        classif.destroy()


class TestLightGBM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(100, 10)
        y = (x[:, 0] > .5).astype(int)
        self.dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        self.dataset.from_data({"x": x, "y": y})
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with self.dataset:
            stc = cv.apply(self.dataset)
            ds = to_data(stc, driver=HDF5())

        classif = LightGBM(
            model_name="test",
            check_point_path="/tmp/")
        classif.set_dataset(self.dataset)
        self.params={'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
                     'objective': 'binary', 'seed': 99, "verbosity": 0, "learning_rate": 0.1,
                     'boosting_type': "gbdt", 'max_bin': 255, 'num_leaves': 25,
                     'metric': 'binary_logloss'}
        self.num_steps = 10
        self.model_version = "1"
        classif.train(ds, num_steps=self.num_steps, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=self.params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
        classif.save(model_version=self.model_version)

    def tearDown(self):
        self.dataset.destroy()

    def test_train_params(self):
        classif = LightGBM(
            model_name="test",
            check_point_path="/tmp/")
        classif.load(model_version=self.model_version)
        meta = classif.load_meta()
        self.assertCountEqual(meta["train"]["params"].keys(), self.params.keys())
        self.assertEqual(meta["train"]["model_version"], self.model_version)
        self.assertEqual(meta["train"]["num_steps"], self.num_steps)
        self.assertEqual(len(meta["train"]["score"].keys()) > 0, True)
        classif.destroy()

    def test_predict(self):
        classif = LightGBM(
            model_name="test",
            check_point_path="/tmp/")
        classif.load(model_version=self.model_version)
        with self.dataset:
            predict = classif.predict(self.dataset["x"], batch_size=1)[:1]
            for pred in predict:
                self.assertEqual(pred[0], 1)
        classif.destroy()


class TestModelVersion(unittest.TestCase):
    def test_load_add_version(self):
        x = np.random.rand(100)
        y = x > .5
        dataset = Data(name="test", dataset_path="/tmp", driver=HDF5(), clean=True)
        dataset.from_data({"x": x.reshape(-1, 1), "y": y})

        classif = RandomForest()
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with dataset:
            stc = cv.apply(dataset)
            ds = to_data(stc, driver=HDF5())
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y',
                          data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save("test", path="/tmp/", model_version="1")

        classif = RandomForest.load("test", path="/tmp/", model_version="1")
        classif.train(ds, num_steps=10, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
        classif.save("test", path="/tmp/", model_version="2")

        classif2 = RandomForest.load("test", path="/tmp/", model_version="2")
        self.assertEqual(classif2.model_version, "2")
        self.assertEqual(classif2.base_path, "/tmp/")
        self.assertEqual(classif2.num_steps, 10)
        self.assertEqual(classif2.model_name, "test")
        classif.destroy()
        classif2.destroy()
        dataset.destroy()


if __name__ == '__main__':
    unittest.main()
