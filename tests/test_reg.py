import unittest
import numpy as np

from ml.data.ds import Data
from ml.reg.extended.w_sklearn import RandomForestRegressor, GradientBoostingRegressor
from ml.reg.extended.w_xgboost import Xgboost
from ml.reg.extended.w_lgb import LightGBM
from ml.utils.model_selection import CV
from ml.data.drivers import HDF5


def mulp(row):
    return row * 2


class TestRegSKL(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.rand(100, 10)
        self.Y = np.random.rand(100)

    def tearDown(self):
        pass

    def test_load_meta(self):
        dataset = Data(name="reg0", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data({"x": self.X, "y": self.Y})
        reg = RandomForestRegressor()
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        stc = cv.apply(dataset)
        ds = Data(name="test_cv", dataset_path="/tmp/", driver=HDF5(), clean=True)
        ds.from_data(stc)
        model_params = dict(n_estimators=25, min_samples_split=2)
        reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
        reg.save(name="test_model", path="/tmp/", model_version="1")
        reg = RandomForestRegressor.load(model_name="test_model", path="/tmp/", model_version="1")
        self.assertEqual(reg.model_version, "1")
        self.assertEqual(reg.ds.driver.module_cls_name(), "ml.data.drivers.HDF5")
        reg.destroy()
        dataset.destroy()

    def test_empty_load(self):
        dataset = Data(name="reg0", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data({"x": self.X, "y": self.Y})
        reg = RandomForestRegressor()
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        stc = cv.apply(dataset)
        ds = Data(name="test_cv", dataset_path="/tmp/", driver=HDF5(), clean=True)
        ds.from_data(stc)
        model_params = dict(n_estimators=25, min_samples_split=2)
        reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                  data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                  data_validation_group="validation_x", target_validation_group="validation_y")
        reg.save(name="test_model", path="/tmp/", model_version="1")
        reg = RandomForestRegressor.load(model_name="test_model", path="/tmp/", model_version="1")
        self.assertEqual(reg.base_path, "/tmp/")
        reg.destroy()
        dataset.destroy()

    def test_scores(self):
        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data({"x": self.X, "y": self.Y})

        reg = RandomForestRegressor()
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        stc = cv.apply(dataset)
        ds = Data(name="test_cv", dataset_path="/tmp/", driver=HDF5(), clean=True)
        ds.from_data(stc)
        model_params = dict(n_estimators=25, min_samples_split=2)
        reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
        reg.save(name="test", path="/tmp/", model_version="1")
        scores_table = reg.scores2table()
        reg.destroy()
        dataset.destroy()
        self.assertCountEqual(list(scores_table.headers), ['', 'mse', 'msle', 'gini_normalized'])
        self.assertEqual(scores_table.measures[0][1] <= 1, True)

    def test_predict(self):
        np.random.seed(0)
        x = np.random.rand(100)
        y = np.random.rand(100)
        dataset = Data(name="test", dataset_path="/tmp", driver=HDF5(), clean=True)
        dataset.from_data({"x": x.reshape(-1, 1), "y": y})
        reg = RandomForestRegressor()
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with dataset:
            stc = cv.apply(dataset)
            ds = Data(name="test_cv", dataset_path="/tmp/", driver=HDF5(), clean=True)
            ds.from_data(stc)
            model_params = dict(n_estimators=25, min_samples_split=2)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test", path="/tmp/", model_version="1")
        values = np.asarray([1, 2, .4, .1, 0, 1])
        ds = Data(name="test2", clean=True)
        ds.from_data(values)
        with ds:
            for pred in reg.predict(ds):
                self.assertCountEqual(pred > .5, [False, False, True, False, True, False])
        dataset.destroy()
        reg.destroy()
        ds.destroy()


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
            ds = Data(name="test_cv", dataset_path="/tmp/", driver=HDF5(), clean=True)
            ds.from_data(stc)

        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        reg = Xgboost()
        reg.train(ds, num_steps=100, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
        reg.save(name="test", path="/tmp/", model_version="1")

    def tearDown(self):
        self.dataset.destroy()

    def test_predict(self):
        reg = Xgboost.load(model_name="test", path="/tmp/", model_version="1")
        with self.dataset:
            predict = reg.predict(self.dataset["x"], batch_size=1)
            for pred in predict:
                print(pred)
                self.assertEqual(pred[0] < 1, True)
                break
        reg.destroy()


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
            ds = Data(name="test_cv", dataset_path="/tmp/", driver=HDF5(), clean=True)
            ds.from_data(stc)

        reg = LightGBM()
        self.params={'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
                     'objective': 'regression', 'seed': 99, "verbosity": 0, "learning_rate": 0.1,
                     'boosting_type': "gbdt", 'max_bin': 255, 'num_leaves': 25,
                     'metric': 'mse'}
        self.num_steps = 10
        self.model_version = "1"
        reg.train(ds, num_steps=self.num_steps, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=self.params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
        reg.save(name="test", path="/tmp/", model_version=self.model_version)

    def tearDown(self):
        self.dataset.destroy()

    def test_predict(self):
        reg = LightGBM.load(model_name="test", path="/tmp/", model_version=self.model_version)
        with self.dataset:
            predict = reg.predict(self.dataset["x"], batch_size=1)[:1]
            for pred in predict:
                self.assertEqual(pred[0] < 1, True)
        reg.destroy()

    def test_feature_importance(self):
        reg = LightGBM.load(model_name="test", path="/tmp/", model_version=self.model_version)
        with self.dataset:
            self.assertEqual(reg.feature_importance()["gain"].iloc[0], 100)


class TestWrappers(unittest.TestCase):
    def train(self, reg, model_params=None):
        np.random.seed(0)
        x = np.random.rand(100)
        y = x > .5
        dataset = Data(name="test", dataset_path="/tmp", driver=HDF5(), clean=True)
        dataset.from_data({"x": x.reshape(-1, 1), "y": y})

        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with dataset:
            stc = cv.apply(dataset)
            ds = Data(name="test_cv", dataset_path="/tmp/", driver=HDF5(), clean=True)
            ds.from_data(stc)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save("test", path="/tmp/", model_version="1")
        dataset.destroy()
        return reg

    def test_gbr(self):
        reg = GradientBoostingRegressor()
        reg = self.train(reg, model_params=dict(learning_rate=0.2, random_state=3))
        with reg.ds:
            self.assertEqual(reg.ds.hash, "$sha1$fb894bc728ca9a70bad40b856bc8e37bf67f74b6")
        reg.destroy()


if __name__ == '__main__':
    unittest.main()
