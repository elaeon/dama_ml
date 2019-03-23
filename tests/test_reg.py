import unittest
import numpy as np
import os
from dama.data.ds import Data
from dama.utils.files import check_or_create_path_dir


TMP_PATH = check_or_create_path_dir(os.path.dirname(os.path.abspath(__file__)), 'dama_data_test')


try:
    from dama.reg.extended.w_xgboost import Xgboost
except ImportError:
    from dama.reg.extended.w_sklearn import RandomForestRegressor as Xgboost

try:
    from dama.reg.extended.w_lgb import LightGBM
except ImportError:
    from dama.reg.extended.w_sklearn import RandomForestRegressor  as LightGBM

from dama.reg.extended.w_sklearn import RandomForestRegressor, GradientBoostingRegressor
from dama.utils.model_selection import CV
from dama.drivers.core import HDF5


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
        with Data(name="reg0") as dataset, Data(name="test_cv_skl", driver=HDF5(mode="w", path=TMP_PATH),
                                                metadata_path=TMP_PATH) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            reg = RandomForestRegressor(metadata_path=TMP_PATH)
            model_params = dict(n_estimators=25, min_samples_split=2)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test_model", path=TMP_PATH, model_version="1")
            dataset.destroy()

        with RandomForestRegressor.load(model_name="test_model", path=TMP_PATH, model_version="1",
                                        metadata_path=TMP_PATH) as reg:
            self.assertEqual(reg.model_version, "1")
            self.assertEqual(reg.ds.driver.module_cls_name(), "dama.drivers.core.HDF5")
            self.assertEqual(len(reg.metadata_train()), 7)
            reg.destroy()

    def test_empty_load(self):
        with Data(name="reg0") as dataset, Data(name="test_cv_skl", driver=HDF5(mode="w", path=TMP_PATH),
                                                metadata_path=TMP_PATH) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            reg = RandomForestRegressor(metadata_path=TMP_PATH)
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            model_params = dict(n_estimators=25, min_samples_split=2)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                  data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                  data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test_model", path=TMP_PATH, model_version="1")

        with RandomForestRegressor.load(model_name="test_model", path=TMP_PATH, model_version="1",
                                        metadata_path=TMP_PATH) as reg:
            self.assertEqual(reg.base_path, TMP_PATH)
            reg.destroy()
            dataset.destroy()

    def test_scores(self):
        with Data(name="test") as dataset, Data(name="test_cv_skl", driver=HDF5(mode="w", path=TMP_PATH),
                                                metadata_path=TMP_PATH) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            reg = RandomForestRegressor(metadata_path=TMP_PATH)
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            model_params = dict(n_estimators=25, min_samples_split=2)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test", path=TMP_PATH, model_version="1")
            scores_table = reg.scores2table()
            reg.destroy()
            dataset.destroy()
            self.assertCountEqual(list(scores_table.headers), ['', 'mse', 'msle', 'gini_normalized'])
            self.assertEqual(scores_table.measures[0][1] <= 1, True)

    def test_predict(self):
        np.random.seed(0)
        x = np.random.rand(100)
        y = np.random.rand(100)
        with Data(name="test_pred", driver=HDF5(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as dataset, \
                Data(name="test_cv_skl_3", driver=HDF5(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as ds:
            dataset.from_data({"x": x.reshape(-1, 1), "y": y})
            reg = RandomForestRegressor(metadata_path=TMP_PATH)
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            model_params = dict(n_estimators=25, min_samples_split=2)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test", path=TMP_PATH, model_version="1")
            dataset.destroy()

        values = np.asarray([1, 2, .4, .1, 0, 1])
        with Data(name="test2") as ds:
            ds.from_data(values.reshape(-1, 1))
            for pred in reg.predict(ds, batch_size=10):
                self.assertCountEqual(pred.batch > .5, [False, False, True, False, True, False])
            ds.destroy()

        with reg.ds:
            reg.destroy()

    def test_from_hash(self):
        with Data(name="reg0") as dataset, Data(name="test_from_hash", driver=HDF5(mode="w", path=TMP_PATH),
                                                metadata_path=TMP_PATH) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc, from_ds_hash=dataset.hash)
            reg = RandomForestRegressor(metadata_path=TMP_PATH)
            model_params = dict(n_estimators=25, min_samples_split=2)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test_model", path=TMP_PATH, model_version="1")
            dataset.destroy()


class TestXgboost(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(100, 10)
        y = (x[:, 0] > .5).astype(int)
        with Data(name="test") as self.dataset, \
                Data(name="test_cv", driver=HDF5(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as ds:
            self.dataset.from_data({"x": x, "y": y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(self.dataset)
            ds.from_data(stc)
            if Xgboost == RandomForestRegressor:
                params = {}
            else:
                params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
            reg = Xgboost(metadata_path=TMP_PATH)
            reg.train(ds, num_steps=100, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test", path=TMP_PATH, model_version="1")

    def tearDown(self):
        pass

    def test_predict(self):
        with Xgboost.load(model_name="test", path=TMP_PATH, model_version="1",
                          metadata_path=TMP_PATH) as reg, self.dataset:
            predict = reg.predict(self.dataset["x"], batch_size=1)
            for pred in predict.only_data():
                self.assertEqual(pred[0] <= 1, True)
                break
            reg.destroy()


class TestLightGBM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(100, 10)
        y = (x[:, 0] > .5).astype(int)
        with Data(name="test") as self.dataset, Data(name="test_cv", driver=HDF5(path=TMP_PATH, mode="w"),
                                                     metadata_path=TMP_PATH) as ds:
            self.dataset.from_data({"x": x, "y": y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(self.dataset)
            ds.from_data(stc)
            if LightGBM == RandomForestRegressor:
                self.params = {}
            else:
                self.params = {'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
                               'objective': 'regression', 'seed': 99, "verbosity": 0, "learning_rate": 0.1,
                               'boosting_type': "gbdt", 'max_bin': 255, 'num_leaves': 25,
                               'metric': 'mse'}
            reg = LightGBM(metadata_path=TMP_PATH)
            self.num_steps = 10
            self.model_version = "1"
            reg.train(ds, num_steps=self.num_steps, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=self.params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save(name="test", path=TMP_PATH, model_version=self.model_version)

    def tearDown(self):
        pass

    def test_predict(self):
        with LightGBM.load(model_name="test", path=TMP_PATH, model_version=self.model_version,
                           metadata_path=TMP_PATH) as reg:
            predict = reg.predict(self.dataset["x"], batch_size=1)[:1]
            for pred in predict.only_data():
                self.assertEqual(pred[0] <= 1, True)
            reg.destroy()

    def test_feature_importance(self):
        with LightGBM.load(model_name="test", path=TMP_PATH, model_version=self.model_version,
                            metadata_path=TMP_PATH) as reg:
            self.assertEqual(reg.feature_importance()["gain"].shape, (10,))


class TestWrappers(unittest.TestCase):
    @staticmethod
    def train(reg, model_params=None):
        np.random.seed(0)
        x = np.random.rand(100)
        y = x > .5
        with Data(name="test", driver=HDF5(path=TMP_PATH, mode="w"), metadata_path=TMP_PATH) as dataset, \
                Data(name="test_cv", driver=HDF5(path=TMP_PATH, mode="w"), metadata_path=TMP_PATH) as ds:
            dataset.from_data({"x": x.reshape(-1, 1), "y": y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            reg.save("test", path=TMP_PATH, model_version="1")
            dataset.destroy()
        return reg

    def test_gbr(self):
        reg = GradientBoostingRegressor(metadata_path=TMP_PATH)
        reg = TestWrappers.train(reg, model_params=dict(learning_rate=0.2, random_state=3))
        reg.ds.driver.mode = "r"
        with reg.ds:
            self.assertEqual(reg.ds.hash, "sha1.fb894bc728ca9a70bad40b856bc8e37bf67f74b6")
            reg.destroy()


if __name__ == '__main__':
    unittest.main()
