import unittest
import numpy as np

from ml.data.ds import Data
from ml.reg.extended.w_sklearn import RandomForestRegressor
from ml.utils.numeric_functions import CV
from ml.data.drivers import HDF5


def mulp(row):
    return row * 2


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
        ds = to_data(stc, driver=HDF5())
        reg.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
        reg.save(name="test", path="/tmp/", model_version="1")
        reg = RandomForestRegressor.load(model_name="test_model", path="/tmp/", model_version="1")
        self.assertEqual(reg.model_version, "1")
        self.assertEqual(reg.ds.driver.module_cls_name(), "ml.data.drivers.HDF5")
        reg.destroy()
        dataset.destroy()

    def test_empty_load(self):
        dl = DataLabel(name="reg0", dataset_path="/tmp/", clean=True)
        with dl:
            dl.from_data(self.X, self.Y)
        reg = RandomForestRegressor(
            model_name="test",
            check_point_path="/tmp/")
        reg.set_dataset(dl)
        reg.train(num_steps=1)
        reg.save(model_version="1")

        reg = RandomForestRegressor(
            model_name="test", 
            check_point_path="/tmp/")
        reg.load(model_version="1")
        self.assertEqual(reg.original_dataset_path, "/tmp/")
        reg.destroy()
        dl.destroy()

    def test_scores(self):
        dl = DataLabel(name="reg0", dataset_path="/tmp/", clean=True)
        with dl:
            dl.from_data(self.X, self.Y)
        reg = RandomForestRegressor(
            model_name="test", 
            check_point_path="/tmp/")
        reg.set_dataset(dl)
        reg.train(num_steps=1)
        reg.save(model_version="1")
        scores_table = reg.scores2table()
        reg.destroy()
        dl.destroy()
        self.assertEqual(list(scores_table.headers), ['', 'msle'])

    def test_predict(self):
        from ml.processing import Transforms
        t = Transforms()
        t.add(mulp)
        X = np.random.rand(100, 1)
        Y = np.random.rand(100)
        dataset = DataLabel(name="test_0", dataset_path="/tmp/", 
            clean=True)
        dataset.transforms = t
        with dataset:
            dataset.from_data(X, Y)
        reg = RandomForestRegressor(
            model_name="test", 
            check_point_path="/tmp/")
        reg.set_dataset(dataset)
        reg.train()
        reg.save(model_version="1")
        values = np.asarray([[1], [2], [.4], [.1], [0], [1]])
        self.assertEqual(reg.predict(values).to_memory(6).shape[0], 6)
        self.assertEqual(reg.predict(values, chunks_size=0).to_memory(6).shape[0], 6)
        dataset.destroy()
        reg.destroy()

    def test_add_version(self):
        X = np.random.rand(100, 1)
        Y = np.random.rand(100)
        dataset = DataLabel(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data(X, Y)
        reg = RandomForestRegressor(
            model_name="test", 
            check_point_path="/tmp/")
        reg.set_dataset(dataset)
        reg.train()
        reg.save(model_version="1")
        reg.save(model_version="2")
        reg.save(model_version="3")
        metadata = reg.load_meta()
        self.assertCountEqual(metadata["model"]["versions"], ["1", "2", "3"])
        dataset.destroy()
        reg.destroy()


if __name__ == '__main__':
    unittest.main()
