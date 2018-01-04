import unittest
import numpy as np

from ml.ds import DataLabel
from ml.reg.extended.w_sklearn import RandomForestRegressor
np.random.seed(0)


class TestRegSKL(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(100, 10)
        self.Y = np.random.rand(100)

    def tearDown(self):
        pass

    def test_load_meta(self):
        dl = DataLabel(name="reg0", dataset_path="/tmp/", rewrite=True)
        with dl:
            dl.build_dataset(self.X, self.Y)
        reg = RandomForestRegressor(dataset=dl, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        reg.train(num_steps=1)
        self.assertEqual(type(reg.load_meta()), type({}))
        reg.destroy()
        dl.destroy()

    def test_empty_load(self):
        dl = DataLabel(name="reg0", dataset_path="/tmp/", rewrite=True)
        with dl:
            dl.build_dataset(self.X, self.Y)
        reg = RandomForestRegressor(dataset=dl, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        reg.train(num_steps=1)

        reg = RandomForestRegressor(
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        reg.destroy()
        dl.destroy()

    def test_scores(self):
        dl = DataLabel(name="reg0", dataset_path="/tmp/", rewrite=True)
        with dl:
            dl.build_dataset(self.X, self.Y)
        reg = RandomForestRegressor(dataset=dl, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        reg.train(num_steps=1)
        scores_table = reg.scores2table()
        reg.destroy()
        dl.destroy()
        self.assertEqual(scores_table.headers, ['', 'msle'])


if __name__ == '__main__':
    unittest.main()
