import unittest
import numpy as np

from ml.ds import DataLabel
from ml.reg.extended.w_sklearn import RandomForestRegressor
np.random.seed(0)


def mulp(row):
    return row * 2


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
        reg = RandomForestRegressor( 
            model_name="test", 
            check_point_path="/tmp/")
        reg.set_dataset(dl)
        reg.train(num_steps=1)
        reg.save(model_version="1")
        meta = reg.load_meta()
        self.assertEqual(meta["model"]["original_dataset_path"], "/tmp/")
        self.assertEqual(meta["train"]["model_version"], "1")
        reg.destroy()
        dl.destroy()

    def test_empty_load(self):
        dl = DataLabel(name="reg0", dataset_path="/tmp/", rewrite=True)
        with dl:
            dl.build_dataset(self.X, self.Y)
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
        dl = DataLabel(name="reg0", dataset_path="/tmp/", rewrite=True)
        with dl:
            dl.build_dataset(self.X, self.Y)
        reg = RandomForestRegressor(
            model_name="test", 
            check_point_path="/tmp/")
        reg.set_dataset(dl)
        reg.train(num_steps=1)
        reg.save(model_version="1")
        scores_table = reg.scores2table()
        reg.destroy()
        dl.destroy()
        self.assertEqual(scores_table.headers, ['', 'msle'])

    def test_predict(self):
        from ml.processing import Transforms
        t = Transforms()
        t.add(mulp)
        X = np.random.rand(100, 1)
        Y = np.random.rand(100)
        dataset = DataLabel(name="test_0", dataset_path="/tmp/", 
            rewrite=True, transforms=t, apply_transforms=True)
        with dataset:
            dataset.build_dataset(X, Y)
        reg = RandomForestRegressor(
            model_name="test", 
            check_point_path="/tmp/")
        reg.set_dataset(dataset)
        reg.train()
        reg.save(model_version="1")
        values = np.asarray([[1], [2], [.4], [.1], [0], [1]])
        self.assertEqual(len(reg.predict(values).to_memory()), 6)
        self.assertEqual(len(reg.predict(np.asarray(values), chunks_size=0).to_memory()), 6)
        dataset.destroy()
        reg.destroy()


if __name__ == '__main__':
    unittest.main()
