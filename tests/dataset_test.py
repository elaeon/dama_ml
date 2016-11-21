import unittest
import ml
import numpy as np

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.X = np.ones((10, 10))
        self.Y = self.X * 10

    def test_build_dataset_dim_7_1_2(self):
        dataset = ml.ds.DataSetBuilder(
            "test",
            dataset_path="/tmp/", 
            transforms=[('scale', None)],
            validator="cross")
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.train_labels.shape, (7, 10))
        self.assertEqual(dataset.valid_labels.shape, (1, 10))
        self.assertEqual(dataset.test_labels.shape, (2, 10))

    def test_build_dataset_dim_5_2_3(self):
        dataset = ml.ds.DataSetBuilder(
            "test",
            dataset_path="/tmp/", 
            transforms=[('scale', None)],
            train_size=.5,
            valid_size=.2,
            validator="cross")
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.train_labels.shape, (5, 10))
        self.assertEqual(dataset.valid_labels.shape, (2, 10))
        self.assertEqual(dataset.test_labels.shape, (3, 10))

if __name__ == '__main__':
    unittest.main()
