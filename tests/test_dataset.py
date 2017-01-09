import unittest
import numpy as np
import csv

from ml.ds import DataSetBuilder, DataSetBuilderFile
from ml.processing import Preprocessing


class TestDataset(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        self.dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/", 
            #transforms_row=[('scale', None)],
            train_size=.5,
            valid_size=.2,
            validator="cross")
            #processing_class=Preprocessing)
        self.dataset.build_dataset(self.X, self.Y)

    def test_build_dataset_dim_7_1_2(self):
        dataset = DataSetBuilder(
            name="test_ds_0",
            dataset_path="/tmp/",
            validator="cross")
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.train_labels.shape, (7,))
        self.assertEqual(dataset.validation_labels.shape, (1,))
        self.assertEqual(dataset.test_labels.shape, (2,))

    def test_build_dataset_dim_5_2_3(self):
        self.assertEqual(self.dataset.train_labels.shape, (5,))
        self.assertEqual(self.dataset.validation_labels.shape, (2,))
        self.assertEqual(self.dataset.test_labels.shape, (3,))

    def test_only_labels(self):
        dataset0, label0 = self.dataset.only_labels([0])
        self.assertItemsEqual(label0, np.zeros(5))
        dataset1, label1 = self.dataset.only_labels([1])
        self.assertItemsEqual(label1, np.ones(5))

    def test_labels_info(self):
        labels_counter = self.dataset.labels_info()
        self.assertEqual(labels_counter[0], 5)
        self.assertEqual(labels_counter[1], 5)

    def test_distinct_data(self):
        self.assertEqual(self.dataset.distinct_data() > 0, True)

    def test_sparcity(self):
        self.assertEqual(self.dataset.sparcity() > .9, True)

    def test_subset(self):
        self.assertEqual(self.dataset.subset(.5).train_data.shape[0], 3)


class TestDatasetFile(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        dataset = np.c_[self.X, self.Y]
        with open('/tmp/test.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(map(str, range(10)) + ['target']) 
            for row in dataset:
                csv_writer.writerow(row)

    def test_load(self):
        dataset = DataSetBuilderFile(
            "test",
            dataset_path="/tmp/", 
            transforms_row=[('scale', None)],
            processing_class=Preprocessing,
            validator="cross")
        data, labels = dataset.from_csv('/tmp/test.csv', 'target')
        self.assertItemsEqual(self.Y, labels.astype(int))


class TestDatasetNoTransforms(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        self.dataset = DataSetBuilder(
            "test",
            dataset_path="/tmp/", 
            transforms_row=[('scale', None)],
            train_size=.5,
            valid_size=.2,
            validator="cross",
            processing_class=Preprocessing,
            transforms_apply=False)
        self.dataset.build_dataset(self.X, self.Y)

    def test_no_transforms(self):
        self.dataset.info()
        self.assertEqual(DataSetBuilder.load_dataset_raw(
            self.dataset.name, self.dataset.dataset_path)["applied_transforms"], False)


if __name__ == '__main__':
    unittest.main()
