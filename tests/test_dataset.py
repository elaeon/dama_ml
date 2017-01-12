import unittest
import numpy as np
import csv

from ml.ds import DataSetBuilder, DataSetBuilderFile
from ml.processing import Transforms


class TestDataset(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        self.dataset = DataSetBuilder(
            name="test_ds",
            dataset_path="/tmp/",
            train_size=.5,
            valid_size=.2,
            ltype='int',
            validator="cross",
            chunks=2)
        self.dataset.build_dataset(self.X, self.Y)

    def tearDown(self):
        self.dataset.destroy()

    def test_build_dataset_dim_7_1_2(self):
        dataset = DataSetBuilder(
            name="test_ds_0",
            dataset_path="/tmp/",
            ltype='int',
            validator="cross")
        dataset.build_dataset(self.X, self.Y)
        self.assertEqual(dataset.train_labels.shape, (7,))
        self.assertEqual(dataset.validation_labels.shape, (1,))
        self.assertEqual(dataset.test_labels.shape, (2,))
        dataset.destroy()

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
        self.assertEqual(self.dataset.sparcity() > .3, True)

    def test_copy(self):
        self.assertEqual(self.dataset.copy(.5).train_data.shape[0], 3)

    def test_apply_transforms_flag(self):
        self.dataset.apply_transforms = True
        copy = self.dataset.copy()
        transforms_to_apply = copy.transforms_to_apply
        copy.destroy()
        self.assertEqual(transforms_to_apply, False)

        self.dataset.apply_transforms = False
        copy = self.dataset.copy()
        transforms_to_apply = copy.transforms_to_apply
        copy.destroy()
        self.assertEqual(transforms_to_apply, False)

    def test_convert(self):
        dsb = self.dataset.convert("convert_test", dtype='float32', ltype='|S1')
        #apply_transforms=False, percentaje=1, applied_transforms=False):
        self.assertEqual(dsb.train_data.dtype, np.dtype('float32'))
        self.assertEqual(dsb.train_labels.dtype, np.dtype('|S1'))
        dsb.destroy()

    def test_add_transform(self):
        transforms = Transforms()
        if self.dataset.transforms_to_apply is True:
            dsb = self.dataset.add_transforms("add_transform", transforms)
            self.assertEqual(dsb.transforms_to_apply, True)
            dsb.destroy()
        
        dataset = DataSetBuilder(
            name="test_ds_0",
            dataset_path="/tmp/",
            ltype='int',
            apply_transforms=False,
            validator="cross")

        dataset.build_dataset(self.X, self.Y)
        dsb = dataset.add_transforms("add_transform", transforms)
        self.assertEqual(dsb.transforms_to_apply, False)
        dsb.destroy()
        dataset.destroy()


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
