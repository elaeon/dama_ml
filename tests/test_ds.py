import unittest
import numpy as np
import pandas as pd
import csv

from ml.ds import DataLabelFold, DataLabel#,DataLabelSetFile
from ml.ds import Data
from ml.processing import Transforms


def linear(x, b=0):
    return x + b


def parabole(x, b=0):
    return x**2 + b


def to_int(x, col=None):
    x[:, col] = x[:, col].astype(np.int)
    return x


def label_t(x):
    return np.log1p(x)


class TestDataset(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0).astype(float)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)

    def tearDown(self):
        pass

    def test_from_data_dim_7_1_2(self):
        dataset = DataLabel(
            name="test_ds_0",
            dataset_path="/tmp/")
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            X_train, X_validation, X_test, y_train, y_validation, y_test = dataset.cv()
            self.assertEqual(y_train.shape, (7,))
            self.assertEqual(y_validation.shape, (1,))
            self.assertEqual(y_test.shape, (2,))
        dataset.destroy()

    def test_from_data_dim_5_2_3(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            X_train, X_validation, X_test, y_train, y_validation, y_test = dataset.cv(train_size=.5, valid_size=.2)
            self.assertEqual(y_train.shape, (5,))
            self.assertEqual(y_validation.shape, (2,))
            self.assertEqual(y_test.shape, (3,))
        dataset.destroy()

    def test_only_labels(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            dataset0, label0 = dataset.only_labels([0])
            self.assertItemsEqual(label0, np.zeros(5))
            dataset1, label1 = dataset.only_labels([1])
            self.assertItemsEqual(label1, np.ones(5))
        dataset.destroy()

    def test_labels_info(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            labels_counter = dataset.labels_info()
            self.assertEqual(labels_counter[0]+labels_counter[1], 10)
        dataset.destroy()

    def test_distinct_data(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            self.assertEqual(dataset.distinct_data() > 0, True)
        dataset.destroy()

    def test_sparcity(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            self.assertEqual(dataset.sparcity() > .3, True)
        dataset.destroy()

    def test_copy(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            ds = dataset.convert("test_convert", percentaje=.5, dataset_path="/tmp")

        with ds:
            self.assertEqual(ds.data.shape[0], 5)

        ds.destroy()
        dataset.destroy()

    def test_apply_transforms_flag(self):
        dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            dataset.apply_transforms = False
            copy = dataset.convert("test_2", apply_transforms=True, dataset_path="/tmp/")
        with copy:
            self.assertEqual(copy.apply_transforms, True)
        copy.destroy()
        dataset.destroy()

    def test_convert_percentaje(self):
        with DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True) as dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            dsb = dataset.convert("convert_test", dataset_path="/tmp/", percentaje=.5)
        with dsb:
            self.assertEqual(round(self.X.shape[0]/2,0), dsb.data.shape[0])
            self.assertEqual(round(self.Y.shape[0]/2,0), dsb.labels.shape[0])
        dsb.destroy()
        dataset.destroy()

    def test_convert_transforms_true(self):
        transforms = Transforms()
        transforms.add(linear, b=1)
        dataset = DataLabel(name="test_ds", dataset_path="/tmp/", clean=True)
        
        with dataset:
            dataset.transforms = transforms
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            transforms = Transforms()
            transforms.add(parabole)
            dsb = dataset.convert("convert_test", dataset_path="/tmp/",
                                transforms=transforms, apply_transforms=True)

        with dsb:
            self.assertEqual(len(dsb.transforms.transforms), 1)
            self.assertEqual(len(dsb.transforms.transforms[0].transforms), 2)

        dsb.destroy()
        dataset.destroy()

    def test_convert_data_transforms_true(self):
        transforms = Transforms()
        transforms.add(linear, b=1)
        dataset = Data(name="test_ds", dataset_path="/tmp/", clean=True)
        dataset.transforms = transforms
        with dataset:
            dataset.from_data(self.X, self.X.shape[0])
            transforms = Transforms()
            transforms.add(parabole)
            dsb = dataset.convert("convert_test", dataset_path="/tmp/",
                                transforms=transforms, apply_transforms=True)

        with dsb:
            self.assertEqual(len(dsb.transforms.transforms), 1)
            self.assertEqual(len(dsb.transforms.transforms[0].transforms), 2)

        dsb.destroy()
        dataset.destroy()

    def test_convert_transforms_false(self):
        transforms = Transforms()
        transforms.add(linear, b=1)
        dataset = DataLabel(name="test_ds", dataset_path="/tmp/", clean=True)
        dataset.transforms = transforms
        dataset.apply_transforms = False
        transforms = Transforms()
        transforms.add(parabole)
        with dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            dsb = dataset.convert("convert_test", dataset_path="/tmp/",
                                transforms=transforms, apply_transforms=False)

        with dsb:
            self.assertEqual(len(dsb.transforms.transforms), 1)
            self.assertEqual(len(dsb.transforms.transforms[0].transforms), 2)
        dsb.destroy()
        dataset.destroy()


    def test_add_transform_convert(self):
        transforms = Transforms()
        with DataLabel(name="test_ds", dataset_path="/tmp/", clean=True) as dataset:
            dataset.apply_transforms=False
            dataset.transforms.add(linear, b=1)
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            dsb = dataset.convert("add_transform", transforms=transforms, 
                                apply_transforms=True)
        with dsb:
            print(dsb.transforms.to_json(), dataset.transforms.to_json())
            self.assertEqual(dsb.transforms.to_json() == dataset.transforms.to_json(), True)
        dsb.destroy()
        dataset.destroy()

    def test_add_transform(self):
        transforms = Transforms()
        transforms.add(linear, b=1)
        transforms.add(linear, b=2)
        transforms.add(linear, b=3)
        with DataLabel(name="test_ds", dataset_path="/tmp/", clean=True) as dataset:
            dataset.apply_transforms=False
            dataset.transforms.add(linear, b=1)
            dataset.transforms.add(linear, b=2)
            dataset.transforms.add(linear, b=3)
            dataset.from_data(self.X, self.Y, self.X.shape[0])

        with DataLabel(name="test_ds", dataset_path="/tmp/") as dataset:
            self.assertEqual(dataset.transforms.to_json(), transforms.to_json())
        
        dataset.destroy()

    def test_add_transform_setter(self):
        transforms = Transforms()
        transforms.add(linear, b=1)
        transforms.add(linear, b=2)
        transforms.add(linear, b=3)
        with DataLabel(name="test_ds", dataset_path="/tmp/", clean=True) as dataset:
            dataset.apply_transforms=False
            dataset.transforms = transforms
            dataset.from_data(self.X, self.Y, self.X.shape[0])

        with DataLabel(name="test_ds", dataset_path="/tmp/") as dataset:
            self.assertEqual(dataset.transforms.to_json(), transforms.to_json())
        
        dataset.destroy()

    def test_to_df(self):
        with DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True) as dataset:
            dataset.from_data(self.X, self.Y, 10)
            df = dataset.to_df()
        self.assertEqual(list(df.columns), ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'target'])
        dataset.destroy()
        with Data(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True) as dataset:
            dataset.from_data(self.X, 10)
            df = dataset.to_df()
        self.assertEqual(list(df.columns), ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
        dataset.destroy()

    def test_ds_build(self):
        X = np.asarray([
            [1,2,3,4,5,6],
            [6,5,4,3,2,1],
            [0,0,0,0,0,0],
            [-1,0,-1,0,-1,0]], dtype=np.float)
        dl = Data(name="test", dataset_path="/tmp", clean=True)
        with dl:
            dl.from_data(X, X.shape[0])
            self.assertItemsEqual(dl.data[0], X[0])
            self.assertItemsEqual(dl.data[1], X[1])
            self.assertItemsEqual(dl.data[2], X[2])
            self.assertItemsEqual(dl.data[3], X[3])
        dl.destroy()

    def test_ds_dtype(self):
        X = np.asarray([[1,2,3,4,5]], dtype=np.int)
        Y = np.asarray(['1','2','3','4','5'], dtype=str)
        dl = DataLabel(name="test", dataset_path="/tmp", clean=True)
        with dl:
            dl.from_data(X, Y, self.X.shape[0])
            self.assertEqual(dl.dtype, X.dtype)
            self.assertEqual(dl.ltype, Y.dtype)
        dl.destroy()

    def test_get_set(self):
        from ml.processing import rgb2gray
        transforms = Transforms()
        transforms.add(rgb2gray)
        with DataLabel(name="test", dataset_path="/tmp",
            author="AGMR", clean=True,
            description="description text", compression_level=5) as dsb:
            dsb.transforms.add(rgb2gray)
            self.assertEqual(dsb.author, "AGMR")
            self.assertEqual(dsb.transforms.to_json(), transforms.to_json())
            self.assertEqual(dsb.description, "description text")
            self.assertEqual(dsb.compression_level, 5)
            self.assertEqual(dsb.dataset_class, 'ml.ds.DataLabel')
            self.assertEqual(type(dsb.timestamp), type(''))
            self.assertEqual(dsb.apply_transforms, True)
            self.assertEqual(dsb.hash_header is not None, True)

            dsb.from_data(self.X.astype('float32'), self.Y, self.X.shape[0])
            self.assertEqual(dsb.md5 is not None, True)
            dsb.destroy()

    def test_to_libsvm(self):
        from ml.utils.files import rm
        def check(path):
            with open(path, "r") as f:
                row = f.readline()
                row = row.split(" ")
                self.assertEqual(row[0] in ["0", "1", "2"], True)
                self.assertEqual(len(row), 3)
                elem1 = row[1].split(":")
                elem2 = row[2].split(":")
                self.assertEqual(int(elem1[0]), 1)
                self.assertEqual(int(elem2[0]), 2)
                self.assertEqual(2 == len(elem1) == len(elem2), True)

        X = np.random.rand(100, 2)
        Y = np.asarray([0 if .5 < sum(e) <= 1 else -1 if 0 < sum(e) < .5 else 1 for e in X])
        with DataLabel(
            name="test_ds_1",
            dataset_path="/tmp/",
            clean=True) as dataset:
            dataset.from_data(X, Y, self.X.shape[0])
            dataset.to_libsvm(name="test.txt", save_to="/tmp")
            check("/tmp/test.txt")
            dataset.destroy()
        rm("/tmp/test.txt")

    def test_no_data(self):
        from ml.processing import rgb2gray
        dsb = DataLabel(name="test", dataset_path="/tmp",
            author="AGMR", clean=True,
            description="description text", compression_level=5, mode='w')
        dsb.transforms.add(rgb2gray)
        dsb.md5 = ""
        timestamp = dsb.timestamp

        dsb2 = DataLabel(name="test", dataset_path="/tmp")
        self.assertEqual(dsb2.author, "AGMR")
        self.assertEqual(dsb2.hash_header is not None, True)
        self.assertEqual(dsb2.description, "description text")
        self.assertEqual(dsb2.timestamp, timestamp)
        self.assertEqual(dsb2.compression_level, 5)
        self.assertEqual(dsb2.dataset_class, "ml.ds.DataLabel")
        dsb.destroy()

    def test_to_data(self):
        with DataLabel(
            name="test_ds_1",
            dataset_path="/tmp/", clean=True) as dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            data = dataset.to_data()
        with data:
            self.assertEqual(data.shape, (10, 10))
        dataset.destroy()
        data.destroy()

    def test_datalabel_to_data(self):
        with DataLabel(name="test_ds_1", dataset_path="/tmp/", clean=True) as dataset:
            dataset.from_data(self.X, self.Y, self.X.shape[0])
            data = dataset.to_data()
        with data:
            self.assertEqual(data.shape, (10, 10))
        dataset.destroy()
        data.destroy()

    def test_text_ds(self):
        X = np.asarray([(str(line)*10, "1") for line in range(100)], dtype=np.dtype("O"))
        with Data(name="test", dataset_path="/tmp/", clean=True) as ds:
            ds.from_data(X)
            self.assertEqual(ds.shape, (100, 2))
            self.assertEqual(ds.dtype, X.dtype)
            ds.destroy()

    def test_fmtypes(self):
        with Data(name="test", dataset_path="/tmp/", clean=True) as data:
            data.from_data(self.X, self.X.shape[0])
            columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            data.columns = columns
            self.assertEqual(data.columns.shape[0], self.X.shape[1])
            self.assertItemsEqual(data.columns[:], columns)
            data.destroy()

    def test_fmtypes_set_columns(self):
        with Data(name="test", dataset_path="/tmp/", clean=True) as data:
            data.from_data(self.X, self.X.shape[0])
            columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            data.columns = columns
            data.columns[2] = 'X'
            self.assertEqual(data.columns[2], 'X')
            data.destroy()

    def test_fmtypes_empty(self):
        with Data(name="test", dataset_path="/tmp/", clean=True) as data:
            data.from_data(self.X, self.X.shape[0])
            self.assertItemsEqual(data.columns, [u'c0', u'c1', u'c2', u'c3', 
                u'c4', u'c5', u'c6', u'c7', u'c8', u'c9'])
            data.destroy()

    def test_rewrite_data(self):
        with Data(name="test", dataset_path="/tmp/", clean=True) as data:
            array = np.zeros((10, 2))
            data.from_data(array, array.shape[0])
            data.data[:, 1] = np.ones((10))
            self.assertItemsEqual(data.data[:, 1], np.ones((10)))
            self.assertItemsEqual(data.data[:, 0], np.zeros((10)))

        with Data(name="test", dataset_path="/tmp/", mode='r') as data:
            data.info()
            data.destroy()

    def test_cv_ds(self):
        dl = DataLabel(name="test", dataset_path="/tmp/", clean=True)
        with dl:
            dl.from_data(self.X, self.Y, self.X.shape[0])
            train_ds, validation_ds, test_ds = dl.cv_ds(train_size=.6, valid_size=.2)
        with train_ds:            
            self.assertEqual(train_ds.shape, (6, 10))
        with validation_ds:
            self.assertEqual(validation_ds.shape, (2, 10))
        with test_ds:
            self.assertEqual(test_ds.shape, (2, 10))

        dl.destroy()
        train_ds.destroy()
        validation_ds.destroy()
        test_ds.destroy()

    def test_cv_unbalanced(self):
        from ml.utils.numeric_functions import sampling_size
        X = np.random.rand(1000, 2)
        Y = np.asarray([str(e) for e in (X[:, 1] < .5)], dtype="|O")
        ds = DataLabel(name="test", dataset_path="/tmp/", clean=True)
        unbalanced = sampling_size({u'True': .2, u'False': 350}, Y)
        with ds:
            ds.from_data(X, Y, X.shape[0])
            X_train, X_validation, X_test, y_train, y_validation, y_test = ds.cv(train_size=.7, valid_size=0, unbalanced=unbalanced)
        counter = np.unique(Y, return_counts=True)
        un = np.unique(y_test, return_counts=True)
        self.assertEqual(np.unique(y_test, return_counts=True)[1][1] - 4 <= round(counter[1][1]*.2, 0), True)
        ds.destroy()

    def test_labels_transforms(self):
        transforms = Transforms()
        transforms.add(label_t)
        dl = DataLabel(name="test", dataset_path="/tmp/", clean=True)
        X = np.random.rand(10, 1)
        Y_0 = np.random.randint(1, 10, size=(10, 1))
        Y = transforms.apply(Y_0, chunks_size=0)
        with dl:
            dl.from_data(X, Y, self.X.shape[0])
            self.assertEqual(dl.labels[0], np.log1p(Y_0[0]))
        dl.destroy()

    def test_label_index(self):
        X = np.random.rand(10, 2)
        X[:, 1] = X[:, 1] > .5
        with DataLabel(name="test", dataset_path="/tmp/", clean=True) as ds:
            ds.from_data(X, "1")
            self.assertEqual(ds.shape, (10, 1))
            self.assertItemsEqual(ds.data, X[:, 0])
            self.assertItemsEqual(ds.labels, X[:, 1])
            ds.destroy()

        with DataLabel(name="test", dataset_path="/tmp/", clean=True) as ds:
            X = pd.DataFrame({"a": [0,1,2,3,4,5,6,7,8,9], "b": [0,1,1,0,1,1,0,0,0,1]})
            ds.from_data(X, "b")
            self.assertEqual(ds.shape, (10, 1))
            self.assertItemsEqual(ds.data, X["a"])
            self.assertItemsEqual(ds.labels, X["b"])
            ds.destroy()


class TestDataSetFold(unittest.TestCase):
    def setUp(self):
        NUM_FEATURES = 10
        self.X = np.append(np.zeros((5, NUM_FEATURES)), np.ones((5, NUM_FEATURES)), axis=0)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        self.dataset = DataLabel(
            name="test_ds",
            dataset_path="/tmp/",
            clean=True)
        with self.dataset:
            self.dataset.from_data(self.X, self.Y)

    def tearDown(self):
        self.dataset.destroy()

    def test_fold(self):
        n_splits = 5
        dsbf = DataLabelFold(n_splits=n_splits, dataset_path="/tmp")
        dsbf.from_data(self.dataset)
        for dsb in dsbf.get_splits():
            with dsb:
                self.assertEqual(dsb.shape[0], 8)
                self.assertEqual(dsb.shape[1], 10)
        self.assertEqual(len(dsbf.splits), n_splits)
        dsbf.destroy()


if __name__ == '__main__':
    unittest.main()
