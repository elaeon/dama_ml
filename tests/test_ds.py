import unittest
import numpy as np
import pandas as pd

from ml.data.ds import DataLabelFold
from ml.data.ds import Data, DataLabel
from ml.processing import Transforms
from ml.random import sampling_size
from ml.data.ds import Memory


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

    def test_from_unique_dtype(self):
        dataset = Data(name="test_ds_0", dataset_path="/tmp/", clean=True)
        X = np.random.rand(10, 2).astype(int)
        dataset.from_data(X, X.shape[0])
        with dataset:
            self.assertEqual(dataset[:].shape, (10, 2))
        dataset.destroy()

    def test_from_dtypes(self):
        dataset = Data(name="test_ds_0", dataset_path="/tmp/")
        X0 = np.random.rand(10).astype(int)
        X1 = np.random.rand(10).astype(float)
        X2 = np.random.rand(10).astype(object)
        df = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2})
        dataset.from_data(df)
        with dataset:
            self.assertCountEqual(dataset["X0"][:], X0)
            self.assertCountEqual(dataset["X1"][:], X1)
            self.assertCountEqual(dataset["X2"][:], map(str, X2))
            self.assertEqual(dataset["X0"].dtype, int)
            self.assertEqual(dataset["X1"].dtype, float)
            self.assertEqual(dataset["X2"].dtype, object)
            self.assertCountEqual(dataset[:]["X0"], X0)
        dataset.destroy()

    def test_from_data_dim_7_1_2(self):
        dataset = Data(name="test_ds_0", dataset_path="/tmp/")
        dataset.from_data(self.X, self.X.shape[0])

        #with dataset:
            #X_train, X_validation, X_test, y_train, y_validation, y_test = dataset.cv()
            #self.assertEqual(y_train.shape, (7,))
            #self.assertEqual(y_validation.shape, (1,))
            #self.assertEqual(y_test.shape, (2,))
        dataset.destroy()

    def test_from_data_dim_5_2_3(self):
        dataset = Data(name="test_ds", dataset_path="/tmp/", clean=True)
        #dataset.from_data(self.X, self.X.shape[0])
        #with dataset:
        #    X_train, X_validation, X_test, y_train, y_validation, y_test = dataset.cv(train_size=.5, valid_size=.2)
        #    self.assertEqual(y_train.shape, (5,))
        #    self.assertEqual(y_validation.shape, (2,))
        #    self.assertEqual(y_test.shape, (3,))
        dataset.destroy()

    def test_only_column(self):
        dataset = Data(name="test_ds", dataset_path="/tmp/", clean=True)
        XY = np.hstack((self.X, self.Y.reshape(-1, 1)))
        dataset.from_data(self.X)
        with dataset:
            self.assertCountEqual(dataset["c0"], XY[:, 0])
        dataset.destroy()

    def test_to_df(self):
        dataset = Data(name="test_ds", dataset_path="/tmp/", clean=True)
        dataset.from_data(self.X)
        with dataset:
            df = dataset.to_df()
            self.assertEqual(list(df.columns), ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
        dataset.destroy()

    def test_to_df_columns(self):
        dataset = Data(name="test_ds", dataset_path="/tmp/", clean=True)
        df = pd.DataFrame({"X": self.X[:, 0], "Y": self.Y})
        dataset.from_data(df)
        with dataset:
            df = dataset.to_df()
            self.assertEqual(list(df.columns), ['X', 'Y'])
        dataset.destroy()

    def test_ds_build(self):
        X = np.asarray([
            [1,2,3,4,5,6],
            [6,5,4,3,2,1],
            [0,0,0,0,0,0],
            [-1,0,-1,0,-1,0]], dtype=np.float)
        dl = Data(name="test", dataset_path="/tmp", clean=True)
        dl.from_data(X, X.shape[0])
        with dl:
            self.assertCountEqual(dl[0]["c0"], X[0])
            self.assertCountEqual(dl[1]["c0"], X[1])
            self.assertCountEqual(dl[2]["c0"], X[2])
            self.assertCountEqual(dl[3]["c0"], X[3])
        dl.destroy()


    def test_attrs(self):
        dsb = Data(name="test", dataset_path="/tmp", author="AGMR", clean=True,
            description="description text", compression_level=5)
        with dsb:
            self.assertEqual(dsb.author, "AGMR")
            self.assertEqual(dsb.description, "description text")
            self.assertEqual(dsb.zip_params["compression_opts"], 5)
            self.assertEqual(type(dsb.timestamp), type(''))
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
        df = pd.DataFrame({"X0": self.X[:, 0], "X1": self.X[:, 1], "Y": self.Y})
        dataset = Data(name="test_ds_1", dataset_path="/tmp/", clean=True)
        dataset.from_data(df)
        with dataset:
            dataset.to_libsvm("Y", save_to="/tmp/test.txt")
        check("/tmp/test.txt")
        dataset.destroy()
        rm("/tmp/test.txt")

    def test_no_data(self):
        dsb = Data(name="test", dataset_path="/tmp",
            author="AGMR", clean=True,
            description="description text", compression_level=5, mode='a')
        with dsb:
            timestamp = dsb.timestamp

        with Data(name="test", dataset_path="/tmp") as dsb2:
            self.assertEqual(dsb2.author, "AGMR")
            self.assertEqual(dsb2.description, "description text")
            self.assertEqual(dsb2.timestamp, timestamp)
            self.assertEqual(dsb2.zip_params["compression_opts"], 5)
        dsb.destroy()

    def test_text_ds(self):
        X = np.asarray([(str(line)*10, "1") for line in range(100)], dtype=np.dtype("O"))
        ds = Data(name="test", dataset_path="/tmp/", clean=True)
        ds.from_data(X)
        with ds:
            self.assertEqual(ds.shape, (100, 2))
            self.assertEqual(ds.dtype[0][1], X.dtype)
            ds.destroy()

    def test_dtypes(self):
        data = Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(self.X)
        dtypes = [("c"+str(i), np.dtype("float64")) for i in range(10)]
        with data:
            self.assertCountEqual([dtype for _, dtype in data.dtypes], [dtype for _, dtype in dtypes])
        data.destroy()

    def test_columns_rename(self):
        data =  Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(self.X)
        columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        data.columns = columns
        with data:
            self.assertCountEqual(data.columns, columns)
        data.destroy()

    def test_columns(self):
        data = Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(self.X)
        with data:
            self.assertCountEqual(data.columns, [u'c0', u'c1', u'c2', u'c3', 
                u'c4', u'c5', u'c6', u'c7', u'c8', u'c9'])
        data.destroy()

    def test_length(self):
        data = Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(self.X, 3)
        with data:
            self.assertCountEqual(data[:].shape, self.X[:3].shape)

    def test_cv_ds(self):
        dl = Data(name="test", dataset_path="/tmp/", clean=True)
        XY = np.hstack((self.X, self.Y.reshape(-1, 1)))
        dl.from_data(XY)
        with dl:
            train_ds, validation_ds, test_ds = dl.cv_ds(train_size=.6, valid_size=.2)
        #with train_ds:
        #    self.assertEqual(train_ds.shape, (6, 10))
        #with validation_ds:
        #    self.assertEqual(validation_ds.shape, (2, 10))
        #with test_ds:
        #    self.assertEqual(test_ds.shape, (2, 10))

        dl.destroy()
        train_ds.destroy()
        validation_ds.destroy()
        test_ds.destroy()

    def test_cv_unbalanced(self):
        X = np.random.rand(1000, 2)
        Y = np.asarray([str(e) for e in (X[:, 1] < .5)], dtype="|O")
        ds = Data(name="test", dataset_path="/tmp/", clean=True)
        unbalanced = sampling_size({u'True': .2, u'False': 350}, Y)
        ds.from_data(X, Y, X.shape[0])
        #with ds:
        #    X_train, X_validation, X_test, y_train, y_validation, y_test = ds.cv(train_size=.7, valid_size=0, unbalanced=unbalanced)
        #counter = np.unique(Y, return_counts=True)
        #un = np.unique(y_test, return_counts=True)
        #self.assertEqual(np.unique(y_test, return_counts=True)[1][1] - 4 <= round(counter[1][1]*.2, 0), True)
        ds.destroy()

    def test_labels_transforms(self):
        transforms = Transforms()
        transforms.add(label_t)
        dl = DataLabel(name="test", dataset_path="/tmp/", clean=True)
        X = np.random.rand(10, 1)
        Y_0 = np.random.randint(1, 10, size=(10, 1))
        Y = transforms.apply(Y_0, chunks_size=0)
        dl.from_data(X, Y, self.X.shape[0])
        with dl:
            self.assertEqual(dl.labels[0], np.log1p(Y_0[0]))
        dl.destroy()

    def test_label_index(self):
        X = np.random.rand(10, 2)
        X[:, 1] = X[:, 1] > .5
        ds = DataLabel(name="test", dataset_path="/tmp/", clean=True)
        ds.from_data(X, "1")
        with ds:
            self.assertEqual(ds.shape, (10, 1))
            self.assertCountEqual(ds.data, X[:, 0])
            self.assertCountEqual(ds.labels, X[:, 1])
        ds.destroy()

        ds = DataLabel(name="test", dataset_path="/tmp/", clean=True)
        X = pd.DataFrame({"a": [0,1,2,3,4,5,6,7,8,9], "b": [0,1,1,0,1,1,0,0,0,1]})
        ds.from_data(X, "b")
        with ds:            
            self.assertEqual(ds.shape, (10, 1))
            self.assertCountEqual(ds.data, X["a"])
            self.assertCountEqual(ds.labels, X["b"])
        ds.destroy()
    
    def test_from_it(self):
        from ml.data.it import Iterator
        l = [1,2,3,4,4,4,5,6,3,8,1]
        it = Iterator(l)
        it.set_length(10)
        data = Data(name="test", dataset_path="/tmp", clean=True)
        data.from_data(it, chunks_size=20)
        with data:
            self.assertCountEqual(data.columns[:], ["c0"])
        data.destroy()

    def test_concat(self):
        data0 = Data(name="test0", dataset_path="/tmp", clean=True)
        data1 = Data(name="test1", dataset_path="/tmp", clean=True)
        data2 = Data(name="test2", dataset_path="/tmp", clean=True)
        data0.from_data(np.random.rand(10, 2))
        data1.from_data(np.random.rand(10, 2))
        data2.from_data(np.random.rand(10, 2))

        dataC = Data.concat([data0, data1, data2], chunksize=10, name="concat")
        data0.destroy()
        data1.destroy()
        data2.destroy()

        with dataC:
            self.assertEqual(dataC.name, "concat")
            self.assertEqual(dataC.shape, (30, 2))
        dataC.destroy()

    def test_reader(self):
        data0 = Data(name="test0", dataset_path="/tmp", clean=True)
        data0.from_data(np.random.rand(10, 2))
        with data0:
            self.assertEqual(data0.reader(chunksize=5, df=True).to_memory().shape, (10, 2))
        data0.destroy()
    
    def test_memory_ds(self):
        data0 = Data(name="test0", dataset_path="/tmp", clean=True, driver='core')
        data0.from_data(np.random.rand(10, 2))
        with data0:
            self.assertEqual(data0.shape, (10, 2))
        data0.destroy()

    def test_group_name(self):
        data = Data(name="test0", dataset_path="/tmp", clean=True, group_name="test_ds")
        self.assertEqual(data.exists(), True)
        data.destroy()

    def test_hash(self):
        data = Data(name="test0", dataset_path="/tmp", clean=True)
        data.from_data(np.ones(100))
        with data:
            self.assertEqual(data.calc_hash(), "$sha1$fe0e420a6aff8c6f81ef944644cc78a2521a0495")
            self.assertEqual(data.calc_hash(hash_fn='md5'), "$md5$2376a2375977070dc32209a8a7bd2a99")


class TestMemoryDs(unittest.TestCase):
    def test_memory_ds(self):
        m = Memory()
        m.require_group("data")
        m["/data/data"] = "y"
        self.assertEqual(m["data"]["data"], "y")
        self.assertEqual(m["/data/data"], "y")
        m["/data/label"] = "z"
        self.assertEqual(m["data"]["label"], "z")
        self.assertEqual(m["/data/label"], "z")
        m.require_group("fmtypes")
        m["fmtypes"].require_dataset("name", (10, 1))
        self.assertEqual(m["fmtypes"]["name"].shape, (10, 1))
        self.assertEqual(m["/fmtypes/name"].shape, (10, 1))

    def test_memory_add(self):
        m = Memory()
        m.require_group("data")
        m["/data"] = "y"
        m.require_group("data")
        self.assertEqual(m["data"], "y")
        


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
