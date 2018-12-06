import unittest
import numpy as np
import pandas as pd

from ml.data.ds import Data
from ml.data.it import Iterator
from ml.utils.basic import StructArray
from ml.data.drivers import Zarr, HDF5
from numcodecs import GZip


class TestDataset(unittest.TestCase):
    def setUp(self):
        num_features = 10
        self.X = np.append(np.zeros((5, num_features)), np.ones((5, num_features)), axis=0).astype(float)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)

    def tearDown(self):
        pass

    def test_from_unique_dtype(self):
        dataset = Data(name="test_ds_0", dataset_path="/tmp/", clean=True)
        array = np.random.rand(10, 2)
        dataset.from_data(array)
        with dataset:
            self.assertEqual(dataset.shape, (10, 2))
        dataset.destroy()

    def test_to_list(self):
        dataset = Data(name="test_ds_0", dataset_path="/tmp/", clean=True)
        X0 = np.random.rand(10).astype(int)
        X1 = np.random.rand(10).astype(float)
        X2 = np.random.rand(10).astype(object)
        df = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2})
        dataset.from_data(df)
        with dataset:
            self.assertEqual(len(list(dataset["X0"])), 10)

    def test_from_dtypes(self):
        dataset = Data(name="test_ds_0", dataset_path="/tmp/", clean=True)
        X0 = np.random.rand(10).astype(int)
        X1 = np.random.rand(10).astype(float)
        X2 = np.random.rand(10).astype(object)
        df = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2})
        dataset.from_data(df)
        with dataset:
            self.assertCountEqual(dataset["X0"].to_ndarray(), X0)
            self.assertCountEqual(dataset["X1"].to_ndarray(), X1)
            self.assertCountEqual(dataset["X2"].to_ndarray(), X2)
            self.assertEqual(dataset["X0"].dtype, int)
            self.assertEqual(dataset["X1"].dtype, float)
            self.assertEqual(dataset["X2"].dtype, object)
            self.assertCountEqual(dataset[:]["X0"].to_ndarray(), X0)
        dataset.destroy()

    def test_from_data_dim_7_1_2(self):
        dataset = Data(name="test_ds_0", dataset_path="/tmp/", clean=True)
        dataset.from_data(self.X)

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
            self.assertEqual((dataset["c0"][:, 0].to_ndarray() == XY[:, 0]).all(), True)
        dataset.destroy()

    def test_groups(self):
        dataset = Data(name="test_ds", dataset_path="/tmp/", clean=True)
        dataset.from_data(self.X)
        with dataset:
            self.assertEqual(dataset.groups, ['c0'])
        dataset.destroy()

    def test_groups_df(self):
        dataset = Data(name="test_ds", dataset_path="/tmp/", clean=True)
        df = pd.DataFrame({"X": self.X[:, 0], "Y": self.Y})
        dataset.from_data(df)
        with dataset:
            df = dataset.to_df()
            self.assertEqual(list(df.columns), ['X', 'Y'])
        dataset.destroy()

    def test_to_df(self):
        data0 = Data(name="test0", dataset_path="/tmp", clean=True)
        array = np.random.rand(10)
        data0.from_data(array)
        with data0:
            self.assertEqual((data0.to_df().values.reshape(-1) == array).all(), True)
        data0.destroy()

    def test_to_ndarray(self):
        data0 = Data(name="test0", dataset_path="/tmp", clean=True)
        array = np.random.rand(10, 2)
        data0.from_data(array)
        with data0:
            self.assertEqual((data0.to_ndarray() == array).all(), True)
        data0.destroy()

    def test_to_structured(self):
        data = Data(name="test")
        array = np.array([[1, 'x1'], [2, 'x2'], [3, 'x3'], [4, 'x4'],
                  [5, 'x5'], [6, 'x6'], [7, 'x7'], [8, 'x8'],
                  [9, 'x9'], [10, 'x10']])
        data.from_data(array)
        with data:
            self.assertEqual((data.to_structured()["c0"] == array).all(), True)
        data.destroy()

    def test_ds_build(self):
        X = np.asarray([
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [0, 0, 0, 0, 0, 0],
            [-1, 0, -1, 0, -1, 0]], dtype=np.float)
        dl = Data(name="test", dataset_path="/tmp", clean=True)
        dl.from_data(X)
        with dl:
            self.assertEqual((dl["c0"].to_ndarray()[:, 0] == X[:, 0]).all(), True)
        dl.destroy()

    def test_attrs(self):
        dsb = Data(name="test", dataset_path="/tmp", author="AGMR", clean=True,
            description="description text")
        with dsb:
            self.assertEqual(dsb.author, "AGMR")
            self.assertEqual(dsb.description, "description text")
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

    def test_filename(self):
        dsb = Data(name="test", dataset_path="/tmp", driver=HDF5(GZip(level=5)), clean=True)
        self.assertEqual(dsb.url, "/tmp/test.h5")
        dsb.destroy()
        dsb = Data(name="test", dataset_path="/tmp", driver=Zarr(GZip(level=5)), clean=True)
        self.assertEqual(dsb.url, "/tmp/test.zarr")
        dsb.destroy()

    def test_no_data(self):
        dsb = Data(name="test", dataset_path="/tmp",
            author="AGMR", clean=True,
            description="description text", driver=HDF5(GZip(level=5)))
        with dsb:
            timestamp = dsb.timestamp

        with Data(name="test", dataset_path="/tmp", driver=HDF5()) as dsb2:
            self.assertEqual(dsb2.author, "AGMR")
            self.assertEqual(dsb2.description, "description text")
            self.assertEqual(dsb2.timestamp, timestamp)
            self.assertEqual(dsb2.compressor_params["compression_opts"], 5)
        dsb.destroy()

    def test_text_ds(self):
        X = np.asarray([(str(line)*10, "1") for line in range(100)], dtype=np.dtype("O"))
        ds = Data(name="test", dataset_path="/tmp/", clean=True)
        ds.from_data(X)
        with ds:
            self.assertEqual(ds.shape, (100, 2))
            self.assertEqual(ds.dtype, X.dtype)
            ds.destroy()

    def test_dtypes(self):
        data = Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(self.X)
        dtypes = [("c"+str(i), np.dtype("float64")) for i in range(1)]
        with data:
            self.assertCountEqual([dtype for _, dtype in data.dtypes], [dtype for _, dtype in dtypes])
        data.destroy()

    def test_dtypes_2(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        data = Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(df)
        with data:
            self.assertCountEqual([e for _, e in data.dtypes], df.dtypes.values)
        data.destroy()

    def test_labels_rename(self):
        data =  Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(self.X)
        columns = ['a']
        data.groups = columns
        with data:
            self.assertCountEqual(data.groups, columns)
        data.destroy()

    def test_groups_rename_2(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        data =  Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(df)
        columns = ['x0', 'x1']
        data.groups = columns
        with data:
            self.assertCountEqual(data.groups, columns)
        data.destroy()

    def test_length(self):
        data = Data(name="test", dataset_path="/tmp/", clean=True)
        data.from_data(self.X)
        with data:
            self.assertCountEqual(data[:3].shape, self.X[:3].shape)

    def test_from_struct(self):
        X = StructArray([("x", np.random.rand(10, 2)), ("y", np.random.rand(10))])
        Y = StructArray([("y", np.random.rand(10))])
        data = Data(name="test", dataset_path="/tmp", clean=True)
        data.from_data({"x": X, "y": Y})

    def test_from_it(self):
        seq = [1, 2, 3, 4, 4, 4, 5, 6, 3, 8, 1]
        it = Iterator(seq)
        data = Data(name="test", dataset_path="/tmp", clean=True)
        data.from_data(it, batch_size=20)
        with data:
            self.assertCountEqual(data.groups, ["c0"])
        data.destroy()

    def test_group_name(self):
        data = Data(name="test0", dataset_path="/tmp", clean=True, group_name="test_ds", driver=HDF5())
        self.assertEqual(data.exists(), True)
        data.destroy()

    def test_hash(self):
        data = Data(name="test0", dataset_path="/tmp", clean=True)
        data.from_data(np.ones(100))
        with data:
            self.assertEqual(data.hash, "$sha1$fe0e420a6aff8c6f81ef944644cc78a2521a0495")
            self.assertEqual(data.calc_hash(with_hash='md5'), "$md5$2376a2375977070dc32209a8a7bd2a99")

    def test_empty_hash(self):
        data = Data(name="test0", dataset_path="/tmp", clean=True)
        data.from_data(np.ones(100), with_hash=None)
        with data:
            self.assertEqual(data.hash, None)

    def test_getitem(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        data = Data(name="test0", dataset_path="/tmp", clean=True)
        data.from_data(df)
        with data:
            self.assertCountEqual(data["a"].to_ndarray(), df["a"].values)
            self.assertEqual((data[["a", "b"]].to_ndarray() == df[["a", "b"]].values).all(), True)
            self.assertEqual((data[0].to_ndarray() == df.iloc[0].values).all(), True)
            self.assertEqual((data[0:1].to_ndarray() == df.iloc[0:1].values).all(), True)
            self.assertEqual((data[3:].to_ndarray() == df.iloc[3:].values).all(), True)
            self.assertEqual((data[:3].to_ndarray() == df.iloc[:3].values).all(), True)
        data.destroy()

    def test_sample(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        data = Data(name="test0", dataset_path="/tmp", clean=True)
        data.from_data(df)
        with data:
            it = Iterator(data).sample(5)
            for e in it:
                self.assertEqual(e.to_ndarray().shape, (1, 2))

    def test_dataset_from_dict(self):
        x = np.asarray([1, 2, 3, 4, 5])
        y =  np.asarray(['a', 'b', 'c', 'd', 'e'], dtype="object")
        data = Data(name="test0", dataset_path="/tmp", clean=True)
        data.from_data({"x": x, "y": y})
        with data:
            df = data.to_df()
            self.assertEqual((df["x"].values == x).all(), True)
            self.assertEqual((df["y"].values == y).all(), True)


class TestDataZarr(unittest.TestCase):
    def test_ds(self):
        data = Data(name="test", dataset_path="/tmp/", driver=Zarr())
        array = [1, 2, 3, 4, 5]
        data.from_data(array)
        with data:
            self.assertCountEqual(data.to_ndarray(), array)
        data.destroy()

    def test_load(self):
        data = Data(name="test", dataset_path="/tmp/", driver=Zarr(), clean=True)
        array = [1, 2, 3, 4, 5]
        data.from_data(array)

        data = Data(name="test", dataset_path="/tmp/", driver=Zarr(), clean=False)
        with data:
            self.assertCountEqual(data.to_ndarray(), array)

    def test_compressor(self):
        data = Data(name="test", dataset_path="/tmp/", driver=Zarr(GZip(level=6)), clean=True)
        array = [1, 2, 3, 4, 5]
        data.from_data(array)
        with data:
            self.assertEqual(data.compressor_params["compression"], "gzip")
            self.assertEqual(data.compressor_params["compression_opts"], 6)



if __name__ == '__main__':
    unittest.main()
