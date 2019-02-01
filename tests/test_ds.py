import unittest
import numpy as np
import pandas as pd
import json

from ml.data.ds import Data
from ml.data.it import Iterator
from ml.data.drivers import Zarr, HDF5
from ml.utils.model_selection import CV
from ml.utils.files import rm
from numcodecs import GZip
from ml.utils.basic import Login


class TestDataset(unittest.TestCase):
    def setUp(self):
        num_features = 10
        self.X = np.append(np.zeros((5, num_features)), np.ones((5, num_features)), axis=0).astype(float)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)

    def tearDown(self):
        pass

    def test_simple_write(self):
        with Data(name="test", dataset_path="/tmp/") as dataset:
            array = np.random.rand(10, 2)
            dataset.from_data(array)
            self.assertEqual(dataset.shape, (10, 2))
            self.assertEqual(dataset.groups, ("c0",))
            self.assertEqual((dataset.to_ndarray() == array).all(), True)
            dataset.destroy()

    def test_from_df_iter(self):
        with Data(name="test_ds_0", dataset_path="/tmp/") as dataset:
            x0 = np.random.rand(10).astype(int)
            x1 = np.random.rand(10).astype(float)
            x2 = np.random.rand(10).astype(object)
            df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
            dataset.from_data(df)
            self.assertEqual(list(dataset["X0"]), list(x0))
            self.assertEqual(list(dataset["X1"]), list(x1))
            self.assertEqual(list(dataset["X2"]), list(x2))
            self.assertEqual(dataset["X0"].dtype, int)
            self.assertEqual(dataset["X1"].dtype, float)
            self.assertEqual(dataset["X2"].dtype, object)

    def test_from_df(self):
        with Data(name="test_ds_0", dataset_path="/tmp/") as dataset:
            x0 = np.random.rand(10).astype(int)
            x1 = np.random.rand(10).astype(float)
            x2 = np.random.rand(10).astype(object)
            df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
            dataset.from_data(df)
            self.assertCountEqual(dataset["X0"].to_ndarray(), x0)
            self.assertCountEqual(dataset["X1"].to_ndarray(), x1)
            self.assertCountEqual(dataset["X2"].to_ndarray(), x2)
            dataset.destroy()

    def test_groups(self):
        with Data(name="test_ds", dataset_path="/tmp/") as dataset:
            dataset.from_data(self.X)
            self.assertEqual(dataset.groups, ('c0',))
            dataset.destroy()

    def test_groups_df(self):
        with Data(name="test_ds", dataset_path="/tmp/") as dataset:
            df = pd.DataFrame({"X": self.X[:, 0], "Y": self.Y})
            dataset.from_data(df)
            self.assertEqual(dataset.groups, ('X', 'Y'))
            dataset.destroy()

    def test_to_df(self):
        array_x = np.random.rand(10)
        array_y = np.random.rand(10)
        with Data(name="test0", dataset_path="/tmp") as data:
            odf = pd.DataFrame({"x": array_x, "y": array_y})
            data.from_data(odf)
            self.assertEqual((data.to_df().values == odf.values).all(), True)
            data.destroy()

        with Data(name="test0", dataset_path="/tmp") as data:
            odf = pd.DataFrame({"x": array_x})
            data.from_data(odf)
            self.assertEqual((data.to_df().values == odf.values).all(), True)
            data.destroy()

        with Data(name="test0", dataset_path="/tmp") as data:
            data.from_data(array_x)
            self.assertEqual((data.to_df().values.reshape(-1) == array_x).all(), True)
            data.destroy()

    def test_to_ndarray(self):
        with Data(name="test0", dataset_path="/tmp") as data:
            array = np.random.rand(10, 2)
            data.from_data(array)
            self.assertEqual((data.to_ndarray() == array).all(), True)
            data.destroy()

    def test_ds_build(self):
        x = np.asarray([
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [0, 0, 0, 0, 0, 0],
            [-1, 0, -1, 0, -1, 0]], dtype=np.float)
        with Data(name="test", dataset_path="/tmp") as data:
            data.from_data(x)
            self.assertEqual((data["c0"].to_ndarray()[:, 0] == x[:, 0]).all(), True)
            data.destroy()

    def test_attrs(self):
        with Data(name="test", dataset_path="/tmp") as data:
            data.author = "AGMR"
            data.description = "description text"
            self.assertEqual(data.author, "AGMR")
            self.assertEqual(data.description, "description text")
            self.assertEqual(data.timestamp, None)
            data.destroy()

    def test_to_libsvm(self):
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

        df = pd.DataFrame({"X0": self.X[:, 0], "X1": self.X[:, 1], "Y": self.Y})
        with Data(name="test_ds_1", dataset_path="/tmp/") as dataset:
            dataset.from_data(df)
            dataset.to_libsvm("Y", save_to="/tmp/test.txt")
            check("/tmp/test.txt")
            dataset.destroy()
        rm("/tmp/test.txt")

    def test_filename(self):
        with Data(name="test", dataset_path="/tmp", driver=Zarr(GZip(level=5))) as data:
            self.assertEqual(data.url, "/tmp/{}/test.zarr".format(Zarr.cls_name()))
            data.destroy()

        with Data(name="test", dataset_path="/tmp", driver=Zarr(), group_name='basic') as data:
            data.from_data([1,2,3,4,5])
            self.assertEqual(data.url, "/tmp/{}/basic/test.zarr".format(Zarr.cls_name()))
            data.destroy()

    def test_no_data(self):
        with Data(name="test", dataset_path="/tmp", driver=Zarr(GZip(level=5))) as data:
            data.author = "AGMR"
            data.description = "description text"

        with Data(name="test", dataset_path="/tmp", driver=Zarr(mode='r')) as data:
            self.assertEqual(data.author, "AGMR")
            self.assertEqual(data.description, "description text")

        with Data(name="test", dataset_path="/tmp", driver=Zarr(mode="a")) as data:
            data.from_data([1,2,3,4])
            self.assertEqual(data.compressor_params["compression_opts"], 5)
            self.assertEqual(data.author, "AGMR")
            self.assertEqual(data.description, "description text")
            data.destroy()

    def test_text_ds(self):
        x = np.asarray([(str(line)*10, "1") for line in range(100)], dtype=np.dtype("O"))
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data(x)
            self.assertEqual(data.shape, (100, 2))
            self.assertEqual(data.dtype, x.dtype)
            data.destroy()

    def test_dtypes(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data(df)
            self.assertCountEqual([e for _, (e, _) in data.dtypes.fields.items()], df.dtypes.values)
            data.destroy()

    def test_length(self):
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data(self.X)
            self.assertCountEqual(data[:3].shape, self.X[:3].shape)
            data.destroy()

    def test_from_it(self):
        seq = [1, 2, 3, 4, 4, 4, 5, 6, 3, 8, 1]
        it = Iterator(seq)
        with Data(name="test", dataset_path="/tmp") as data:
            data.from_data(it, batch_size=20)
            self.assertCountEqual(data.groups, ["c0"])
            self.assertEqual((data.to_ndarray() == seq).all(), True)
            data.destroy()

    def test_group_name(self):
        with Data(name="test0", dataset_path="/tmp", group_name="test_ds", driver=Zarr()) as data:
            self.assertEqual(data.driver.exists(data.url), True)
            data.destroy()

    def test_hash(self):
        with Data(name="test0", dataset_path="/tmp") as data:
            data.from_data(np.ones(100))
            self.assertEqual(data.hash, "$sha1$fe0e420a6aff8c6f81ef944644cc78a2521a0495")
            self.assertEqual(data.calc_hash(with_hash='md5'), "$md5$2376a2375977070dc32209a8a7bd2a99")
            data.destroy()

    def test_empty_hash(self):
        with Data(name="test0", dataset_path="/tmp") as data:
            data.from_data(np.ones(100), with_hash=None)
            self.assertEqual(data.hash, None)
            data.destroy()

    def test_getitem(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        with Data(name="test0", dataset_path="/tmp") as data:
            data.from_data(df)
            self.assertCountEqual(data["a"].to_ndarray(), df["a"].values)
            self.assertEqual((data[["a", "b"]].to_ndarray() == df[["a", "b"]].values).all(), True)
            self.assertEqual((data[0].to_ndarray(dtype=np.dtype("O")) == df.iloc[0].values).all(), True)
            self.assertEqual((data[0:1] == df.iloc[0:1].values).all(), True)
            self.assertEqual((data[3:] == df.iloc[3:].values).all(), True)
            self.assertEqual((data[:3] == df.iloc[:3].values).all(), True)
            data.destroy()

    def test_sample(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        with Data(name="test0", dataset_path="/tmp") as data:
            data.from_data(df)
            it = Iterator(data).sample(5)
            self.assertEqual(it.shape.to_tuple(), (5, 2))
            for e in it:
                self.assertEqual(e.to_ndarray().shape, (1, 2))
            data.destroy()

    def test_dataset_from_dict(self):
        x = np.asarray([1, 2, 3, 4, 5])
        y = np.asarray(['a', 'b', 'c', 'd', 'e'], dtype="object")
        with Data(name="test0", dataset_path="/tmp") as data:
            data.from_data({"x": x, "y": y})
            df = data.to_df()
            self.assertEqual((df["x"].values == x).all(), True)
            self.assertEqual((df["y"].values == y).all(), True)
            data.destroy()

    def test_from_batch(self):
        x = np.random.rand(100)
        it = Iterator(x).batchs(batch_size=10)
        with Data(name="test") as data:
            data.from_data(it)
            self.assertEqual((data.to_df().values.reshape(-1) == x).all(), True)
            data.destroy()

    def test_from_batch_array(self):
        x = np.random.rand(100)
        it = Iterator(x).batchs(batch_size=10)
        with Data(name="test") as data:
            data.from_data(it)
            self.assertEqual((data.to_ndarray() == x).all(), True)
            data.destroy()

    def test_index_dim(self):
        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        z = np.random.rand(11, 2)
        a = np.random.rand(8, 2, 1)
        columns = dict([("x", x), ("y", y), ("z", z), ("a", a)])
        with Data(name="test") as data:
            data.from_data(columns)
            self.assertEqual(data["x"].shape, x.shape)
            self.assertEqual(data["y"].shape, y.shape)
            self.assertEqual(data["z"].shape, z.shape)
            self.assertEqual(data["a"].shape, a.shape)

    def test_nested_array(self):
        values = np.asarray([[1], [2], [.4], [.1], [0], [1]])
        with Data(name="test2") as data:
            data.from_data(values)
            self.assertEqual((data.to_ndarray() == values).all(), True)
            data.destroy()

    def test_array_from_multidim_it(self):
        def _it(x_a):
            for e in x_a:
                yield (e, 1)

        with Data(name="test", dataset_path="/tmp/", driver=Zarr(mode="w")) as dataset:
            x = np.random.rand(100).reshape(-1, 1)
            dtypes = np.dtype([("x", np.dtype(float)), ("y", np.dtype(float))])
            x_p = Iterator(_it(x), dtypes=dtypes)
            dataset.from_data(x_p[:100], batch_size=0)
            dataset.destroy()

    def test_ds_it(self):
        x = np.random.rand(100, 1)
        y = np.random.rand(100, 5)
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data({"x": x, "y": y})
            it = Iterator(data).batchs(batch_size=10)
            for item in it:
                self.assertEqual((item.batch[:, 0:1] == x[item.slice]).all(), True)
                self.assertEqual((item.batch[:, 1:6] == y[item.slice]).all(), True)
            data.destroy()

    def test_index_iter(self):
        x = np.asarray([1, 2, 3, 4, 5])
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data(x)
            for i, e in enumerate(data, 1):
                self.assertEqual(e, [i])

    def test_context_index(self):
        x = np.asarray([1, 2, 3, 4, 5])
        with Data(name="test", dataset_path="/tmp/", driver=Zarr(mode="w")) as data:
            data.from_data({"x": x})
            self.assertEqual(data[0].to_ndarray(), [1])

    def test_metadata(self):
        with Data(name="test", dataset_path="/tmp/") as data:
            data.from_data([1, 2, 3, 4, 5])
            metadata = data.metadata()
            self.assertEqual(metadata["hash"], data.hash)
            self.assertEqual(metadata["size"], 0)
            data.destroy()

        with Data(name="test", dataset_path="/tmp/") as data:
            metadata = data.metadata()
            self.assertEqual(metadata["hash"], data.hash)
            data.destroy()

    def test_write_metadata(self):
        with Data(name="test", dataset_path="/tmp/", driver=Zarr(mode="w")) as data:
            data.from_data(np.random.rand(100, 10))
            metadata = data.metadata()
            with open(data.metadata_url()) as f:
                obj = json.load(f)
                self.assertEqual(obj, metadata)
            data.destroy()

    def test_many_ds(self):
        x = np.random.rand(1000, 2)
        y = np.random.rand(1000, 1)
        z = np.random.rand(1000)
        with Data(name="test_X", driver=Zarr(mode="w")) as data:
            data.from_data({"x": x, "y": y, "z": z})
            data.description = "hello world {}".format("X")
            self.assertEqual((data["x"].to_ndarray()[100] == x[100]).all(), True)
            self.assertEqual(data["y"].to_ndarray()[100], y[100])
            self.assertEqual(data["z"].to_ndarray()[100], z[100])
            data.destroy()

    def test_from_data_dim_7_1_2(self):
        with Data(name="test_ds_0", dataset_path="/tmp/") as data:
            data.from_data({"x": self.X, "y": self.Y})

            cv = CV("x", "y", train_size=.7, valid_size=.1)
            stc = cv.apply(data)
            self.assertEqual(stc["train_x"].to_ndarray().shape, (7, 10))
            self.assertEqual(stc["test_x"].to_ndarray().shape, (2, 10))
            self.assertEqual(stc["validation_x"].to_ndarray().shape, (1, 10))
            self.assertEqual(stc["train_y"].shape, (7,))
            self.assertEqual(stc["validation_y"].to_ndarray().shape, (1,))
            self.assertEqual(stc["test_y"].to_ndarray().shape, (2,))
            data.destroy()


class TestDataZarr(unittest.TestCase):
    def test_ds(self):
        with Data(name="test", dataset_path="/tmp/", driver=Zarr(mode="w")) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array)
            self.assertCountEqual(data.to_ndarray(), array)
            data.destroy()

    def test_load(self):
        with Data(name="test", dataset_path="/tmp/", driver=Zarr(mode="w")) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array)

        with Data(name="test", dataset_path="/tmp/", driver=Zarr(mode="r")) as data:
            self.assertCountEqual(data.to_ndarray(), array)
            data.destroy()

    def test_load_compression(self):
        with Data(name="test", dataset_path="/tmp/", driver=Zarr(GZip(level=6), mode="w")) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array)

        with Data(name="test", dataset_path="/tmp/", driver=Zarr(mode="r")) as data:
            self.assertEqual(data.compressor_params["compression"], "gzip")
            self.assertEqual(data.compressor_params["compression_opts"], 6)
            self.assertCountEqual(data.to_ndarray(), array)
            data.destroy()

    def test_author_description(self):
        author = "Anonymous"
        description = "Description text 00"
        with Data(name="test", dataset_path="/tmp/", driver=Zarr(GZip(level=6), mode="w")) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array)
            data.author = author
            data.description = description

        with Data(name="test", dataset_path="/tmp/", driver=Zarr(GZip(level=6), mode="r")) as data:
            self.assertEqual(data.author, author)
            self.assertEqual(data.description, description)
            data.destroy()


class TestPsqlDriver(unittest.TestCase):
    def setUp(self):
        self.login = Login(username="alejandro", resource="ml")

    def test_driver(self):
        from ml.data.db import Postgres
        x = np.random.rand(10)*100
        y = np.random.rand(10)*100
        with Data(name="test", driver=Postgres(login=self.login, mode="w")) as data:
            data.destroy()
            data.from_data({"x": x, "y": y}, batch_size=3)
            self.assertEqual((data["x"].to_ndarray(dtype=np.dtype("int8")) == x.astype("int8")).all(), True)
            self.assertEqual((data["y"].to_ndarray(dtype=np.dtype("int8")) == y.astype("int8")).all(), True)
            data.destroy()