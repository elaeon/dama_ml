import unittest
import numpy as np
import pandas as pd
from io import StringIO
import os
from dama.data.ds import Data
from dama.data.it import Iterator
from dama.data.drivers.core import Zarr
from dama.data.drivers.sqlite import Sqlite
from dama.utils.model_selection import CV
from dama.utils.files import rm
from numcodecs import GZip
from dama.fmtypes import DEFAUL_GROUP_NAME
from dama.utils.files import check_or_create_path_dir
from dama.utils.core import Chunks
from dama.utils.core import Metadata, Login

try:
    from dama.data.drivers.postgres import Postgres
    driver = Postgres(login=Login(username="alejandro", resource="ml"))
    driver.open()
    driver.close()
except:
    from dama.data.drivers.core import Memory as Postgres


TMP_PATH = check_or_create_path_dir(os.path.dirname(os.path.abspath(__file__)), 'dama_data_test')
np.random.seed(0)


class TestDataset(unittest.TestCase):
    def setUp(self):
        num_features = 10
        self.X = np.append(np.zeros((5, num_features)), np.ones((5, num_features)), axis=0).astype(float)
        self.Y = (np.sum(self.X, axis=1) / 10).astype(int)
        self.hash = None

    def tearDown(self):
        login = Login(table="metadata")
        if self.hash is not None:
            with Metadata(Sqlite(path=TMP_PATH, login=login)) as metadata_db:
                metadata_db.remove_data(self.hash)

    def test_simple_write(self):
        with Data(name="test") as dataset:
            array = np.random.rand(10, 2)
            dataset.from_data(array, chunks=(5, 2))
            self.assertEqual(dataset.shape, (10, 2))
            self.assertEqual(dataset.groups, (DEFAUL_GROUP_NAME,))
            self.assertEqual((dataset.to_ndarray() == array).all(), True)
            dataset.destroy()

    def test_from_df_iter(self):
        with Data(name="test_ds_0") as dataset:
            x0 = np.random.rand(10).astype(int)
            x1 = np.random.rand(10).astype(float)
            x2 = np.random.rand(10).astype(object)
            df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
            dataset.from_data(df, chunks=(5, ))
            self.assertEqual([e.to_ndarray() for e in dataset["X0"]], list(x0))
            self.assertEqual([e.to_ndarray() for e in dataset["X1"]], list(x1))
            self.assertEqual([e.to_ndarray() for e in dataset["X2"]], list(x2))
            self.assertEqual(dataset["X0"].dtype, int)
            self.assertEqual(dataset["X1"].dtype, float)
            self.assertEqual(dataset["X2"].dtype, object)

    def test_from_df(self):
        with Data(name="test_ds_0") as dataset:
            x0 = np.random.rand(10).astype(int)
            x1 = np.random.rand(10).astype(float)
            x2 = np.random.rand(10).astype(object)
            df = pd.DataFrame({"X0": x0, "X1": x1, "X2": x2})
            dataset.from_data(df, chunks=(5, ))
            self.assertCountEqual(dataset["X0"].to_ndarray(), x0)
            self.assertCountEqual(dataset["X1"].to_ndarray(), x1)
            self.assertCountEqual(dataset["X2"].to_ndarray(), x2)
            dataset.destroy()

    def test_groups(self):
        with Data(name="test_ds") as dataset:
            dataset.from_data(self.X, chunks=(5, 5))
            self.assertEqual(dataset.groups, (DEFAUL_GROUP_NAME,))
            dataset.destroy()

    def test_groups_df(self):
        with Data(name="test_ds") as dataset:
            df = pd.DataFrame({"X": self.X[:, 0], "Y": self.Y})
            dataset.from_data(df, chunks=(5, ))
            self.assertEqual(dataset.groups, ('X', 'Y'))
            dataset.destroy()

    def test_to_df(self):
        array_x = np.random.rand(10)
        array_y = np.random.rand(10)
        with Data(name="test0") as data:
            odf = pd.DataFrame({"x": array_x, "y": array_y})
            data.from_data(odf, chunks=Chunks({"x": (5, ), "y": (5, )}))
            self.assertEqual((data.to_df().values == odf.values).all(), True)
            data.destroy()

        with Data(name="test0") as data:
            odf = pd.DataFrame({"x": array_x})
            data.from_data(odf, chunks=(5, ))
            self.assertEqual((data.to_df().values == odf.values).all(), True)
            data.destroy()

        with Data(name="test0") as data:
            data.from_data(array_x, chunks=(5, ))
            self.assertEqual((data.to_df().values.reshape(-1) == array_x).all(), True)
            data.destroy()

    def test_to_ndarray(self):
        with Data(name="test0") as data:
            array = np.random.rand(10, 2)
            data.from_data(array, chunks=(5, 2))
            self.assertEqual((data.to_ndarray() == array).all(), True)
            data.destroy()

    def test_ds_build(self):
        x = np.asarray([
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [0, 0, 0, 0, 0, 0],
            [-1, 0, -1, 0, -1, 0]], dtype=np.float)
        with Data(name="test") as data:
            data.from_data(x, chunks=(2, 3))
            self.assertEqual((data[DEFAUL_GROUP_NAME].to_ndarray()[:, 0] == x[:, 0]).all(), True)
            data.destroy()

    def test_attrs(self):
        with Data(name="test") as data:
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
        with Data(name="test_ds_1") as dataset:
            dataset.from_data(df, chunks=(5, ))
            dataset.to_libsvm("Y", save_to="/tmp/test.txt")
            check("/tmp/test.txt")
            dataset.destroy()
        rm("/tmp/test.txt")

    def test_filename(self):
        with Data(name="test", driver=Zarr(GZip(level=5), path=TMP_PATH), metadata_path=TMP_PATH) as data:
            self.assertEqual(data.url, os.path.join(TMP_PATH, "{}/test.zarr".format(Zarr.cls_name())))
            data.destroy()

        with Data(name="test", driver=Zarr(path=TMP_PATH), group_name='basic', metadata_path=TMP_PATH) as data:
            data.from_data([1,2,3,4,5], chunks=(2, ))
            self.hash = data.hash
            self.assertEqual(data.url, os.path.join(TMP_PATH, "{}/basic/test.zarr".format(Zarr.cls_name())))
            data.destroy()

    def test_text_ds(self):
        x = np.asarray([(str(line)*10, "1") for line in range(100)], dtype=np.dtype("O"))
        with Data(name="test") as data:
            data.from_data(x, chunks=(10, 2))
            self.assertEqual(data.shape, (100, 2))
            self.assertEqual(data.dtype, x.dtype)
            data.destroy()

    def test_dtypes(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        with Data(name="test") as data:
            data.from_data(df, chunks=(5, ))
            self.assertCountEqual([e for _, (e, _) in data.dtypes.fields.items()], df.dtypes.values)
            data.destroy()

    def test_length(self):
        with Data(name="test") as data:
            data.from_data(self.X, chunks=(5, 5))
            self.assertCountEqual(data[:3].shape, self.X[:3].shape)
            data.destroy()

    def test_from_it(self):
        seq = [1, 2, 3, 4, 4, 4, 5, 6, 3, 8, 1]
        it = Iterator(seq)
        with Data(name="test") as data:
            data.from_data(it, chunks=(20, ))
            self.assertCountEqual(data.groups, [DEFAUL_GROUP_NAME])
            self.assertEqual((data.to_ndarray() == seq).all(), True)
            data.destroy()

    def test_group_name(self):
        with Data(name="test0", group_name="test_ds", driver=Zarr(path=TMP_PATH), metadata_path=TMP_PATH) as data:
            self.assertEqual(data.driver.exists(), True)
            data.destroy()

    def test_hash(self):
        with Data(name="test0") as data:
            data.from_data(np.ones(100), chunks=(20, ))
            self.hash = data.hash
            self.assertEqual(data.hash, "sha1.fe0e420a6aff8c6f81ef944644cc78a2521a0495")
            self.assertEqual(data.calc_hash(with_hash='md5'), "md5.2376a2375977070dc32209a8a7bd2a99")
            data.destroy()

    def test_empty_hash(self):
        with Data(name="test0") as data:
            data.from_data(np.ones(100), chunks=(20, ), with_hash=None)
            self.assertEqual(data.hash, None)
            data.destroy()

    def test_getitem(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        with Data(name="test0") as data:
            data.from_data(df, chunks=(5, ))
            self.assertCountEqual(data["a"].to_ndarray(), df["a"].values)
            self.assertEqual((data[["a", "b"]].to_ndarray() == df[["a", "b"]].values).all(), True)
            self.assertEqual((data[0].to_ndarray(dtype=np.dtype("O")) == df.iloc[0].values).all(), True)
            self.assertEqual((data[0:1].to_ndarray() == df.iloc[0:1].values).all(), True)
            self.assertEqual((data[3:].to_ndarray() == df.iloc[3:].values).all(), True)
            self.assertEqual((data[:3].to_ndarray() == df.iloc[:3].values).all(), True)
            data.destroy()

    def test_sample(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        with Data(name="test0") as data:
            data.from_data(df, chunks=(5, ))
            it = Iterator(data).sample(5)
            self.assertEqual(it.shape.to_tuple(), (5, 2))
            for e in it:
                self.assertEqual(e.to_ndarray().shape, (1, 2))
            data.destroy()

    def test_dataset_from_dict(self):
        x = np.asarray([1, 2, 3, 4, 5])
        y = np.asarray(['a', 'b', 'c', 'd', 'e'], dtype="object")
        with Data(name="test0") as data:
            data.from_data({"x": x, "y": y}, chunks=(5, ))
            df = data.to_df()
            self.assertEqual((df["x"].values == x).all(), True)
            self.assertEqual((df["y"].values == y).all(), True)
            data.destroy()

    def test_from_batch(self):
        x = np.random.rand(100)
        it = Iterator(x).batchs(chunks=(20, ))
        with Data(name="test") as data:
            data.from_data(it)
            self.assertEqual((data.to_df().values.reshape(-1) == x).all(), True)
            data.destroy()

    def test_from_batch_array(self):
        x = np.random.rand(100)
        it = Iterator(x).batchs(chunks=(10, ))
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
            data.from_data(columns, chunks=Chunks([("x", (5, 1)), ("y", (5, )), ("z", (5, 1)), ("a", (5, 1, 1))]))
            self.assertEqual(data["x"].shape, x.shape)
            self.assertEqual(data["y"].shape, y.shape)
            self.assertEqual(data["z"].shape, z.shape)
            self.assertEqual(data["a"].shape, a.shape)

    def test_nested_array(self):
        values = np.asarray([[1], [2], [.4], [.1], [0], [1]])
        with Data(name="test2") as data:
            data.from_data(values, chunks=(5, 1))
            self.assertEqual((data.to_ndarray() == values).all(), True)
            data.destroy()

    def test_array_from_multidim_it(self):
        def _it(x_a):
            for e in x_a:
                yield (e, 1)

        with Data(name="test", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as dataset:
            x = np.random.rand(100).reshape(-1, 1)
            dtypes = np.dtype([("x", np.dtype(float)), ("y", np.dtype(float))])
            x_p = Iterator(_it(x), dtypes=dtypes)
            dataset.from_data(x_p[:100], chunks=Chunks({"x": (10, 1), "y": (10, )}))
            self.hash = dataset.hash
            dataset.destroy()

    def test_ds_it(self):
        x = np.random.rand(100, 1)
        y = np.random.rand(100, 5)
        with Data(name="test") as data:
            data.from_data({"x": x, "y": y}, chunks=Chunks({"x": (20, 1), "y": (20, 2)}))
            it = Iterator(data).batchs(chunks=(10, ))
            for item in it:
                value = item.batch.to_ndarray()
                self.assertEqual((value[:, 0:1] == x[item.slice]).all(), True)
                self.assertEqual((value[:, 1:6] == y[item.slice]).all(), True)
            data.destroy()

    def test_index_iter(self):
        x = np.asarray([1, 2, 3, 4, 5])
        with Data(name="test") as data:
            data.from_data(x, chunks=(5, ))
            for i, e in enumerate(data, 1):
                self.assertEqual(e.to_ndarray(), [i])

    def test_context_index(self):
        x = np.asarray([1, 2, 3, 4, 5])
        with Data(name="test", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data:
            data.from_data({"x": x}, chunks=(5, ))
            self.hash = data.hash
            self.assertEqual(data[0].to_ndarray(), [1])

    def test_metadata(self):
        with Data(name="test") as data:
            data.from_data([1, 2, 3, 4, 5], chunks=(5, ))
            metadata = data.metadata()
            self.assertEqual(metadata["hash"], data.hash)
            self.assertEqual(metadata["size"], 0)
            data.destroy()

        with data:
            metadata = data.metadata()
            self.assertEqual(metadata["hash"], data.hash)
            data.destroy()

    def test_many_ds(self):
        x = np.random.rand(1000, 2)
        y = np.random.rand(1000, 1)
        z = np.random.rand(1000)
        with Data(name="test_X", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data:
            data.from_data({"x": x, "y": y, "z": z}, chunks=Chunks({"x": (100, 2), "y": (100, 1), "z": (100, )}))
            self.hash = data.hash
            data.description = "hello world {}".format("X")
            self.assertEqual((data["x"].to_ndarray()[100] == x[100]).all(), True)
            self.assertEqual(data["y"].to_ndarray()[100], y[100])
            self.assertEqual(data["z"].to_ndarray()[100], z[100])
            data.destroy()

    def test_from_data_dim_7_1_2(self):
        with Data(name="test_ds_0") as data:
            data.from_data({"x": self.X, "y": self.Y}, chunks=Chunks({"x": (10, 10), "y": (10, )}))

            cv = CV("x", "y", train_size=.7, valid_size=.1)
            stc = cv.apply(data)
            self.assertEqual(stc["train_x"].to_ndarray().shape, (7, 10))
            self.assertEqual(stc["test_x"].to_ndarray().shape, (2, 10))
            self.assertEqual(stc["validation_x"].to_ndarray().shape, (1, 10))
            self.assertEqual(stc["train_y"].shape, (7,))
            self.assertEqual(stc["validation_y"].to_ndarray().shape, (1,))
            self.assertEqual(stc["test_y"].to_ndarray().shape, (2,))
            data.destroy()

    def test_metadata_to_json(self):
        with Data(name="test", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data:
            data.from_data(np.random.rand(100, 10), chunks=(10, 10))
            self.hash = data.hash
            with StringIO() as f:
                data.metadata_to_json(f)
                self.assertEqual(len(f.getvalue()) > 10, True)
            data.destroy()

    def test_delete_metadata_info(self):
        with Data(name="test", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data:
            data.from_data(np.random.rand(100, 11), chunks=(20, 5))
            self.hash = data.hash
            metadata_url = data.metadata_url

        driver = Sqlite(path=TMP_PATH, login=Login(table="metadata"))
        with Metadata(driver) as metadata:
            self.assertEqual(metadata.exists(self.hash), True)

        with Data(name="test", driver=Zarr(mode="r", path=TMP_PATH), metadata_path=TMP_PATH) as data:
            data.destroy()
        with Metadata(driver) as metadata:
            self.assertEqual(metadata.is_valid(self.hash), False)

    def test_concat_axis_0(self):  # length
        with Data(name="test") as dataset, Data(name="test2") as dataset2,\
            Data(name="concat", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data_c:
            array = np.random.rand(10, 2)
            dataset.from_data(array, chunks=(5, 2))
            array = np.random.rand(10, 2)
            dataset2.from_data(array, chunks=(5, 2))
            data_c.concat((dataset, dataset2), axis=0)
            self.hash = data_c.hash
            self.assertEqual((data_c.to_ndarray()[:10] == dataset.to_ndarray()).all(), True)
            self.assertEqual((data_c.to_ndarray()[10:] == dataset2.to_ndarray()).all(), True)
            dataset.destroy()
            dataset2.destroy()
            data_c.destroy()

    def test_concat_axis_0_dtypes(self):  # length
        with Data(name="test") as dataset, Data(name="test2") as dataset2,\
            Data(name="concat", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data_c:
            array0_c0 = np.random.rand(10)
            array0_c1 = np.random.rand(10)
            dataset.from_data(pd.DataFrame({"a": array0_c0, "b": array0_c1}), chunks=(5, 2))
            array1_c0 = np.random.rand(10)
            array1_c1 = np.random.rand(10)
            dataset2.from_data(pd.DataFrame({"a": array1_c0, "b": array1_c1}) , chunks=(5, 2))
            data_c.concat((dataset, dataset2), axis=0)
            self.hash = data_c.hash
            self.assertEqual((data_c.to_ndarray()[:10] == dataset.to_ndarray()).all(), True)
            self.assertEqual((data_c.to_ndarray()[10:] == dataset2.to_ndarray()).all(), True)
            dataset.destroy()
            dataset2.destroy()
            data_c.destroy()


class TestDataZarr(unittest.TestCase):
    def setUp(self):
        self.hash = None

    def tearDown(self):
        login = Login(table="metadata")
        with Metadata(Sqlite(path=TMP_PATH, login=login)) as metadata_db:
            if self.hash is not None:
                metadata_db.remove_data(self.hash)

    def test_ds(self):
        with Data(name="test", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array, chunks=(5, ))
            self.hash = data.hash
            self.assertCountEqual(data.to_ndarray(), array)
            data.destroy()

    def test_load(self):
        with Data(name="test", driver=Zarr(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array, chunks=(5, ))
            self.hash = data.hash

        with Data(name="test", driver=Zarr(mode="r", path=TMP_PATH),
                  chunks=Chunks({"g0": (5, )}), metadata_path=TMP_PATH) as data:
            self.assertCountEqual(data.to_ndarray(), array)
            data.destroy()

    def test_load_compression(self):
        with Data(name="test", driver=Zarr(GZip(level=6), mode="w", path=TMP_PATH),
                  metadata_path=TMP_PATH) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array, chunks=(5, ))
            self.hash = data.hash

        with Data(name="test", driver=Zarr(mode="r", path=TMP_PATH),
                  chunks=Chunks({"g0": (5, )}), metadata_path=TMP_PATH) as data:
            self.assertEqual(data.compressor_params["compression"], "gzip")
            self.assertEqual(data.compressor_params["compression_opts"], 6)
            self.assertCountEqual(data.to_ndarray(), array)
            data.destroy()

    def test_author_description(self):
        author = "Anonymous"
        description = "Description text 00"
        with Data(name="test", driver=Zarr(GZip(level=6), mode="w", path=TMP_PATH),
                  metadata_path=TMP_PATH) as data:
            array = [1, 2, 3, 4, 5]
            data.from_data(array, chunks=(5, ))
            self.hash = data.hash
            data.author = author
            data.description = description

        with Data(name="test", driver=Zarr(GZip(level=6), mode="r", path=TMP_PATH),
                  chunks=Chunks({"g0": (5, )}), metadata_path=TMP_PATH) as data:
            self.assertEqual(data.author, author)
            self.assertEqual(data.description, description)
            data.destroy()

    def test_no_data(self):
        with Data(name="test", driver=Zarr(GZip(level=5), mode='w', path=TMP_PATH),
                  metadata_path=TMP_PATH) as data:
            data.author = "AGMR"
            data.description = "description text"
            data.from_data([], chunks=(5, ))
            self.hash = data.hash

        with Data(name="test", driver=Zarr(mode='r', path=TMP_PATH),
                  chunks=Chunks({"g0": (5, )}), metadata_path=TMP_PATH) as data:
            self.assertEqual(data.author, "AGMR")
            self.assertEqual(data.description, "description text")

        #with Data(name="test", dataset_path=TMP_PATH, driver=Zarr(mode="a"), chunks=Chunks({"g0": (5, )})) as data:
        #    data.from_data([1, 2, 3, 4], chunks=(5, ))
        #    self.assertEqual(data.compressor_params["compression_opts"], 5)
        #    self.assertEqual(data.author, "AGMR")
        #    self.assertEqual(data.description, "description text")
        #    data.destroy()



class TestPsqlDriver(unittest.TestCase):
    def setUp(self):
        self.login = Login(username="alejandro", resource="ml")
        self.hash = None

    def tearDown(self):
        login = Login(table="metadata")
        with Metadata(Sqlite(path=TMP_PATH, login=login)) as metadata_db:
            if self.hash is not None:
                metadata_db.remove_data(self.hash)

    def test_driver(self):
        x = np.random.rand(10)*100
        y = np.random.rand(10)*100
        with Data(name="test", driver=Postgres(login=self.login, mode="w"), metadata_path=TMP_PATH) as data:
            data.destroy()
            data.from_data({"x": x, "y": y}, chunks=Chunks({"x": (10,), "y": (10, )}))
            self.hash = data.hash
            data.clean_data_cache()
            self.assertEqual((data["x"].to_ndarray(dtype=np.dtype("int8")) == x.astype("int8")).all(), True)
            self.assertEqual((data["y"].to_ndarray(dtype=np.dtype("int8")) == y.astype("int8")).all(), True)
            data.destroy()

    def test_iter(self):
        login = Login(username="alejandro", resource="ml")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ['a', 'b', 'c', 'd', 'e']})
        with Data(name="test0", driver=Postgres(login=login), metadata_path=TMP_PATH) as data:
            data.destroy()
            data.from_data(df, chunks=(5, ))
            self.hash = data.hash
            it = Iterator(data)
            self.assertEqual(it.shape.to_tuple(), (5, 2))
            for i, e in enumerate(it):
                self.assertEqual((e.to_ndarray()[0] == df.iloc[i].values).all(), True)
            data.destroy()

    def test_iter_uni(self):
        login = Login(username="alejandro", resource="ml")
        array = [1., 2., 3., 4., 5.]
        with Data(name="test0", driver=Postgres(login=login), metadata_path=TMP_PATH) as data:
            data.destroy()
            data.from_data(array, chunks=(5, ))
            self.hash = data.hash
            it = Iterator(data)
            self.assertEqual(it.shape.to_tuple(), (5,))
            for i, e in enumerate(it):
                if len(e.to_ndarray().shape) == 1:
                    self.assertEqual((e.to_ndarray()[0] == array[i]).all(), True)
                else:
                    self.assertEqual((e.to_ndarray() == array[i]).all(), True)
            data.destroy()