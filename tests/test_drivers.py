import unittest
import numpy as np
import pandas as pd
import os
from dama.drivers.core import Memory, HDF5, Zarr
from dama.utils.core import Shape, Login, Chunks
from dama.drivers.postgres import Postgres
from dama.drivers.sqlite import Sqlite
from dama.drivers.csv import CSV
from dama.utils.files import check_or_create_path_dir
from dama.data.ds import Data


TMP_PATH = check_or_create_path_dir(os.path.dirname(os.path.abspath(__file__)), 'dama_data_test')

array_c0 = np.arange(10)
array_c1 = (np.arange(10) + 1).astype(np.dtype(float))
array_c2 = np.asarray([
    "2018-01-01 08:31:28",
    "2018-01-01 09:31:28",
    "2018-01-01 10:31:28",
    "2018-01-01 11:31:28",
    "2018-01-01 12:31:28",
    "2018-01-01 13:31:28",
    "2018-01-01 14:31:28",
    "2018-01-01 15:31:28",
    "2018-01-01 16:31:28",
    "2018-01-01 17:31:28"], dtype=np.dtype("datetime64[ns]"))
shape = Shape({"c0": array_c0.shape, "c1": array_c1.shape, "c2": array_c2.shape})
dtype = np.dtype([("c0", array_c0.dtype), ("c1", array_c1.dtype), ("c2", array_c2.dtype)])
chunks = Chunks({"c0": (10, ), "c1": (10, ), "c2": (10, )})


class TestDriver(unittest.TestCase):
    def setUp(self):
        self.login = Login(username="alejandro", resource="ml", table="test")
        #self.driver = Zarr(path=TMP_PATH, login=self.login)
        self.driver = Memory()
        #self.driver = Sqlite(path=TMP_PATH, login=self.login)
        #self.driver = HDF5(path=TMP_PATH, login=self.login)
        #self.driver.build_url("test")

        with self.driver:
            self.driver.set_schema(dtype)
            self.driver.set_data_shape(shape)
            if self.driver.insert_by_rows is True:
                array = np.concatenate((array_c0.reshape(-1, 1),
                                        array_c1.reshape(-1, 1),
                                        array_c2.reshape(-1, 1).astype(str)), axis=1)
                self.driver.absgroup.insert(array)
            else:
                cast = self.driver.cast
                self.driver["c0"][0:10] = cast(array_c0)
                self.driver["c1"][0:10] = cast(array_c1)
                self.driver["c2"][0:10] = cast(array_c2)

    def tearDown(self):
        with self.driver:
            self.driver.destroy()

    def test_spaces(self):
        with self.driver:
            spaces = self.driver.spaces()
            self.assertEqual(spaces, ["data", "metadata"])

    def test_dtypes(self):
        with self.driver:
            self.assertEqual(self.driver.dtypes, dtype)

    def test_shape(self):
        with self.driver:
            self.assertEqual(self.driver.shape, shape)
            self.assertEqual(self.driver["c0"].shape, shape["c0"])
            self.assertEqual(self.driver["c1"].shape, shape["c1"])

    def test_getitem(self):
        with self.driver:
            if self.driver == Sqlite or self.driver == Postgres:
                for i in range(10):
                    self.assertEqual(self.driver["c0"][i][0], array_c0[i])
                    self.assertEqual(self.driver["c1"][i][0], array_c1[i])
            else:
                for i in range(10):
                    self.assertEqual(self.driver["c0"][i], array_c0[i])
                    self.assertEqual(self.driver["c1"][i], array_c1[i])
            self.assertEqual((self.driver["c0"][4:9] == array_c0[4:9]).all(), True)
            self.assertEqual((self.driver["c0"][0:10] == array_c0[0:10]).all(), True)
            self.assertEqual((self.driver["c0"][1] == array_c0[1]).all(), True)

    def test_datetime(self):
        with self.driver:
            if isinstance(self.driver, HDF5):
                self.assertEqual(self.driver["c2"].dtype, np.dtype("int8"))
            else:
                self.assertEqual(self.driver["c2"].dtype, np.dtype("datetime64[ns]"))


class TestDaGroupDict(unittest.TestCase):

    def setUp(self):
        self.driver = Memory()

        with self.driver:
            self.driver.set_schema(dtype)
            self.driver.set_data_shape(shape)
            cast = self.driver.cast
            self.driver["c0"][0:10] = cast(array_c0)
            self.driver["c1"][0:10] = cast(array_c1)
            self.driver["c2"][0:10] = cast(array_c2)

    def test_rename(self):
        with self.driver:
            manager = self.driver.manager(chunks)
            manager.rename_group("c0", "group0")
            self.assertEqual(manager.dtypes.names[1:], dtype.names[1:])
            self.assertEqual(manager["group0"].dtypes, [("group0", array_c0.dtype)])

            self.driver["c0"][8] = -1
            self.assertEqual(manager["group0"][8].to_ndarray(), -1)

    def test_multicolum_get(self):
        with self.driver:
            manager = self.driver.manager(chunks)
            da_group = manager[["c0", "c1"]]
            array = da_group.to_ndarray()
            self.assertEqual((array[:, 0] == manager["c0"].to_ndarray()).all(), True)
            self.assertEqual((array[:, 1] == manager["c1"].to_ndarray()).all(), True)

    def test_getitem(self):
        with self.driver:
            manager = self.driver.manager(chunks)
            self.assertEqual((manager["c0"].to_ndarray() == array_c0).all(), True)
            self.assertEqual((manager["c1"].to_ndarray() == array_c1).all(), True)

    #def test_setitem(self):
    #    with self.driver:
    #        absgroup = self.driver.absgroup
    #        absgroup[11] = [1, 2., "2018-01-01 11:31:28"]
    #        self.assertEqual((absgroup["c0"][0:10] == self.array_c0[0:10]).all(), True)
    #        self.assertEqual(absgroup["c0"][10:11], 1)

    def test_iteration(self):
        with self.driver:
            manager = self.driver.manager(chunks)
            for d, a in zip(manager["c0"], array_c0):
                self.assertEqual(d.to_ndarray(), a)
            for d, a in zip(manager["c1"], array_c1):
                self.assertEqual(d.to_ndarray(), a)

            for ac0, ac1, driver in zip(array_c0, array_c1, manager):
                self.assertEqual(driver["c0"].to_ndarray(), ac0)
                self.assertEqual(driver["c1"].to_ndarray(), ac1)

    def test_datetime(self):
        with self.driver:
            manager = self.driver.manager(chunks)
            self.assertEqual(manager.dtype, np.dtype("datetime64[ns]"))

    def test_to_dtype(self):
        with self.driver:
            manager = self.driver.manager(chunks)
            self.assertEqual(manager.to_ndarray(dtype=np.dtype(float)).dtype, np.dtype(float))

    def test_store(self):
        with Data(name="test") as data, self.driver:
            manager = self.driver.manager(chunks)
            manager["c0"] = manager.getitem("c0") + 1
            data.from_data(manager)
            self.assertEqual((data.data["c0"].to_ndarray() == array_c0 + 1).all(), True)


class TestDriverCSV(unittest.TestCase):
    def setUp(self):
        self.array = np.asarray([
            ["a", "1.1", "2018-01-01 08:31:28"],
            ["b", "2.1", "2018-01-01 09:31:28"],
            ["c", "3.1", "2018-01-01 10:31:28"],
            ["d", "4.1", "2018-01-01 11:31:28"],
            ["e", "5.1", "2018-01-01 12:31:28"],
            ["g", "6.1", "2018-01-01 13:31:28"],
            ["h", "7.1", "2018-01-01 14:31:28"],
            ["i", "8.1", "2018-01-01 15:31:28"],
            ["j", "9.1", "2018-01-01 16:31:28"],
            ["k", "10.1", "2018-01-01 17:31:28"]], dtype=object)

        df = pd.DataFrame({"a": self.array[:, 0], "b": self.array[:, 1], "c": self.array[:, 2]})
        url = os.path.join(TMP_PATH, "CSV", "test.csv")
        df.to_csv(url)
        print(url)
        self.login = Login()
        self.driver = CSV(path=TMP_PATH, mode="r")
        self.driver.build_url("test")
        with self.driver:
            print(self.driver.dtypes)
            chunks = Chunks({"Unnamed: 0": 10, "a": 10, "b": 10, "c": "10"})
            manager = self.driver.manager(chunks=chunks)
            print(manager.shape)
            print(manager.to_df())
            print(manager)
            print(self.driver.spaces())
            #self.driver.set_schema(self.dtype)
            #self.driver.set_data_shape(self.shape)
        #    absgroup = self.driver.absgroup
            #print(absgroup.conn)
            #absgroup.conn = self.data_list

    def test_info(self):
        print(self.driver)

    def tearDown(self):
        #self.driver.destroy()
        pass