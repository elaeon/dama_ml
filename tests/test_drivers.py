import unittest
import numpy as np
import pandas as pd
import os
from dama.drivers.core import Memory, HDF5
from dama.utils.core import Shape, Login, Chunks
from dama.drivers.postgres import Postgres
from dama.drivers.sqlite import Sqlite
from dama.utils.files import check_or_create_path_dir


TMP_PATH = check_or_create_path_dir(os.path.dirname(os.path.abspath(__file__)), 'dama_data_test')


class TestDriverAbsBaseGroup(unittest.TestCase):
    def setUp(self):
        self.array_c0 = np.arange(10)
        self.array_c1 = (np.arange(10) + 1).astype(np.dtype(float))
        self.array_c2 = np.asarray([
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
        self.shape = Shape({"c0": self.array_c0.shape, "c1": self.array_c1.shape, "c2": self.array_c2.shape})
        self.dtype = np.dtype([("c0", self.array_c0.dtype), ("c1", self.array_c1.dtype), ("c2", self.array_c2.dtype)])

        self.login = Login(username="alejandro", resource="ml", table="test")
        # self.driver = Zarr(login=self.login)
        self.driver = Memory(path=TMP_PATH)
        self.driver.build_url("test_{}".format(10))
        # self.driver = HDF5(login=self.login)

        with self.driver:
            self.driver.set_schema(self.dtype)
            self.driver.set_data_shape(self.shape)
            absgroup = self.driver.absgroup
            if absgroup.inblock is True:
                array = np.concatenate((self.array_c0.reshape(-1, 1),
                                        self.array_c1.reshape(-1, 1),
                                        self.array_c2.reshape(-1, 1).astype(str)), axis=1)
                absgroup.insert(array)
            else:
                cast = absgroup.cast
                absgroup.conn["c0"][0:10] = cast(self.array_c0)
                absgroup.conn["c1"][0:10] = cast(self.array_c1)
                absgroup.conn["c2"][0:10] = cast(self.array_c2)

    def test_spaces(self):
        with self.driver:
            spaces = self.driver.spaces()
            self.assertEqual(spaces, ["data", "metadata"])

    def test_dtypes(self):
        with self.driver:
            self.assertEqual(self.driver.dtypes, self.dtype)

    def test_shape(self):
        with self.driver:
            absgroup = self.driver.absgroup
            self.assertEqual(absgroup.shape, self.shape)
            self.assertEqual(absgroup.conn["c0"].shape, self.shape["c0"])
            self.assertEqual(absgroup.conn["c1"].shape, self.shape["c1"])

    def test_getitem(self):
        with self.driver:
            absgroup = self.driver.absgroup
            if self.driver == Sqlite or self.driver == Postgres:
                for i in range(10):
                    self.assertEqual(absgroup.conn["c0"][i][0], self.array_c0[i])
                    self.assertEqual(absgroup.conn["c1"][i][0], self.array_c1[i])
            else:
                for i in range(10):
                    self.assertEqual(absgroup.conn["c0"][i], self.array_c0[i])
                    self.assertEqual(absgroup.conn["c1"][i], self.array_c1[i])
            self.assertEqual((absgroup.conn["c0"][4:9] == self.array_c0[4:9]).all(), True)
            self.assertEqual((absgroup.conn["c0"][0:10] == self.array_c0[0:10]).all(), True)
            self.assertEqual((absgroup.conn["c0"][1] == self.array_c0[1]).all(), True)

    def test_datetime(self):
        with self.driver:
            absgroup = self.driver.absgroup
            if isinstance(self.driver, HDF5):
                self.assertEqual(absgroup.conn["c2"].dtype, np.dtype("int8"))
            else:
                self.assertEqual(absgroup.conn["c2"].dtype, np.dtype("datetime64[ns]"))


class TestDriverAbsGroup(unittest.TestCase):

    def setUp(self):
        self.array_c0 = np.arange(10)
        self.array_c1 = (np.arange(10) + 1).astype(np.dtype(float))
        self.array_c2 = np.asarray([
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
        self.shape = Shape({"c0": self.array_c0.shape, "c1": self.array_c1.shape, "c2": self.array_c2.shape})
        self.dtype = np.dtype([("c0", self.array_c0.dtype), ("c1", self.array_c1.dtype), ("c2", self.array_c2.dtype)])

        self.login = Login(username="alejandro", resource="ml", table="test", host="/var/run/postgresql/", port=5432)
        #self.driver = Postgres(login=self.login, path=TMP_PATH)
        self.driver = Sqlite(login=self.login, path=TMP_PATH, mode="a")
        self.driver.build_url("test_db")
        #self.driver = Zarr(login=self.login)
        #self.driver = Memory()
        #self.driver = HDF5(login=self.login)

        with self.driver:
            self.driver.set_schema(self.dtype)
            self.driver.set_data_shape(self.shape)
            absgroup = self.driver.absgroup
            if absgroup.inblock is True:
                array = np.concatenate((self.array_c0.reshape(-1, 1),
                                        self.array_c1.reshape(-1, 1),
                                        self.array_c2.reshape(-1, 1).astype(str)), axis=1)
                absgroup.insert(array)
            else:
                cast = absgroup.cast
                absgroup.conn["c0"][0:10] = cast(self.array_c0)
                absgroup.conn["c1"][0:10] = cast(self.array_c1)
                absgroup.conn["c2"][0:10] = cast(self.array_c2)

    def tearDown(self):
        with self.driver:
            self.driver.destroy()

    def test_spaces(self):
        with self.driver:
            spaces = self.driver.spaces()
            self.assertEqual(spaces, ["data", "metadata"])

    def test_dtypes(self):
        with self.driver:
           self.assertEqual(self.driver.dtypes, self.dtype)

    def test_shape(self):
        with self.driver:
            absgroup = self.driver.absgroup
            self.assertEqual(absgroup.shape, self.shape)
            self.assertEqual(absgroup["c0"].shape, self.shape["c0"])
            self.assertEqual(absgroup["c1"].shape, self.shape["c1"])

    def test_getitem(self):
        with self.driver:
            absgroup = self.driver.absgroup
            for i in range(10):
                self.assertEqual(absgroup["c0"][i], self.array_c0[i])
                self.assertEqual(absgroup["c1"][i], self.array_c1[i])
            self.assertEqual((absgroup["c0"][4:9] == self.array_c0[4:9]).all(), True)
            self.assertEqual((absgroup["c0"][0:10] == self.array_c0[0:10]).all(), True)
            self.assertEqual((absgroup["c0"][1] == self.array_c0[1]).all(), True)

    def test_setitem(self):
        with self.driver:
            absgroup = self.driver.absgroup
            absgroup[11] = [1, 2., "2018-01-01 11:31:28"]
            self.assertEqual((absgroup["c0"][0:10] == self.array_c0[0:10]).all(), True)
            self.assertEqual(absgroup["c0"][10:11], 1)

    def test_datetime(self):
        with self.driver:
            absgroup = self.driver.absgroup
            if isinstance(self.driver, HDF5):
                self.assertEqual(absgroup["c2"].dtype, np.dtype("int8"))
            else:
                self.assertEqual(absgroup["c2"].dtype, np.dtype("datetime64[ns]"))


class TestDaGroup(unittest.TestCase):

    def setUp(self):
        self.array_c0 = np.arange(10)
        self.array_c1 = (np.arange(10) + 1).astype(np.dtype(float))
        self.array_c2 = np.asarray([
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
        self.shape = Shape({"c0": self.array_c0.shape, "c1": self.array_c1.shape, "c2": self.array_c2.shape})
        self.dtype = np.dtype([("c0", self.array_c0.dtype), ("c1", self.array_c1.dtype), ("c2", self.array_c2.dtype)])
        self.driver = Memory()
        self.chunks = Chunks({"c0": (10, ), "c1": (10, ), "c2": (10, )})

        with self.driver:
            self.driver.set_schema(self.dtype)
            self.driver.set_data_shape(self.shape)
            absgroup = self.driver.absgroup(self.chunks)
            cast = absgroup.cast
            absgroup["c0"][0:10] = cast(self.array_c0)
            absgroup["c1"][0:10] = cast(self.array_c1)
            absgroup["c2"][0:10] = cast(self.array_c2)

    def test_rename(self):
        with self.driver:
            absgroup = self.driver.absgroup(self.chunks)
            absgroup.write_to_group = absgroup
            absgroup.manager.rename_group("c0", "group0")
            self.assertEqual(absgroup.dtypes.names[1:], self.dtype.names[1:])
            self.assertEqual(absgroup.manager["group0"].dtypes, [("group0", self.array_c0.dtype)])

            absgroup["c0"][8] = -1
            self.assertEqual(absgroup.manager["group0"][8].to_ndarray(), -1)

    def test_multicolum_get(self):
        with self.driver:
            da_group = self.driver.data(self.chunks)[["c0", "c1"]]
            array = da_group.to_ndarray()
            self.assertEqual((array[:, 0] == self.driver.data(self.chunks)["c0"].to_ndarray()).all(), True)
            self.assertEqual((array[:, 1] == self.driver.data(self.chunks)["c1"].to_ndarray()).all(), True)

    def test_get(self):
        with self.driver:
            stc_da = self.driver.data(self.chunks)
            self.assertEqual((stc_da["c0"].to_ndarray() == self.array_c0).all(), True)
            self.assertEqual((stc_da["c1"].to_ndarray() == self.array_c1).all(), True)

    def test_iteration(self):
        with self.driver:
            stc_da = self.driver.data(self.chunks)
            for d, a in zip(stc_da["c0"], self.array_c0):
                self.assertEqual(d.to_ndarray(), a)
            for d, a in zip(stc_da["c1"], self.array_c1):
                self.assertEqual(d.to_ndarray(), a)

            for ac0, ac1, driver in zip(self.array_c0, self.array_c1, stc_da):
                self.assertEqual(driver["c0"].to_ndarray(), ac0)
                self.assertEqual(driver["c1"].to_ndarray(), ac1)

    def test_datetime(self):
        with self.driver:
            stc_da = self.driver.data(self.chunks)
            self.assertEqual(stc_da.dtype, np.dtype("datetime64[ns]"))

    def test_to_dtype(self):
        with self.driver:
            stc_da = self.driver.data(self.chunks)
            self.assertEqual(stc_da.to_ndarray(dtype=np.dtype(float)).dtype, np.dtype(float))


class TestDriverCSV(unittest.TestCase):
    def setUp(self):
        from dama.drivers.csv import CSV
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

        #self.shape = Shape({"c0": self.array_c0.shape})
        #self.dtype = np.dtype([("c0", self.array_c0.dtype), ("c1", self.array_c1.dtype), ("c2", self.array_c2.dtype)])

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
            absgroup = self.driver.absgroup(chunks=chunks)
            print(absgroup.shape)
            print(absgroup.to_df())
            print(absgroup)
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