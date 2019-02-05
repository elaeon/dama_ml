import unittest
import numpy as np
from ml.data.drivers import Memory, Zarr
from ml.utils.basic import Shape, Login
from ml.data.db import Postgres


class TestDriver(unittest.TestCase):
    def setUp(self):
        self.url = "/tmp/test.dr"
        self.array_c0 = np.arange(10)
        self.array_c1 = (np.arange(10) + 1).astype(np.dtype(float))
        self.shape = Shape({"c0": self.array_c0.shape, "c1": self.array_c1.shape})
        self.dtype = np.dtype([("c0", self.array_c0.dtype), ("c1", self.array_c1.dtype)])
        self.login = Login(username="alejandro", resource="ml", url=self.url)
        self.driver = Postgres(login=self.login)
        #self.driver = Zarr(login=self.login)
        self.driver.data_tag = "test"
        with self.driver:
            self.driver.set_schema(self.dtype)
            self.driver.set_data_shape(self.shape)
            if self.driver.inblock is True:
                array = np.concatenate((self.array_c0.reshape(-1, 1), self.array_c1.reshape(-1, 1)), axis=1)
                self.driver.data.writer_conn.insert(array)
            else:
                self.driver.data.writer_conn.conn["c0"] = self.array_c0
                self.driver.data.writer_conn.conn["c1"] = self.array_c1

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
            self.assertEqual(self.driver.data.shape, self.shape)
            self.assertEqual(self.driver.data["c0"].shape, self.shape["c0"])
            self.assertEqual(self.driver.data["c1"].shape, self.shape["c1"])

    def test_getitem(self):
        with self.driver:
            for i in range(10):
                self.assertEqual(self.driver.data.conn["c0"][i].compute(), self.array_c0[i])
                self.assertEqual(self.driver.data.conn["c1"][i].compute(), self.array_c1[i])

            self.assertEqual((self.driver.data.conn["c0"][4:9].compute() == self.array_c0[4:9]).all(), True)
            self.assertEqual((self.driver.data.conn["c0"][0:10].compute() == self.array_c0[0:10]).all(), True)

    def test_iteration(self):
        with self.driver:
            for d, a in zip(self.driver.data["c0"], self.array_c0):
                self.assertEqual(d, a)
            for d, a in zip(self.driver.data["c1"], self.array_c1):
                self.assertEqual(d, a)

            for ac0, ac1, driver in zip(self.array_c0, self.array_c1, self.driver.data):
                self.assertEqual(driver["c0"].to_ndarray(), ac0)
                self.assertEqual(driver["c1"].to_ndarray(), ac1)

    def test_setitem(self):
        with self.driver:
            self.driver.data["c0"][11:13] = [1, 2]

    def test_rename(self):
        with self.driver:
            data = self.driver.data
            data.rename_group("c0", "group0")
            self.assertEqual(data.dtypes, [("group0", self.array_c0.dtype), ("c1", self.array_c1.dtype)])
            self.assertEqual(data["group0"].dtypes, [("group0", self.array_c0.dtype)])

            data["group0"][9] = -1
            self.assertEqual(data["group0"][9].to_ndarray(), -1)

    def test_multicolum_get(self):
        with self.driver:
            da_group = self.driver.data[["c0", "c1"]]
            array = da_group.to_ndarray()
            self.assertEqual((array[:, 0] == self.driver.data["c0"][:]).all(), True)
            self.assertEqual((array[:, 1] == self.driver.data["c1"][:]).all(), True)

    def test_to_dagroup(self):
        with self.driver:
            stc_da = self.driver.data
            self.assertEqual((stc_da["c0"].to_ndarray() == self.array_c0).all(), True)
            self.assertEqual((stc_da["c1"].to_ndarray() == self.array_c1).all(), True)
