import unittest
import numpy as np
from ml.data.drivers import Memory, Zarr
from ml.utils.basic import Shape


class TestDriver(unittest.TestCase):
    def setUp(self):
        self.url = "/tmp/test.dr"
        self.array_c0 = np.arange(10)
        self.array_c1 = (np.arange(10) + 1).astype(np.dtype(float))
        self.shape = Shape({"c0": self.array_c0.shape, "c1": self.array_c1.shape})
        self.dtype = np.dtype([("c0", self.array_c0.dtype), ("c1", self.array_c1.dtype)])
        self.driver = Zarr()
        self.driver.enter(url=self.url)
        self.driver.set_schema(self.dtype)
        self.driver.set_data_shape(self.shape)
        self.driver.data["c0"] = self.array_c0
        self.driver.data["c1"] = self.array_c1
        self.driver.exit()

    def tearDown(self):
        self.driver.enter(self.url)
        self.driver.destroy("data")
        self.driver.exit()

    def test_spaces(self):
        self.driver.enter(self.url)
        spaces = self.driver.spaces()
        self.assertEqual(spaces, ["data", "metadata"])
        self.driver.exit()

    def test_dtypes(self):
        self.driver.enter(self.url)
        self.assertEqual(self.driver.dtypes, self.dtype)
        self.driver.exit()

    def test_shape(self):
        self.driver.enter(self.url)
        self.assertEqual(self.driver.data.shape, self.shape)
        self.assertEqual(self.driver.data["c0"].shape, self.shape["c0"])
        self.assertEqual(self.driver.data["c1"].shape, self.shape["c1"])
        self.driver.exit()

    def test_slice_attrb(self):
        self.driver.enter(self.url)
        self.assertEqual(self.driver.data.slice, slice(0, None))
        self.assertEqual(self.driver.data["c0"].slice, slice(0, self.shape["c0"][0]))
        self.assertEqual(self.driver.data["c1"].slice, slice(0, self.shape["c1"][0]))
        self.driver.exit()

    def test_iteration(self):
        self.driver.enter(self.url)
        for d, a in zip(self.driver.data["c0"], self.array_c0):
            self.assertEqual(d, a)
        for d, a in zip(self.driver.data["c1"], self.array_c1):
            self.assertEqual(d, a)

        for ac0, ac1, driver in zip(self.array_c0, self.array_c1, self.driver.data):
            self.assertEqual(driver["c0"].to_ndarray(), ac0)
            self.assertEqual(driver["c1"].to_ndarray(), ac1)
        self.driver.exit()

    def test_set_alias(self):
        self.driver.enter(self.url)
        data = self.driver.data
        data.set_alias("c0", "group0")
        self.assertEqual(data.dtypes, [("group0", self.array_c0.dtype), ("c1", self.array_c1.dtype)])
        self.assertEqual(data["group0"].dtypes, [("group0", self.array_c0.dtype)])

        data["group0"][9] = -1
        self.assertEqual(data["group0"][9].to_ndarray(), -1)
        self.driver.exit()

    def test_multicolum_get(self):
        self.driver.enter(self.url)
        da_group = self.driver.data[["c0", "c1"]]
        array = da_group.to_ndarray()
        self.assertEqual((array[:, 0] == self.driver.data["c0"][:]).all(), True)
        self.assertEqual((array[:, 1] == self.driver.data["c1"][:]).all(), True)
        self.driver.exit()


    def test_to_dagroup(self):
        self.driver.enter(self.url)
        stc_da = self.driver.data.to_dagroup(chunks=(3,))
        self.assertEqual((stc_da["c0"].compute() == self.array_c0).all(), True)
        self.assertEqual((stc_da["c1"].compute() == self.array_c1).all(), True)
        self.driver.exit()

    def test_add(self):
        pass
