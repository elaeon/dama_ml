import unittest
import numpy as np
from ml.data.groups import StructuredGroup
from ml.data.drivers import ZarrGroup
from ml.data.drivers import Zarr
from ml.data.drivers import Memory
from ml.utils.basic import Shape


class TestDriver(unittest.TestCase):
    def setUp(self):
        self.array_c0 = np.arange(10)
        self.array_c1 = (np.arange(10) + 1).astype(np.dtype(float))
        self.shape = Shape({"c0": self.array_c0.shape, "c1": self.array_c1.shape})
        self.dtype = np.dtype([("c0", self.array_c0.dtype), ("c1", self.array_c1.dtype)])
        self.driver = Memory()
        self.driver.enter()
        self.driver.set_schema(self.dtype)
        self.driver.set_data_shape(self.shape)
        group = ZarrGroup(self.driver)
        self.driver["data"]["c0"] = self.array_c0
        self.driver["data"]["c1"] = self.array_c1
        self.driver.exit()

    def tearDown(self):
        self.driver.enter()
        self.driver.destroy("data")
        self.driver.exit()

    def test_spaces(self):
        spaces = self.driver.spaces()
        self.assertEqual(spaces, ["data", "metadata"])

    def test_dtypes(self):
        self.assertEqual(self.driver.dtypes(), self.dtype)

    def test_shape(self):
        self.assertEqual(self.driver["data"].shape, self.shape)
        self.assertEqual(self.driver["data"]["c0"].shape, self.shape["c0"])
        self.assertEqual(self.driver["data"]["c1"].shape, self.shape["c1"])

    def test_slice(self):
        self.assertEqual(self.driver["data"].slice, slice(0, None))
        self.assertEqual(self.driver["data"]["c0"].slice, slice(0, self.shape["c0"][0]))
        self.assertEqual(self.driver["data"]["c1"].slice, slice(0, self.shape["c1"][0]))

    def test_iteration(self):
        for d, a in zip(self.driver["data"]["c0"], self.array_c0):
            self.assertEqual(d, a)
        for d, a in zip(self.driver["data"]["c1"], self.array_c1):
            self.assertEqual(d, a)

        for ac0, ac1, driver in zip(self.array_c0, self.array_c1, self.driver["data"]):
            self.assertEqual(driver["c0"].to_ndarray(), ac0)
            self.assertEqual(driver["c1"].to_ndarray(), ac1)

    def test_set_alias(self):
        self.driver["data"].set_alias("c0", "group0")
        #self.driver["data"]["group0"]
        data["group0"]

    def test_to_array(self):
        pass


    #def test_dtypes(self):
    #    columns = [("x", np.random.rand(10).astype('uint8')), ("y", np.random.rand(10))]
    #    str_array = StructArray(columns)
    #    self.assertEqual(str_array.dtypes, [('x', np.dtype('uint8')), ('y', np.dtype('float64'))])
    #    self.assertEqual(str_array.dtype, np.dtype('float64'))