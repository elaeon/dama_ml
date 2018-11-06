import unittest
import numpy as np
from ml.utils.basic import StructArray


class TestStructArray(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dtype(self):
        columns = [("x", np.random.rand(10).astype('uint8')), ("y", np.random.rand(10))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.dtype, [('x', np.dtype('uint8')), ('y', np.dtype('float64'))])

    def test_length(self):
        columns = [("x", np.random.rand(10).astype('uint8')), ("y", np.random.rand(10))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.length, 10)

    def test_slice(self):
        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        self.assertCountEqual(str_array[0:2]['x'].to_ndarray(), np.array([1, 2]))
        self.assertCountEqual(str_array[0:2]['y'].to_ndarray(), np.array([6, 7]))
        self.assertCountEqual(str_array[2:5]['x'].to_ndarray(), np.array([3, 4, 5]))
        self.assertCountEqual(str_array[2:5]['y'].to_ndarray(), np.array([8, 9, 10]))
        self.assertEqual((str_array[1:5].to_ndarray() == np.array([[2, 7], [3, 8], [4, 9], [5, 10]])).all(), True)

    def test_index(self):
        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        self.assertEqual(str_array[0]["x"].to_ndarray(), 1)
        self.assertEqual(str_array[0]["y"].to_ndarray(), 6)

    def test_to_df(self):
        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        df = str_array.to_df()
        self.assertCountEqual(df["x"], columns[0][1])
        self.assertCountEqual(df["y"], columns[1][1])

    def test_multi_dim(self):
        array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        columns = [("x", array)]
        str_array = StructArray(columns)
        self.assertEqual((str_array.to_ndarray() == array).all(), True)

    def test_is_multidim(self):
        array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        columns = [("x", array)]
        str_array = StructArray(columns)
        self.assertEqual(str_array.is_multidim(), True)

        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.is_multidim(), False)

    def test_multindex_str(self):
        array = np.empty(5, dtype=[("x", "uint8"), ("y", "uint8"), ("z", "object")])
        array["x"] = [1, 2, 3, 4, 5]
        array["y"] = [6, 7, 8, 9, 10]
        array["z"] = ["a", "b", "c", "d", "e"]
        columns = [("x", array["x"]),
                   ("y", array["y"]),
                   ("z", array["z"])]
        str_array = StructArray(columns)
        self.assertCountEqual(str_array["x"].to_ndarray(), array["x"])
        array_h = np.hstack((array["x"].reshape(-1, 1), array["y"].reshape(-1, 1)))
        self.assertEqual((str_array[["x", "y"]].to_ndarray() == array_h).all(), True)
        self.assertCountEqual(str_array["y"].to_ndarray(), array["y"])

    def test_multindex_int(self):
        array = np.empty(5, dtype=[("x", "uint8"), ("y", "uint8"), ("z", "object")])
        array["x"] = [1, 2, 3, 4, 5]
        array["y"] = [6, 7, 8, 9, 10]
        array["z"] = ["a", "b", "c", "d", "e"]
        columns = [("x", array["x"]),
                   ("y", array["y"]),
                    ("z", array["z"])]
        str_array = StructArray(columns)
        self.assertCountEqual(str_array[[0]].to_ndarray(), array["x"])
        array_h = np.hstack((array["x"].reshape(-1, 1), array["z"].reshape(-1, 1)))
        self.assertEqual((str_array[[0,2]].to_ndarray() == array_h).all(), True)