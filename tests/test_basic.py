import unittest
import numpy as np
from ml.utils.basic import StructArray, Shape


class TestStructArray(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dtypes(self):
        columns = [("x", np.random.rand(10).astype('uint8')), ("y", np.random.rand(10))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.dtypes, [('x', np.dtype('uint8')), ('y', np.dtype('float64'))])
        self.assertEqual(str_array.dtype, np.dtype('float64'))

    def test_length(self):
        columns = [("x", np.random.rand(10).astype('uint8')), ("y", np.random.rand(10))]
        str_array = StructArray(columns)
        self.assertEqual(len(str_array), 10)

    def test_shape_multidim(self):
        columns = [("x", np.random.rand(10).astype('uint8')), ("y", np.random.rand(10, 2))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.shape["x"], (10,))
        self.assertEqual(str_array.shape["y"], (10, 2))

    def test_shape_item(self):
        columns = [("x", np.random.rand(10)), ("y", np.random.rand(10)), ("z", np.random.rand(10))]
        str_array = StructArray(columns)
        self.assertEqual(str_array["x"].shape, (10,))
        self.assertEqual(str_array[["x", "y"]].shape, (10, 2))
        self.assertEqual(str_array[["x", "y", "z"]].shape, (10, 3))

        columns = [("x", np.random.rand(10, 2)), ("y", np.random.rand(10, 2))]
        str_array = StructArray(columns)
        self.assertEqual(str_array["x"].shape, (10, 2))
        self.assertEqual(str_array[["x", "y"]].shape, (10, 2, 2))

        self.assertEqual(str_array["x"].to_ndarray().shape, (10, 2))
        self.assertEqual(str_array[["x", "y"]].to_ndarray().shape, (10, 2, 2))

    def test_shape_null(self):
        columns = []
        str_array = StructArray(columns)
        self.assertEqual(str_array.shape, (0,))

    def test_shape(self):
        columns = [("x", np.random.rand(10, 3, 1).astype('uint8')), ("y", np.random.rand(10, 3, 1))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.shape, (10, 2, 3, 1))

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
        self.assertEqual((str_array[3].to_ndarray() == [4, 9]).all(), True)

    def test_index_multi(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([6, 7, 8, 9, 10])
        columns = [("x", x), ("y", y)]
        str_array = StructArray(columns)
        xrds = str_array.to_xrds()
        dict_ = xrds.to_dict()
        self.assertEqual((dict_["data_vars"]["x"]["data"] == x).all(), True)
        self.assertEqual((dict_["data_vars"]["y"]["data"] == y).all(), True)

    def test_to_df(self):
        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        df = str_array.to_df()
        self.assertCountEqual(df["x"], columns[0][1])
        self.assertCountEqual(df["y"], columns[1][1])

    def test_multidim(self):
        array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        columns = [("x", array)]
        str_array = StructArray(columns)
        self.assertEqual((str_array.to_ndarray().reshape(2, 5) == array).all(), True)

    def test_is_multidim(self):
        array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        columns = [("x", array)]
        str_array = StructArray(columns)
        self.assertEqual(str_array.is_multidim(), False)

        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9]))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.is_multidim(), True)

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
        columns = [("x", array["x"]), ("y", array["y"]), ("z", array["z"])]
        str_array = StructArray(columns)
        self.assertCountEqual(str_array[[0]].to_ndarray(), array["x"])
        array_h = np.hstack((array["x"].reshape(-1, 1), array["z"].reshape(-1, 1)))
        self.assertEqual((str_array[[0, 2]].to_ndarray() == array_h).all(), True)

    def test_index_iter(self):
        columns = [("x", np.asarray([1, 2, 3, 4, 5]))]
        str_array = StructArray(columns)
        for e in str_array:
            print(e.shape.to_tuple())
            self.assertEqual(e.shape, (1,))

    def test_add(self):
        x = np.random.rand(10, 2)
        y = np.random.rand(10)
        x_train = StructArray([("x", x)])
        y_train = StructArray([("y", y)])
        xy_train = x_train + y_train
        self.assertEqual((xy_train["x"].to_ndarray() == x).all(), True)
        self.assertEqual((xy_train["y"].to_ndarray() == y).all(), True)


class TestShape(unittest.TestCase):

    def test_get_item_shape(self):
        shape = Shape({"x": (10,), "y": (5,), "z": (10, 8)})
        self.assertEqual(shape[0], 10)
        self.assertEqual(shape["z"], (10, 8))

    def test_to_tuple(self):
        shape = Shape({"x": (10, 2), "y": (10,)})
        self.assertEqual(shape.to_tuple(), (10, 2, 2))
        shape = Shape({"x": (10, 2)})
        self.assertEqual(shape.to_tuple(), (10, 2))
        shape = Shape({"x": (10, 2), "y": (3, 2), "z": (11, 1, 1)})
        self.assertEqual(shape.to_tuple(), (11, 3, 2, 1))

    def test_length(self):
        shape = Shape({"x": (10, 2), "y": (3, 2), "z": (11, 1, 1)})
        self.assertEqual(shape.max_length, 11)
        shape = Shape({})
        self.assertEqual(shape.max_length, 0)