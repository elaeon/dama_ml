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
        self.assertCountEqual(str_array[0:2]['x'], np.array([1, 2]))
        self.assertCountEqual(str_array[0:2]['y'], np.array([6, 7]))
        self.assertCountEqual(str_array[2:5]['x'], np.array([3, 4, 5]))
        self.assertCountEqual(str_array[2:5]['y'], np.array([8, 9, 10]))

    def test_index(self):
        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        self.assertEqual(str_array[0]["x"], 1)
        self.assertEqual(str_array[0]["y"], 6)

    def test_to_df(self):
        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        df = str_array.to_df()
        self.assertCountEqual(df["x"], columns[0][1])
        self.assertCountEqual(df["y"], columns[1][1])

    def test_array(self):
        array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        columns = [("x", array)]
        str_array = StructArray(columns)
        print(str_array[:])

    def test_is_multidim(self):
        array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        columns = [("x", array)]
        str_array = StructArray(columns)
        self.assertEqual(str_array.is_multidim(), True)

        columns = [("x", np.array([1, 2, 3, 4, 5])), ("y", np.array([6, 7, 8, 9, 10]))]
        str_array = StructArray(columns)
        self.assertEqual(str_array.is_multidim(), False)
