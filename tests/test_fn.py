import unittest
import numpy as np


class TestNumericFn(unittest.TestCase):
    def test_binary_data(self):
        from ml.utils.numeric_functions import is_binary
        binary = np.asarray([0,0,0,1,1,1,0,1])
        self.assertEqual(is_binary(binary, include_null=False), True)
        self.assertEqual(is_binary(binary, include_null=True), True)
        ternary = np.asarray([0,0,0,np.nan,1,1,0,np.nan])
        self.assertEqual(is_binary(ternary, include_null=True), True)
        self.assertEqual(is_binary(ternary, include_null=False), False)

    def test_categorical_data(self):
        from ml.utils.numeric_functions import is_integer
        integer = np.asarray([0,0,0,1,1,1,0,1,2,3,4,5])
        self.assertEqual(is_integer(integer), True)
        self.assertEqual(is_integer(integer), True)
        integer = np.asarray([-1,0,0,np.nan,1,1,0,1,2,3,4,5])
        self.assertEqual(is_integer(integer), True)
        integer = np.asarray([0,0,0,np.nan,1.1,1,0,1,2,3,4,5])
        self.assertEqual(is_integer(integer), False)

    def test_categorical_if(self):
        from ml.utils.numeric_functions import is_integer_if
        binary = np.asarray([0,0,0,1,1,1,0,1])
        self.assertEqual(is_integer_if(binary, card_size=2), True)
        self.assertEqual(is_integer_if(binary, card_size=3), False)
        binary = np.asarray([0,0,0,1,np.nan,1,0,1])
        self.assertEqual(is_integer_if(binary, card_size=3), True)

    def test_index_if_type(self):
        from ml.utils.numeric_functions import index_if_type_row, index_if_type_col
        from ml.utils.numeric_functions import is_binary, is_integer

        array = np.asarray([
            [0, 1, 2, 0, 4, 5],
            [1, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 5, 7, 8, 10],
            [0, 0,.3,.1, 0, 1],
            [0, 1, np.nan, 0, 1, 1]
        ])
        self.assertEqual(index_if_type_row(array, is_binary), [1, 2, 5])
        self.assertEqual(index_if_type_row(array, is_binary, include_null=False), [1, 2])
        self.assertEqual(index_if_type_row(array, is_integer), [0, 1, 2, 3, 5])
        self.assertEqual(index_if_type_col(array, is_binary), [0, 1])
        self.assertEqual(index_if_type_col(array, is_integer), [0, 1, 4, 5])

    def test_features2rows(self):
        from ml.utils.numeric_functions import features2rows
        data = np.asarray([['a', 'b'], ['c', 'd'], ['e', 'f']])
        f2r = features2rows(data)
        self.assertCountEqual(f2r[0], ['0', 'a'])
        self.assertCountEqual(f2r[1], ['0', 'c']) 
        self.assertCountEqual(f2r[2], ['0', 'e']) 
        self.assertCountEqual(f2r[3], ['1', 'b']) 
        self.assertCountEqual(f2r[4], ['1', 'd']) 
        self.assertCountEqual(f2r[5], ['1', 'f'])

    def test_data_type(self):
        from ml.utils.numeric_functions import data_type, unique_size

        array = np.asarray([
            [0, 1, 2, 0, 4, 5],
            [1, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 5, 7, 8, 10],
            [0, 0,.3,.1, 0, 1],
            [0, -1, np.nan, 0, 1, 1]
        ])

        d_type = []
        for column in array.T:
            d_type.append(data_type(unique_size(column)).name)
        self.assertEqual(d_type, ['boolean', 'nan boolean', 'int', 'int', 'int', 'int'])

    def test_max_type(self):
        from ml.utils.numeric_functions import max_type
        items = [True, True, 1]
        type_e = max_type(items)
        self.assertEqual(int, type_e)
        
        items = [1, False, 1/3.]
        type_e = max_type(items)
        self.assertEqual(float, type_e)

        items = [1, False, 'a', 1.1]
        type_e = max_type(items)
        self.assertEqual(str, type_e)

    def test_nested_shape(self):
        from ml.utils.numeric_functions import nested_shape
        x = [1, [1, 2], [[3, 4], [5, 6]]]
        dtypes = [("a", np.dtype("int")), ("b", np.dtype("int")), ("c", np.dtype("int"))]
        shape = nested_shape(x, dtypes)
        self.assertEqual(shape["a"], [])
        self.assertEqual(shape["b"], [2])
        self.assertEqual(shape["c"], [2, 2])

    def test_nested_shape_one_field(self):
        from ml.utils.numeric_functions import nested_shape
        x = [[1, 2], [2, 3], [1, 1]]
        dtypes = [("a", np.dtype("int"))]
        shape = nested_shape(x, dtypes)
        self.assertEqual(shape["a"], [3, 2])

        x = [[1, 2], [2, 3], [1, ]]
        dtypes = [("a", np.dtype("int"))]
        try:
            shape = nested_shape(x, dtypes)
        except:
            pass
        else:
            self.assertEqual(1, 0)


def count_values(data, y, v):
    true_values = len([e for e in data[:, y] == v if e])
    return true_values*100 / float(data.shape[0]), true_values


if __name__ == '__main__':
    unittest.main()
