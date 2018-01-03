import unittest
import numpy as np
import pandas as pd

from ml.processing import Transforms, FitStandardScaler
from ml.ds import DataLabel


class TestLinearOutLayer(unittest.TestCase):
    def setUp(self):
        self.data = {
            "height": [164, 150, 158, 160, 161, 160, 165, 165, 171, 172, 172, 173, 173, 175, 176, 178], 
            "weight": [ 84,  55,  58,  60,  61,  60,  63,  62,  68,  65,  64,  62,  64,  56,  66,  70],
            "gender": [ -1,   1,   1,   1,   1,   1,   1,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   1]}
        # -1 are outlayers
        self.outlayers =  [ -1,   -1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  -1,   1,  -1]
        self.contamination = .2

    def test_linear_outlayer1(self):
        from sklearn.ensemble import IsolationForest
        df = pd.DataFrame(data=self.data)
        transforms = Transforms()
        transforms.add(FitStandardScaler, type="column")
        dl = DataLabel(name="test", dataset_path="/tmp", transforms=transforms, 
            apply_transforms=True, rewrite=True)
        with dl:
            dl.build_dataset(df.as_matrix(), np.asarray(self.outlayers))
            outlayers = dl.outlayers(type_detector='isolation', n_estimators=25, max_samples=10, 
                contamination=self.contamination)
        dl.destroy()
        self.assertItemsEqual(list(outlayers), [0, 1, 13, 15])

    def test_linear_outlayer2(self):
        df = pd.DataFrame(data=self.data)
        transforms = Transforms()
        transforms.add(FitStandardScaler, type="column")
        ds = DataLabel(name="test", dataset_path="/tmp", transforms=transforms, 
            apply_transforms=True, rewrite=True)
        with ds:
            ds.build_dataset(df.as_matrix(), np.asarray(self.outlayers))
            outlayers = ds.outlayers(type_detector='robust', contamination=self.contamination)
        ds.destroy()
        self.assertItemsEqual(list(outlayers), [0, 8, 13, 15])


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
        self.assertItemsEqual(f2r[0], ['0', 'a'])
        self.assertItemsEqual(f2r[1], ['0', 'c']) 
        self.assertItemsEqual(f2r[2], ['0', 'e']) 
        self.assertItemsEqual(f2r[3], ['1', 'b']) 
        self.assertItemsEqual(f2r[4], ['1', 'd']) 
        self.assertItemsEqual(f2r[5], ['1', 'f'])

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
            d_type.append(data_type(unique_size(column), column.size).name)
        self.assertEqual(d_type, ['boolean', 'nan boolean', 'ordinal', 'ordinal', 'ordinal', 'ordinal'])


if __name__ == '__main__':
    unittest.main()
