import unittest
import numpy as np
import pandas as pd

from ml.processing import Transforms, FitStandardScaler
from ml.data.ds import DataLabel
from ml.random import downsample, sampling_size


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
        transforms.add(FitStandardScaler)
        dl = DataLabel(name="test", dataset_path="/tmp", clean=True)
        dl.transforms = transforms 
        dl.apply_transforms = True 
        with dl:
            dl.from_data(df.as_matrix(), np.asarray(self.outlayers))
            outlayers = dl.outlayers(type_detector='isolation', n_estimators=25, max_samples=10, 
                contamination=self.contamination)
        dl.destroy()
        self.assertItemsEqual(list(outlayers), [0, 1, 13, 15])

    def test_linear_outlayer2(self):
        df = pd.DataFrame(data=self.data)
        transforms = Transforms()
        transforms.add(FitStandardScaler)
        ds = DataLabel(name="test", dataset_path="/tmp", clean=True)
        ds.transforms = transforms 
        ds.apply_transforms = True 
        with ds:
            ds.from_data(df.as_matrix(), np.asarray(self.outlayers))
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

    def test_downsample(self):
        size = 5000
        data = np.random.rand(size, 3)
        data[:, 2] = data[:, 2] <= .9
        v = downsample(data, {0: 200, 1:240}, 2, size)
        true_values = count_values(v.to_memory(), 2, 1)
        self.assertEqual(true_values[0] > 50, True)
        self.assertEqual(true_values[1], 240)

    def test_downsample_params(self):
        size = 5000
        data = np.random.rand(size, 3)
        data[:, 2] = data[:, 2] <= .9
        _, counter = np.unique(data[:, 2], return_counts=True)
        sampling_v = sampling_size({0: 2000, 1:4840}, data[:, 2])
        v = downsample(data, sampling_v, 2, size).to_memory()
        self.assertEqual(count_values(v, 2, 0)[1], counter[0])
        self.assertEqual(count_values(v, 2, 1)[1], counter[1])

        t0 = int(round(counter[0]*.2, 0))
        t1 = int(round(counter[1]*.4, 0))
        sampling_v = sampling_size({0: .2, 1:.4}, data[:, 2])
        v = downsample(data, sampling_v, 2, size).to_memory()
        self.assertEqual(count_values(v, 2, 0)[1], t0)
        self.assertEqual(count_values(v, 2, 1)[1], t1)

    def test_downsample_small(self):
        size = 10
        data = np.random.rand(size, 3)
        data[:, 2] = data[:, 2] <= .9
        v = downsample(data, {0: 0, 1:3}, 2, size)
        self.assertItemsEqual(v.to_memory()[:, 2], [1,1,1])

    def test_downsample_static(self):
        data = [0,0,0,1,1,1,1,1,2,2,2]
        size = len(data)
        v = downsample(data, {0: 2, 1: 4}, None, size)
        self.assertItemsEqual(v.to_memory(), [0,0,1,1,1,1])
        v = downsample(data, {1: 4}, None, size)
        self.assertItemsEqual(v.to_memory(), [1,1,1,1])
        v = downsample(data, {2: 2, 1: 4}, None, size)
        self.assertItemsEqual(v.to_memory(), [2,2,1,1,1,1])


def count_values(data, y, v):
    true_values = len([e for e in data[:, y] == v if e])
    return true_values*100 / float(data.shape[0]), true_values


if __name__ == '__main__':
    unittest.main()
