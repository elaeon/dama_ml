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
        ds = DataLabel(name="test", dataset_path="/tmp", transforms=transforms, 
            apply_transforms=True, rewrite=True)
        ds.build_dataset(df.as_matrix(), np.asarray(self.outlayers))
        outlayers = ds.outlayers(type_detector='isolation', n_estimators=25, max_samples=10, 
            contamination=self.contamination)
        ds.destroy()
        self.assertItemsEqual(list(outlayers), [0, 1, 13, 15])

    def test_linear_outlayer2(self):
        df = pd.DataFrame(data=self.data)
        transforms = Transforms()
        transforms.add(FitStandardScaler, type="column")
        ds = DataLabel(name="test", dataset_path="/tmp", transforms=transforms, 
            apply_transforms=True, rewrite=True)
        ds.build_dataset(df.as_matrix(), np.asarray(self.outlayers))
        outlayers = ds.outlayers(type_detector='robust', contamination=self.contamination)
        ds.destroy()
        self.assertItemsEqual(list(outlayers), [0, 8, 13, 15])


if __name__ == '__main__':
    unittest.main()
