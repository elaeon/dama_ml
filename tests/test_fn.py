import unittest
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.ensemble import IsolationForest


class TestFn(unittest.TestCase):
    def setUp(self):
        pass

    def test_outlayer(self):
        data = {
            "height": [164, 150, 158, 160, 161, 160, 165, 165, 171, 172, 172, 173, 173, 175, 176, 178], 
            "weight": [ 84,  55,  58,  60,  61,  60,  63,  62,  68,  65,  64,  62,  64,  56,  66,  70],
            "gender": [ -1,   1,   1,   1,   1,   1,   1,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   1]}
        outlayers =  [ -1,   -1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  -1,   1,  -1] 
        df = pd.DataFrame(data=data)
        df = pd.DataFrame(data=scale(df))
        clf = IsolationForest(max_samples=10,
            contamination=.20,
            random_state=40,
            n_jobs=-1)
        matrix = df.as_matrix()
        clf.fit(matrix)
        y_pred = clf.predict(matrix)
        self.assertItemsEqual(y_pred, outlayers)


if __name__ == '__main__':
    unittest.main()
