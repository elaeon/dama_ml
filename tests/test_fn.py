import unittest
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale


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
        df = pd.DataFrame(data=scale(df))
        clf = IsolationForest(max_samples=10,
            contamination=self.contamination,
            random_state=40,
            n_jobs=-1)
        matrix = df.as_matrix()
        clf.fit(matrix)
        y_pred = clf.predict(matrix)
        self.assertItemsEqual(y_pred, self.outlayers)

    def test_linear_outlayer2(self):
        from sklearn.covariance import EmpiricalCovariance, MinCovDet
        df = pd.DataFrame(data=self.data)
        df = pd.DataFrame(data=scale(df))
        matrix = df.as_matrix()
        robust_cov = MinCovDet().fit(matrix)
        robust_mahal = robust_cov.mahalanobis(matrix - robust_cov.location_)
        limit = int(round(len(robust_mahal)*self.contamination))
        threshold = sorted(robust_mahal, reverse=True)[limit]
        outlayers = [1 if val < threshold else -1 for val in robust_mahal]
        self.assertItemsEqual(outlayers, self.outlayers)


if __name__ == '__main__':
    unittest.main()
