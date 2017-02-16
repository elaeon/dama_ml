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
            random_state=rng,
            n_jobs=-1)
        matrix = df.as_matrix()
        clf.fit(matrix)
        y_pred = clf.predict(matrix)
        self.assertItemsEqual(y_pred, outlayers)

    def parametric_tsne(self):
        from ml.ds import Data
        from ml.ae.extended.w_keras import PTsne

        dataset = Data(
            name="tsne", 
            dataset_path="/tmp/")
        X = np.random.rand(100, 10)
        Y = np.sin(6*X)
        dataset.build_dataset(Y)
        classif = PTsne(model_name="tsne", model_version="1", 
            check_point_path="/tmp/", dataset=dataset, dim=3)
        classif.train(batch_size=8, num_steps=2)

        classif = PTsne(model_name="tsne", model_version="1", 
            check_point_path="/tmp/")
        self.assertEqual(len(list(classif.predict([X[1]]))[0]), 2)


if __name__ == '__main__':
    unittest.main()
