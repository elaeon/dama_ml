import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

class TestFn(unittest.TestCase):
    def setUp(self):
        pass
        #self.transforms = Transforms()

    def test_outlayer(self):
        data = {
            "height": [164, 150, 158, 160, 161, 160, 165, 165, 171, 172, 172, 173, 173, 175, 176, 178], 
            "weight": [ 84,  55,  58,  60,  61,  60,  63,  62,  68,  65,  64,  62,  64,  56,  66,  70],
            "gender": [ -1,   1,   1,   1,   1,   1,   1,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   1]}
        df = pd.DataFrame(data=data)
        del df["gender"]
        df = pd.DataFrame(data=scale(df))
        print(df)
        median = df.median()
        consistency_constant = 1.4826 #1
        mad = abs(df - median).median() * consistency_constant
        adm = abs(df - df.median()) / mad
        print(mad)
        print(adm)
        #print(df)
        #points = np.asarray(df)
        #print(points)
        #median = np.median(points)
        #diff = np.sum((points - median)**2, axis=-1)
        #diff = np.sqrt(diff)
        #med_abs_deviation = np.median(diff)
        #modified_z_score = 0.6745 * diff / med_abs_deviation
        #print(modified_z_score)
        #print(modified_z_score > 2)
        #print(df.describe())

if __name__ == '__main__':
    unittest.main()
