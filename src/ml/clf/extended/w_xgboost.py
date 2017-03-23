from ml.clf.wrappers import SKLP
from ml.models import MLModel

import xgboost as xgb
import os

class Xgboost(SKLP):
    def __init__(self, params={}, **kwargs):
        super(SKLP, self).__init__(**kwargs)

    def prepare_model(self):
        from sklearn.calibration import CalibratedClassifierCV
        reg = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return sig_clf
