from ml.clf.wrappers import SKLP
from ml.models import MLModel

from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import os


class Xgboost(SKLP):
    def __init__(self, params={}, **kwargs):
        super(SKLP, self).__init__(**kwargs)

    def prepare_model(self):
        reg = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)

    def prepare_model_k(self):
        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        return self.ml_model(model)
