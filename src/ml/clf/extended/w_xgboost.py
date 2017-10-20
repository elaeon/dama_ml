from ml.clf.wrappers import XGB, SKLP
from ml.models import MLModel

from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import os


class Xgboost(XGB):
    def prepare_model(self, obj_fn=None):
        params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}
        d_train = xgb.DMatrix(self.dataset.train_data, self.dataset.train_labels) 
        d_valid = xgb.DMatrix(self.dataset.validation_data, self.dataset.validation_labels) 
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        nrounds = 200
        xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=obj_fn, maximize=True, verbose_eval=100)
        return self.ml_model(xgb, model_2=xgb_model)

    #def prepare_model_k(self, obj_fn=None):
        #model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        #return self.ml_model(model)


class XgboostSKL(SKLP):
    def prepare_model(self, obj_fn=None):
        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        model_clf = model.fit(self.dataset.train_data, self.dataset.train_labels)
        reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
        reg_model.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        return self.ml_model(model)
