from ml.clf.wrappers import XGB, SKLP
from ml.models import MLModel

from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import os
import numpy as np


class Xgboost(XGB):
    def convert_label(self, label, raw=False):
        if raw is True:
            if len(label.shape) == 1:
                label = label.reshape(-1, 1)
                return np.concatenate((np.abs(label - 1), label), axis=1)
            else:
                return label
        elif raw is None:
            return self.position_index(label)
        else:
            return self.le.inverse_transform(self.position_index(label))

    def prepare_model(self, obj_fn=None, **params):
        with self.dataset:
            d_train = xgb.DMatrix(self.dataset.train_data, self.dataset.train_labels) 
            d_valid = xgb.DMatrix(self.dataset.validation_data, self.dataset.validation_labels) 
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        nrounds = 200
        xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=obj_fn, maximize=True, verbose_eval=100)#, tree_method="hist")
        return self.ml_model(xgb, model_2=xgb_model)
    
    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, model_params={}):
        from sklearn.model_selection import StratifiedKFold
        nrounds = num_steps
        cv = StratifiedKFold(n_splits=n_splits)
        with self.dataset:
            data = self.dataset.data_validation
            labels = self.dataset.data_validation_labels
        for k, (train, test) in enumerate(cv.split(data, labels), 1):
            d_train = xgb.DMatrix(data[train], labels[train]) 
            d_valid = xgb.DMatrix(data[test], labels[test]) 
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            xgb_model = xgb.train(model_params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=obj_fn, maximize=True, verbose_eval=100)
            print("fold ", k)
        return self.ml_model(xgb, model_2=xgb_model)


class XgboostSKL(SKLP):
    def prepare_model(self, obj_fn=None, **params):
        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        with self.dataset:
            model_clf = model.fit(self.dataset.train_data, self.dataset.train_labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None, **params):
        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        return self.ml_model(model)
