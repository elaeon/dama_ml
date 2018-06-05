from ml.clf.wrappers import XGB, SKLP
from ml.models import MLModel

from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import os
import numpy as np


class Xgboost(XGB):
    def convert_labels(self, labels, output=None):
        if output is None:
            for chunk in labels:
                yield chunk
        elif output == 'n_dim':
            for chunk in labels:
                if len(chunk.shape) == 1:
                    label = chunk.reshape(-1, 1)
                    yield np.concatenate((np.abs(label - 1), label), axis=1)
                else:
                    yield chunk
        else:
            for chunk in labels:
                tmp_chunk = []
                for e in chunk:
                    tmp_chunk.append(int(round(e, 0)))
                yield self.le.inverse_transform(tmp_chunk)

    def prepare_model(self, obj_fn=None, num_steps=None, **params):
        with self.train_ds, self.validation_ds:
            d_train = xgb.DMatrix(self.train_ds.data[:], self.train_ds.labels[:]) 
            d_valid = xgb.DMatrix(self.validation_ds.data[:], self.validation_ds.labels[:]) 
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        nrounds = 200
        xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=obj_fn, maximize=True, verbose_eval=100)#, tree_method="hist")
        return self.ml_model(xgb, bst=xgb_model)
    
    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, model_params={}):
        from sklearn.model_selection import StratifiedKFold
        nrounds = num_steps
        cv = StratifiedKFold(n_splits=n_splits)
        with self.train_ds, self.validation_ds:
            data = np.concatenate((self.train_ds.data[:], self.validation_ds.data[:]), 
                            axis=0)
            labels = np.concatenate((self.train_ds.labels[:], self.validation_ds.labels[:]), 
                            axis=0)
        for k, (train, test) in enumerate(cv.split(data, labels), 1):
            d_train = xgb.DMatrix(data[train], labels[train]) 
            d_valid = xgb.DMatrix(data[test], labels[test]) 
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            xgb_model = xgb.train(model_params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=obj_fn, maximize=True, verbose_eval=100)
            print("fold ", k)
        return self.ml_model(xgb, bst=xgb_model)


class XgboostSKL(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, **params):
        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        with self.train_ds, self.validation_ds:
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None, **params):
        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
        return self.ml_model(model)
