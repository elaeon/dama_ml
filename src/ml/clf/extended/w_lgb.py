from ml.clf.wrappers import LGB
from ml.models import MLModel

import lightgbm as lgb
import os
import numpy as np


class LightGBM(LGB):
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
        with self.train_ds, self.validation_ds:
            train_data = lgb.Dataset(self.train_ds.data, label=self.train_ds.labels)
            valid_data = lgb.Dataset(self.validation_ds.data, label=self.validation_ds.labels)
        num_round = 200
        bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], 
            early_stopping_rounds=num_round/2, feval=obj_fn, verbose_eval=True)
        return self.ml_model(lgb, model_2=bst)
    
    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, model_params={}):
        from sklearn.model_selection import StratifiedKFold
        num_round = num_steps
        cv = StratifiedKFold(n_splits=n_splits)
        with self.train_ds, self.validation_ds:
            data = np.concatenate((self.train_ds.data[:], self.validation_ds.data[:]), 
                            axis=0)
            labels = np.concatenate((self.train_ds.labels[:], self.validation_ds.labels[:]), 
                            axis=0)
        for k, (train, test) in enumerate(cv.split(data, labels), 1):
            train_data = lgb.Dataset(data[train], label=labels[train])
            valid_data = lgb.Dataset(data[test], label=labels[test])
            lgb_model = lgb.train(model_params, train_data, num_round, 
                valid_sets=[valid_data], early_stopping_rounds=num_round/2, feval=obj_fn, 
                verbose_eval=True)
            print("fold ", k)
        return self.ml_model(lgb, model_2=lgb_model)

