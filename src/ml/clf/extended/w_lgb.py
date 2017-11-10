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
        train_data = lgb.Dataset(self.dataset.train_data, label=self.dataset.train_labels)
        valid_data = lgb.Dataset(self.dataset.validation_data, label=self.dataset.validation_labels)
        num_round = 200
        bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], 
            early_stopping_rounds=num_round/2, feval=obj_fn, verbose_eval=True)
        return self.ml_model(lgb, model_2=bst)
    
    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, model_params={}):
        from sklearn.model_selection import StratifiedKFold
        num_round = num_steps
        cv = StratifiedKFold(n_splits=n_splits)
        data = self.dataset.data_validation
        labels = self.dataset.data_validation_labels
        for k, (train, test) in enumerate(cv.split(data, labels), 1):
            train_data = lgb.Dataset(data[train], label=labels[train])
            valid_data = lgb.Dataset(data[test], label=labels[test])
            lgb_model = lgb.train(model_params, train_data, num_round, 
                valid_sets=[valid_data], early_stopping_rounds=num_round/2, feval=obj_fn, 
                verbose_eval=True)
            print("fold ", k)
        return self.ml_model(lgb, model_2=lgb_model)

