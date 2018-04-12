from ml.reg.wrappers import LGB
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

    def prepare_model(self, obj_fn=None, num_steps=0, **params):
        with self.train_ds, self.validation_ds:
            train_data = lgb.Dataset(self.train_ds.data[:], label=self.train_ds.labels[:])
            valid_data = lgb.Dataset(self.validation_ds.data[:], label=self.validation_ds.labels[:])
        num_round = num_steps
        bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], 
            early_stopping_rounds=num_round/2, feval=obj_fn, verbose_eval=True)
        return self.ml_model(lgb, model_2=bst)

