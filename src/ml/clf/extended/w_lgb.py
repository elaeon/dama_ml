from ml.clf.wrappers import LGB
from ml.models import MLModel

import lightgbm as lgb
import os
import numpy as np


class LightGBM(LGB):
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
                for label in self.position_index(chunk):
                    yield self.le.inverse_transform(int(round(label, 0)))

    def prepare_model(self, obj_fn=None, num_steps=0, **params):
        with self.train_ds, self.validation_ds:
            train_data = lgb.Dataset(self.train_ds.data[:], label=self.train_ds.labels[:])
            valid_data = lgb.Dataset(self.validation_ds.data[:], label=self.validation_ds.labels[:])
        num_round = num_steps
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

