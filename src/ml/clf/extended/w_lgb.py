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
                tmp_chunk = []
                for e in chunk:
                    tmp_chunk.append(int(round(e, 0)))
                yield self.le.inverse_transform(tmp_chunk)

    def prepare_model(self, obj_fn=None, num_steps=0, **params):
        with self.ds:
            data_train = self.ds[self.data_groups["data_train_group"]].to_ndarray()
            target_train = self.ds[self.data_groups["target_train_group"]].to_ndarray()
            data_val = self.ds[self.data_groups["data_validation_group"]].to_ndarray()
            target_val = self.ds[self.data_groups["target_validation_group"]].to_ndarray()
            columns = None
            data_train_ds = lgb.Dataset(data_train, label=target_train, feature_name=columns)
            data_valid_ds = lgb.Dataset(data_val, label=target_val, feature_name=columns)

        num_round = num_steps
        bst = lgb.train(params, data_train_ds, num_round, valid_sets=[data_valid_ds],
            early_stopping_rounds=num_round/2, feval=obj_fn, verbose_eval=True)
        return self.ml_model(lgb, bst=bst)
    
    def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, model_params={}):
        from sklearn.model_selection import StratifiedKFold
        num_round = num_steps
        cv = StratifiedKFold(n_splits=n_splits)
        with self.train_ds, self.validation_ds:
            columns = list(self.train_ds.columns[:])
            data = np.concatenate((self.train_ds.data[:], self.validation_ds.data[:]), 
                            axis=0)
            labels = np.concatenate((self.train_ds.labels[:], self.validation_ds.labels[:]), 
                            axis=0)
        for k, (train, test) in enumerate(cv.split(data, labels), 1):
            train_data = lgb.Dataset(data[train], label=labels[train], 
                feature_name=columns)
            valid_data = lgb.Dataset(data[test], label=labels[test],
                feature_name=columns)
            lgb_model = lgb.train(model_params, train_data, num_round, 
                valid_sets=[valid_data], early_stopping_rounds=num_round/2, feval=obj_fn, 
                verbose_eval=True)
            print("fold ", k)
        return self.ml_model(lgb, bst=lgb_model)

    def feature_importance(self):
        import pandas as pd
        gain = self.bst.feature_importance('gain')
        df = pd.DataFrame({'feature':self.bst.feature_name(), 
            'split':self.bst.feature_importance('split'), 
            'gain':100 * gain /gain.sum()}).sort_values('gain', ascending=False)
        return df
