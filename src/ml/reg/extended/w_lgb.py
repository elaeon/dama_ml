from ml.reg.wrappers import LGB
from ml.models import MLModel

import lightgbm as lgb
import os
import numpy as np


class LightGBM(LGB):
    def convert_label(self, labels, output=None):
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
                    yield label

    def prepare_model(self, obj_fn=None, num_steps=0, **params):
        with self.train_ds, self.validation_ds:
            columns = list(self.train_ds.columns[:])
            train_data = lgb.Dataset(self.train_ds.data[:], 
                label=self.train_ds.labels[:], feature_name=columns)
            valid_data = lgb.Dataset(self.validation_ds.data[:], 
                    label=self.validation_ds.labels[:], feature_name=columns)
        num_round = num_steps
        bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data], 
            early_stopping_rounds=num_round/2, feval=obj_fn, verbose_eval=True)
        return self.ml_model(lgb, bst=bst)

    def feature_importance(self):
        import pandas as pd
        gain = self.bst.feature_importance('gain')
        df = pd.DataFrame({'feature':self.bst.feature_name(), 
            'split':self.bst.feature_importance('split'), 
            'gain':100 * gain /gain.sum()}).sort_values('gain', ascending=False)
        return df
