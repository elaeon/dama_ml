from ml.clf.wrappers import LGB

import lightgbm as lgb


class LightGBM(LGB):
    # def convert_labels(self, labels, output=None):
    #    if output is None:
    #        for chunk in labels:
    #            yield chunk
    #    elif output == 'n_dim':
    #        for chunk in labels:
    #            if len(chunk.shape) == 1:
    #                label = chunk.reshape(-1, 1)
    #                yield np.concatenate((np.abs(label - 1), label), axis=1)
    #            else:
    #                yield chunk
    #    else:
    #        for chunk in labels:
    #            tmp_chunk = []
    #            for e in chunk:
    #                tmp_chunk.append(int(round(e, 0)))
    #            yield self.le.inverse_transform(tmp_chunk)

    def prepare_model(self, obj_fn=None, num_steps=0, model_params=None):
        with self.ds:
            data_train = self.ds[self.data_groups["data_train_group"]].to_ndarray()
            target_train = self.ds[self.data_groups["target_train_group"]].to_ndarray()
            data_val = self.ds[self.data_groups["data_validation_group"]].to_ndarray()
            target_val = self.ds[self.data_groups["target_validation_group"]].to_ndarray()
            columns = None
            data_train_ds = lgb.Dataset(data_train, label=target_train, feature_name=columns)
            data_valid_ds = lgb.Dataset(data_val, label=target_val, feature_name=columns)

        num_round = num_steps
        bst = lgb.train(model_params, data_train_ds, num_round, valid_sets=[data_valid_ds],
                        early_stopping_rounds=num_round/2, feval=obj_fn, verbose_eval=True)
        return self.ml_model(lgb, bst=bst)

    def feature_importance(self):
        import pandas as pd
        gain = self.bst.feature_importance('gain')
        df = pd.DataFrame({'feature': self.bst.feature_name(),
                           'split': self.bst.feature_importance('split'),
                           'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        return df
