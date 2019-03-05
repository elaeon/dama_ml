from dama.reg.wrappers import LGB
import lightgbm as lgb


class LightGBM(LGB):

    def prepare_model(self, obj_fn=None, num_steps=0, model_params=None, batch_size: int = None):
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
        df = pd.DataFrame({'feature':self.bst.feature_name(),
            'split':self.bst.feature_importance('split'), 
            'gain':100 * gain /gain.sum()}).sort_values('gain', ascending=False)
        return df
