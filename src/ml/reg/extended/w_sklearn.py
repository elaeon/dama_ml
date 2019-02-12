from ml.reg.wrappers import SKLP
from sklearn.ensemble import RandomForestRegressor as SkRandomForestReg
from sklearn.ensemble import GradientBoostingRegressor as SkGradientBoostingReg
import pandas as pd


class RandomForestRegressor(SKLP):
    def prepare_model(self, obj_fn=None, num_steps: int = 0, model_params=None, batch_size: int = None):
        model = SkRandomForestReg(**model_params)
        reg_model = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                              self.ds[self.data_groups["target_train_group"]].to_ndarray())
        return self.ml_model(reg_model)

    def feature_importance(self):
        df = pd.DataFrame({'importance': self.model.model.feature_importances_, 'gain': None}).sort_values(
            by=['importance'], ascending=False)
        return df
    

class GradientBoostingRegressor(SKLP):
    def prepare_model(self, obj_fn=None, num_steps: int = 0, model_params=None, batch_size: int = None):
        model = SkGradientBoostingReg(**model_params)
        reg_model = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                              self.ds[self.data_groups["target_train_group"]].to_ndarray())
        return self.ml_model(reg_model)
