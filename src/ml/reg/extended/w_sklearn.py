from ml.reg.wrappers import SKLP
from ml.models import MLModel


class RandomForestRegressor(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=0, **params):
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=25, min_samples_split=2)
        with self.train_ds:
            reg_model = model.fit(self.train_ds.data, self.train_ds.labels)
        return self.ml_model(reg_model)

    def feature_importance(self):
        import pandas as pd
        
        with self.train_ds:
            df = pd.DataFrame({'importance': self.model.model.feature_importances_, 
                'feature': self.train_ds.columns}).sort_values(
                by=['importance'], ascending=False)
        return df
    

class GradientBoostingRegressor(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=0, **params):
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(learning_rate=0.2, random_state=3)
        with self.train_ds:
            reg_model = model.fit(self.train_ds.data, self.train_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None, num_steps=0, batch_size=0, n_splits=2, model_params={}):
        from sklearn.ensemble import GradientBoostingRegressor
        
        model = GradientBoostingRegressor(n_estimators=25, learning_rate=1.0)
        return self.ml_model(model)
