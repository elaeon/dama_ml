from ml.reg.wrappers import SKLP
from ml.models import MLModel


class RandomForestRegressor(SKLP):
    def prepare_model(self, obj_fn=None):
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=25, min_samples_split=2)
        with self.train_ds:
            reg_model = model.fit(self.train_ds.data, self.train_ds.labels)
        return self.ml_model(reg_model)
    

class GradientBoostingRegressor(SKLP):
    def prepare_model(self, obj_fn=None):
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(learning_rate=0.2, random_state=3)
        with self.train_ds:
            reg_model = model.fit(self.train_ds.data, self.train_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        from sklearn.ensemble import GradientBoostingRegressor
        
        model = GradientBoostingRegressor(n_estimators=25, learning_rate=1.0)
        return self.ml_model(model)
