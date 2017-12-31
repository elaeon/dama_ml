from ml.reg.wrappers import SKLP
from ml.models import MLModel


class RandomForestRegressor(SKLP):
    def prepare_model(self, obj_fn=None):
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=25, min_samples_split=2)
        with self.train_ds:
            reg_model = model.fit(self.train_ds.data, self.train_ds.labels)
        return self.ml_model(reg_model)
    
