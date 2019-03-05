from dama.clf.wrappers import XGB

# from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


class Xgboost(XGB):

    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        data_train = self.ds[self.data_groups["data_train_group"]].to_ndarray()
        target_train = self.ds[self.data_groups["target_train_group"]].to_ndarray()
        data_val = self.ds[self.data_groups["data_validation_group"]].to_ndarray()
        target_val = self.ds[self.data_groups["target_validation_group"]].to_ndarray()
        d_train = xgb.DMatrix(data_train, target_train)
        d_valid = xgb.DMatrix(data_val, target_val)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        nrounds = num_steps
        xgb_model = xgb.train(model_params, d_train, nrounds, watchlist, early_stopping_rounds=nrounds/2,
                              feval=obj_fn, maximize=True, verbose_eval=100)
        return self.ml_model(xgb, bst=xgb_model)
