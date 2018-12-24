from ml.clf.wrappers import XGB, SKLP

# from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


class Xgboost(XGB):

    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None):
        with self.ds:
            data_train = self.ds[self.data_groups["data_train_group"]].to_ndarray()
            target_train = self.ds[self.data_groups["target_train_group"]].to_ndarray()
            data_val = self.ds[self.data_groups["data_validation_group"]].to_ndarray()
            target_val = self.ds[self.data_groups["target_validation_group"]].to_ndarray()
            d_train = xgb.DMatrix(data_train, target_train)
            d_valid = xgb.DMatrix(data_val, target_val)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        nrounds = 200
        xgb_model = xgb.train(model_params, d_train, nrounds, watchlist, early_stopping_rounds=100,
                              feval=obj_fn, maximize=True, verbose_eval=100)
        return self.ml_model(xgb, bst=xgb_model)
    
    # def train_kfolds(self, batch_size=0, num_steps=0, n_splits=2, obj_fn=None, model_params={}):
    #    from sklearn.model_selection import StratifiedKFold
    #    nrounds = num_steps
    #    cv = StratifiedKFold(n_splits=n_splits)
    #    with self.train_ds, self.validation_ds:
    #        data = np.concatenate((self.train_ds.data[:], self.validation_ds.data[:]),
    #                        axis=0)
    #        labels = np.concatenate((self.train_ds.labels[:], self.validation_ds.labels[:]),
    #                        axis=0)
    #    for k, (train, test) in enumerate(cv.split(data, labels), 1):
    #        d_train = xgb.DMatrix(data[train], labels[train])
    #        d_valid = xgb.DMatrix(data[test], labels[test])
    #        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    #        xgb_model = xgb.train(model_params, d_train, nrounds, watchlist, early_stopping_rounds=100,
    #                      feval=obj_fn, maximize=True, verbose_eval=100)
    #        print("fold ", k)
    #    return self.ml_model(xgb, bst=xgb_model)


# class XgboostSKL(SKLP):
#    def prepare_model(self, obj_fn=None, num_steps=None, **params):
#        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
#        with self.train_ds, self.validation_ds:
#            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
#            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
#            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
#        return self.ml_model(reg_model)

#    def prepare_model_k(self, obj_fn=None, **params):
#        model = CalibratedClassifierCV(xgb.XGBClassifier(seed=3, n_estimators=25), method="sigmoid")
#        return self.ml_model(model)
