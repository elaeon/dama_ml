from sklearn.calibration import CalibratedClassifierCV
from ml.clf.wrappers import SKL, SKLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression as LReg
from sklearn.linear_model import SGDClassifier as SGDClassif
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


class SVC(SKL):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        model = CalibratedClassifierCV(svm.LinearSVC(**model_params), method="sigmoid")
        with self.ds:
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)


class RandomForest(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        if model_params is None:
            model_params = dict(n_estimators=25, min_samples_split=2)
        model = CalibratedClassifierCV(RandomForestClassifier(**model_params), method="sigmoid")
        with self.ds:
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)
    

class ExtraTrees(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        model = CalibratedClassifierCV(ExtraTreesClassifier(**model_params), method="sigmoid")
        with self.ds:
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)


class LogisticRegression(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        model = CalibratedClassifierCV(LReg(**model_params), method="sigmoid")
        with self.ds:
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)


class SGDClassifier(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        model = CalibratedClassifierCV(SGDClassif(**model_params), method="sigmoid")
        with self.ds:
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)


class AdaBoost(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):

        with self.ds:
            model = CalibratedClassifierCV(AdaBoostClassifier(**model_params), method="sigmoid")
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)


class GradientBoost(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        model = CalibratedClassifierCV(GradientBoostingClassifier(**model_params), method="sigmoid")
        with self.ds:
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)


class KNN(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None, model_params=None, batch_size: int = None):
        model = CalibratedClassifierCV(KNeighborsClassifier(**model_params), method="sigmoid")
        with self.ds:
            model_clf = model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                                  self.ds[self.data_groups["target_train_group"]].to_ndarray())
            cal_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            cal_model.fit(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                          self.ds[self.data_groups["target_validation_group"]].to_ndarray())
        return self.ml_model(cal_model)
