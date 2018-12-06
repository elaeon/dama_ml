from sklearn.calibration import CalibratedClassifierCV
from ml.clf.wrappers import SKL, SKLP
from ml.models import MLModel


class SVC(SKL):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn import svm

        with self.train_ds, self.validation_ds:
            model = CalibratedClassifierCV(svm.LinearSVC(C=1, max_iter=1000), method="sigmoid")
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        from sklearn import svm
        
        model = CalibratedClassifierCV(
            svm.LinearSVC(C=1, max_iter=1000), method="sigmoid")
        return self.ml_model(model)


class RandomForest(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn.ensemble import RandomForestClassifier
        model = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        with self.train_ds, self.validation_ds:
            model_clf = model.fit(self.train_ds[self.data_group].to_ndarray().reshape(70, 10),
                                  self.train_ds[self.target_group].to_ndarray().reshape(70))
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds[self.data_group].to_ndarray().reshape(10, 10),
                          self.validation_ds[self.target_group].to_ndarray().reshape(10))
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        from sklearn.ensemble import RandomForestClassifier
        
        model = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        return self.ml_model(model)
    

class ExtraTrees(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn.ensemble import ExtraTreesClassifier

        with self.train_ds, self.validation_ds:
            model = CalibratedClassifierCV(
                ExtraTreesClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        from sklearn.ensemble import ExtraTreesClassifier
        
        model = CalibratedClassifierCV(
            ExtraTreesClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        return self.ml_model(model)


class LogisticRegression(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn.linear_model import LogisticRegression

        with self.train_ds, self.validation_ds:
            model = CalibratedClassifierCV(
                LogisticRegression(solver="lbfgs", multi_class="multinomial", n_jobs=-1), method="sigmoid")
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        from sklearn.linear_model import LogisticRegression
        
        model = CalibratedClassifierCV(
            LogisticRegression(solver="lbfgs", multi_class="multinomial", n_jobs=-1), 
            method="sigmoid")
        return self.ml_model(model)


class SGDClassifier(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn.linear_model import SGDClassifier

        with self.train_ds, self.validation_ds:
            model = CalibratedClassifierCV(SGDClassifier(loss='log', penalty='elasticnet', 
                alpha=.0001, n_iter=100, n_jobs=-1), method="sigmoid")
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        from sklearn.linear_model import SGDClassifier
        
        model = CalibratedClassifierCV(
            SGDClassifier(loss='log', penalty='elasticnet', 
            alpha=.0001, n_iter=100, n_jobs=-1), method="sigmoid")
        return self.ml_model(model)


class AdaBoost(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn.ensemble import AdaBoostClassifier

        with self.train_ds, self.validation_ds:
            model = CalibratedClassifierCV(
                AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)

    def prepare_model_k(self, obj_fn=None):
        from sklearn.ensemble import AdaBoostClassifier
        
        model = CalibratedClassifierCV(
            AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        return self.ml_model(model)


class GradientBoost(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn.ensemble import GradientBoostingClassifier

        with self.train_ds, self.validation_ds:
            model = CalibratedClassifierCV(
                GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)
    
    def prepare_model_k(self, obj_fn=None):
        from sklearn.ensemble import GradientBoostingClassifier
        
        model = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        return self.ml_model(model)


class KNN(SKLP):
    def prepare_model(self, obj_fn=None, num_steps=None):
        from sklearn.neighbors import KNeighborsClassifier

        with self.train_ds, self.validation_ds:
            model = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=2, weights='distance', 
                algorithm='auto'), method="sigmoid")
            model_clf = model.fit(self.train_ds.data, self.train_ds.labels)
            reg_model = CalibratedClassifierCV(model_clf, method="sigmoid", cv="prefit")
            reg_model.fit(self.validation_ds.data, self.validation_ds.labels)
        return self.ml_model(reg_model)
    
    def prepare_model_k(self, obj_fn=None):
        from sklearn.neighbors import KNeighborsClassifier
        
        model = CalibratedClassifierCV(
            KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='auto'), method="sigmoid")
        return self.ml_model(model)
