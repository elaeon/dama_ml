from sklearn.calibration import CalibratedClassifierCV
from ml.clf.wrappers import SKL, SKLP
from ml.models import MLModel


class SVC(SKL):
    def prepare_model(self):
        from sklearn import svm

        reg = CalibratedClassifierCV(
            svm.LinearSVC(C=1, max_iter=1000), method="sigmoid")
        reg = reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)

    def prepare_model_k(self):
        from sklearn import svm
        
        model = CalibratedClassifierCV(
            svm.LinearSVC(C=1, max_iter=1000), method="sigmoid")
        return self.ml_model(model)


class RandomForest(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import RandomForestClassifier

        reg = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)


class ExtraTrees(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import ExtraTreesClassifier

        reg = CalibratedClassifierCV(
            ExtraTreesClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)


class LogisticRegression(SKLP):
    def prepare_model(self):
        from sklearn.linear_model import LogisticRegression

        reg = CalibratedClassifierCV(
            LogisticRegression(solver="lbfgs", multi_class="multinomial", n_jobs=-1), 
            method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)


class SGDClassifier(SKLP):
    def prepare_model(self):
        from sklearn.linear_model import SGDClassifier

        reg = CalibratedClassifierCV(
            SGDClassifier(loss='log', penalty='elasticnet', 
            alpha=.0001, n_iter=100, n_jobs=-1), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)


class AdaBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import AdaBoostClassifier

        reg = CalibratedClassifierCV(
            AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)


class GradientBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import GradientBoostingClassifier

        reg = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)
