from ml.clf.generic import SKL, SKLP
from sklearn.calibration import CalibratedClassifierCV


class OneClassSVM(SKL):
    def __init__(self, *args, **kwargs):
        super(OneClassSVM, self).__init__(*args, **kwargs)
        self.label_ref = 1
        self.label_other = 0

    def prepare_model(self):
        from sklearn import svm
        self.dataset.dataset = self.dataset.train_data
        self.dataset.labels = self.dataset.train_labels
        dataset_ref, _ = self.dataset.only_labels([self.label_ref])
        reg = svm.OneClassSVM(nu=.2, kernel="rbf", gamma=0.5)
        reg.fit(dataset_ref)
        self.model = reg

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data):
            label = self.label_other if prediction == -1 else self.label_ref
            yield self.convert_label(label, raw=raw)


class SVC(SKL):
    def prepare_model(self):
        from sklearn import svm
        reg = CalibratedClassifierCV(
            svm.LinearSVC(C=1, max_iter=1000), method="sigmoid")
        reg = reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class RandomForest(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import RandomForestClassifier
        reg = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class ExtraTrees(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import ExtraTreesClassifier
        reg = CalibratedClassifierCV(
            ExtraTreesClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class LogisticRegression(SKLP):
    def prepare_model(self):
        from sklearn.linear_model import LogisticRegression
        reg = CalibratedClassifierCV(
            LogisticRegression(solver="lbfgs", multi_class="multinomial")#"newton-cg")
            , method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class SGDClassifier(SKLP):
    def prepare_model(self):
        from sklearn.linear_model import SGDClassifier
        reg = CalibratedClassifierCV(
            SGDClassifier(loss='log', penalty='elasticnet', 
            alpha=.0001, n_iter=100, n_jobs=-1), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class AdaBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import AdaBoostClassifier
        reg = CalibratedClassifierCV(
            AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class GradientBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import GradientBoostingClassifier
        reg = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf
