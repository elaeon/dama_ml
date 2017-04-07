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

    def prepare_model_k(self):
        from sklearn.ensemble import RandomForestClassifier
        
        model = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        return self.ml_model(model)
    

class ExtraTrees(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import ExtraTreesClassifier

        reg = CalibratedClassifierCV(
            ExtraTreesClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)

    def prepare_model_k(self):
        from sklearn.ensemble import ExtraTreesClassifier
        
        model = CalibratedClassifierCV(
            ExtraTreesClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        return self.ml_model(model)


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

    def prepare_model_k(self):
        from sklearn.linear_model import LogisticRegression
        
        model = CalibratedClassifierCV(
            LogisticRegression(solver="lbfgs", multi_class="multinomial", n_jobs=-1), 
            method="sigmoid")
        return self.ml_model(model)


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

    def prepare_model_k(self):
        from sklearn.linear_model import SGDClassifier
        
        model = CalibratedClassifierCV(
            SGDClassifier(loss='log', penalty='elasticnet', 
            alpha=.0001, n_iter=100, n_jobs=-1), method="sigmoid")
        return self.ml_model(model)


class AdaBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import AdaBoostClassifier

        reg = CalibratedClassifierCV(
            AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)

    def prepare_model_k(self):
        from sklearn.ensemble import AdaBoostClassifier
        
        model = CalibratedClassifierCV(
            AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        return self.ml_model(model)


class GradientBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import GradientBoostingClassifier

        reg = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)
    
    def prepare_model_k(self):
        from sklearn.ensemble import GradientBoostingClassifier
        
        model = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        return self.ml_model(model)


class KNN(SKLP):
    def prepare_model(self):
        from sklearn.neighbors import KNeighborsClassifier

        reg = CalibratedClassifierCV(
            KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='auto'), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.validation_data, self.dataset.validation_labels)
        return self.ml_model(sig_clf)
    
    def prepare_model_k(self):
        from sklearn.neighbors import KNeighborsClassifier
        
        model = CalibratedClassifierCV(
            KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='auto'), method="sigmoid")
        return self.ml_model(model)
