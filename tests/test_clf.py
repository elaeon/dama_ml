import unittest
import numpy as np
import os
from ml.data.ds import Data
from ml.clf.extended.w_sklearn import RandomForest, SVC, ExtraTrees, LogisticRegression, SGDClassifier
from ml.clf.extended.w_sklearn import AdaBoost, GradientBoost, KNN
from ml.clf.extended.w_keras import FCNet
from ml.utils.model_selection import CV
from ml.measures import gini_normalized
from ml.data.drivers.core import HDF5
from ml.measures import Measure, MeasureBatch
from ml.models import Metadata
from ml.utils.files import check_or_create_path_dir


TMP_PATH = check_or_create_path_dir(os.path.dirname(os.path.abspath(__file__)), 'softstream_data_test')


try:
    from ml.clf.extended.w_xgboost import Xgboost
except ImportError:
    from ml.clf.extended.w_sklearn import RandomForest as Xgboost


try:
    from ml.clf.extended.w_lgb import LightGBM
except ImportError:
    from ml.clf.extended.w_sklearn import RandomForest as LightGBM


def mulp(row):
    return row * 2


def multi_int(xm):
    try:
        return np.asarray([int(x) for x in xm])
    except TypeError:
        return xm


class TestSKL(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.rand(100, 10)
        self.Y = self.X[:, 0] > .5

    def tearDown(self):
        pass

    def test_load_meta(self):
        with Data(name="test", dataset_path=TMP_PATH) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5(mode="w")) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            classif = RandomForest()
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', data_validation_group="validation_x",
                          target_validation_group="validation_y")
            classif.save(name="test_model", path=TMP_PATH, model_version="1")
            dataset.destroy()

        with RandomForest.load(model_name="test_model", path=TMP_PATH, model_version="1") as classif:
            self.assertEqual(classif.model_version, "1")
            self.assertEqual(classif.ds.driver.module_cls_name(), "ml.data.drivers.core.HDF5")
            self.assertEqual(len(classif.metadata_train()), 8)
            classif.destroy()

    def test_empty_load(self):
        with Data(name="test", dataset_path=TMP_PATH) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            classif = RandomForest()
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save(name="test", path=TMP_PATH, model_version="1")
            ds_hash = classif.ds.hash
            dataset.destroy()

        with RandomForest.load(model_name="test", path=TMP_PATH, model_version="1") as classif:
            self.assertEqual(classif.ds.hash, ds_hash)
            classif.destroy()

    def test_scores(self):
        with Data(name="test", dataset_path=TMP_PATH) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            classif = RandomForest()
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save(name="test", path=TMP_PATH, model_version="1")
            scores_table = classif.scores2table()
            classif.destroy()
            dataset.destroy()
            self.assertCountEqual(list(scores_table.headers), ['', 'f1', 'auc', 'recall', 'precision',
                                                           'logloss', 'accuracy'])
            self.assertEqual(scores_table.measures[0][5] <= 1, True)

    def test_new_scores(self):
        with Data(name="test", dataset_path=TMP_PATH) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5(mode="w")) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            metrics = MeasureBatch().make_metrics()
            metrics.add(gini_normalized, greater_is_better=True, output='uncertain')
            classif = RandomForest(metrics=metrics)
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save(name="test", path=TMP_PATH, model_version="1")
            scores_table = classif.scores2table()
            self.assertCountEqual(list(scores_table.headers), ['', 'f1', 'auc', 'recall', 'precision',
                                                           'logloss', 'gini_normalized', 'accuracy'])
            classif.destroy()
            dataset.destroy()

    def test_predict(self):
        x = np.random.rand(100)
        y = x > .5
        with Data(name="test", dataset_path=TMP_PATH, driver=HDF5()) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            dataset.from_data({"x": x.reshape(-1, 1), "y": y})
            classif = RandomForest()
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y',
                          data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save(name="test", path=TMP_PATH, model_version="1")
            values = np.asarray([1, 2, .4, .1, 0, 1])
            dataset.destroy()

        with Data(name="test2") as ds, classif.ds:
            ds.from_data(values)
            for pred in classif.predict(ds):
                pred_array = pred.batch.to_ndarray()
                self.assertCountEqual(pred_array, [True, True, False, False, False, True])
            classif.destroy()
            ds.destroy()

    def test_simple_predict(self):
        with Data(name="test", dataset_path=TMP_PATH, driver=HDF5()) as dataset, \
            Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            dataset.from_data({"x": self.X, "y": self.Y})
            metrics = MeasureBatch()
            metrics.add(gini_normalized, greater_is_better=True, output='uncertain')

            rf = RandomForest(metrics=metrics)
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            rf.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                     data_test_group="test_x", target_test_group='test_y',
                     data_validation_group="validation_x", target_validation_group="validation_y")
            rf.save(name="test_rf", path=TMP_PATH, model_version="1")
            data = dataset["x"]
            predict_shape = rf.predict(data, output='uncertain', batch_size=0).shape
            self.assertEqual(predict_shape, (100, 2))
            predict_shape = rf.predict(data, batch_size=10).shape
            self.assertEqual(predict_shape, (100,))
            rf.destroy()
            dataset.destroy()

    def multi_output(self):
        # classif.output(lambda x, y: ((x + y) / 2).compose(multi_int))
        pass


class TestXgboost(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(100, 10)
        y = (x[:, 0] > .5).astype(int)
        with Data(name="test", dataset_path=TMP_PATH) as self.dataset,\
            Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            self.dataset.from_data({"x": x, "y": y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(self.dataset)
            ds.from_data(stc)

            if Xgboost == RandomForest:
                params = {}
            else:
                params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
            classif = Xgboost()
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save(name="test", path=TMP_PATH, model_version="1")

    def tearDown(self):
        self.dataset.destroy()

    def test_predict(self):
        with Xgboost.load(model_name="test", path=TMP_PATH, model_version="1") as classif:
            predict = classif.predict(self.dataset["x"], batch_size=1)
            for pred in predict.only_data():
                self.assertEqual(pred[0] >= 0, True)
                break
            classif.destroy()


class TestLightGBM(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(100, 10)
        y = (x[:, 0] > .5).astype(int)
        with Data(name="test", dataset_path=TMP_PATH) as self.dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            self.dataset.from_data({"x": x, "y": y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(self.dataset)
            ds.from_data(stc)

            classif = LightGBM()
            if LightGBM == RandomForest:
                self.params = {}
            else:
                self.params={'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
                     'objective': 'binary', 'seed': 99, "verbosity": 0, "learning_rate": 0.1,
                     'boosting_type': "gbdt", 'max_bin': 255, 'num_leaves': 25,
                     'metric': 'binary_logloss'}
            self.num_steps = 10
            self.model_version = "1"
            classif.train(ds, num_steps=self.num_steps, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y', model_params=self.params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save(name="test", path=TMP_PATH, model_version=self.model_version)

    def tearDown(self):
        self.dataset.destroy()

    def test_train_params(self):
        with LightGBM.load(model_name="test", path=TMP_PATH, model_version=self.model_version) as classif:
            self.assertCountEqual(classif.model_params.keys(), self.params.keys())
            self.assertEqual(classif.model_version, self.model_version)
            self.assertEqual(classif.num_steps, self.num_steps)
            self.assertEqual(len(classif.scores2table().measures[0]), 7)
            classif.destroy()

    def test_predict(self):
        with LightGBM.load(model_name="test", path=TMP_PATH, model_version=self.model_version) as classif:
            predict = classif.predict(self.dataset["x"], batch_size=1)[:1]
            for pred in predict.only_data():
                self.assertEqual(pred[0] >= 0, 1)
            classif.destroy()


class TestModelVersion(unittest.TestCase):
    def test_load_add_version(self):
        x = np.random.rand(100)
        y = x > .5
        with Data(name="test", dataset_path=TMP_PATH, driver=HDF5()) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            dataset.from_data({"x": x.reshape(-1, 1), "y": y})

            classif = RandomForest()
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            classif.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y',
                          data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save("test", path=TMP_PATH, model_version="1")

        with RandomForest.load("test", path=TMP_PATH, model_version="1") as classif:
            classif.train(classif.ds, num_steps=10, data_train_group="train_x", target_train_group='train_y',
                      data_test_group="test_x", target_test_group='test_y',
                      data_validation_group="validation_x", target_validation_group="validation_y")
            classif.save("test", path=TMP_PATH, model_version="2")

        with RandomForest.load("test", path=TMP_PATH, model_version="2") as classif2:
            self.assertEqual(classif2.model_version, "2")
            self.assertEqual(classif2.base_path, TMP_PATH)
            self.assertEqual(classif2.num_steps, 10)
            self.assertEqual(classif2.model_name, "test")
            classif2.destroy()

        with dataset:
            dataset.destroy()


class TestKeras(unittest.TestCase):
    def train(self, clf, model_params=None):
        np.random.seed(0)
        x = np.random.rand(100)
        y = x > .5
        with Data(name="test", dataset_path=TMP_PATH, driver=HDF5()) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            dataset.from_data({"x": x.reshape(-1, 1), "y": y})

            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            clf.train(ds, num_steps=2, data_train_group="train_x", target_train_group='train_y', batch_size=10,
                          data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            clf.save("test", path=TMP_PATH, model_version="1")
            dataset.destroy()
        return clf

    def test_predict(self):
        clf = FCNet()
        clf = self.train(clf, model_params=None)
        self.assertEqual(len(clf.scores2table().measures[0]), 7)
        metadata = Metadata.get_metadata(clf.path_metadata, clf.path_metadata_version)
        self.assertEqual(len(metadata["train"]["model_json"]) > 0, True)
        with clf:
            clf.destroy()


class TestWrappers(unittest.TestCase):
    def train(self, clf, model_params=None):
        np.random.seed(0)
        x = np.random.rand(100)
        y = x > .5
        with Data(name="test", dataset_path=TMP_PATH, driver=HDF5()) as dataset, \
                Data(name="test_cv", dataset_path=TMP_PATH, driver=HDF5()) as ds:
            dataset.from_data({"x": x.reshape(-1, 1), "y": y})
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            clf.train(ds, num_steps=1, data_train_group="train_x", target_train_group='train_y',
                          data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                          data_validation_group="validation_x", target_validation_group="validation_y")
            clf.save("test", path=TMP_PATH, model_version="1")
            dataset.destroy()
        self.hash = "$sha1$0d60c62130c4a95634a79abc8e27cebd5fe5bb70"
        return clf

    def test_svc(self):
        clf = SVC()
        clf = self.train(clf, model_params=dict(C=1, max_iter=1000))
        with clf.ds:
            self.assertEqual(clf.ds.hash, self.hash)
            clf.destroy()

    def test_extratrees(self):
        clf = ExtraTrees()
        clf = self.train(clf, model_params=dict(n_estimators=25, min_samples_split=2))
        with clf.ds:
            self.assertEqual(clf.ds.hash, self.hash)
            clf.destroy()

    def test_logistic_regression(self):
        clf = LogisticRegression()
        clf = self.train(clf, model_params=dict(solver="lbfgs", multi_class="multinomial", n_jobs=-1))
        with clf.ds:
            self.assertEqual(clf.ds.hash, self.hash)
            clf.destroy()

    def test_sgd(self):
        clf = SGDClassifier()
        clf = self.train(clf, model_params=dict(loss='log', penalty='elasticnet',
                                                  alpha=.0001, n_iter=100, n_jobs=-1))
        with clf.ds:
            self.assertEqual(clf.ds.hash, self.hash)
            clf.destroy()

    def test_adaboost(self):
        clf = AdaBoost()
        clf = self.train(clf, model_params=dict(n_estimators=25, learning_rate=1.0))
        with clf.ds:
            self.assertEqual(clf.ds.hash, self.hash)
            clf.destroy()

    def test_gradient_boost(self):
        clf = GradientBoost()
        clf = self.train(clf, model_params=dict(n_estimators=25, learning_rate=1.0))
        with clf.ds:
            self.assertEqual(clf.ds.hash, self.hash)
            clf.destroy()

    def test_knn(self):
        clf = KNN()
        clf = self.train(clf, model_params=dict(n_neighbors=2, weights='distance', algorithm='auto'))
        with clf.ds:
            self.assertEqual(clf.ds.hash, self.hash)
            clf.destroy()


if __name__ == '__main__':
    unittest.main()
