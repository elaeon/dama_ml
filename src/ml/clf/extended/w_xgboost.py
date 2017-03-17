from ml.clf.wrappers import BaseClassif
from ml.models import MLModel

import xgboost as xgb
import os


class Xgboost(BaseClassif):
    def __init__(self, libsvm_path=None, params={}, num_round=2, **kwargs):
        self.libsvm_path = libsvm_path
        self.params = params
        self.num_round = num_round
        super(Xgboost, self).__init__(**kwargs)

    def reformat_all(self, dataset):
        dsb = super(Xgboost, self).reformat_all(dataset)
        dsb.to_libsvm(name=self.model_name, save_to=self.libsvm_path, validation=True)
        return dsb

    def _metadata(self):
        meta = super(Xgboost, self)._metadata()
        meta["libsvm_path"] = self.libsvm_path
        return meta

    def get_dataset(self):
        from ml.ds import DataSetBuilder
        meta = self.load_meta()
        dataset = DataSetBuilder(meta["dataset_name"], dataset_path=meta["dataset_path"],
            apply_transforms=False)
        self._original_dataset_md5 = meta["md5"]
        self.labels_encode(meta["base_labels"])
        self.group_name = meta.get('group_name', None)
        self.libsvm_path = meta.get('libsvm_path', None)
        if meta.get('md5', None) != dataset.md5():
            log.warning("The dataset md5 is not equal to the model '{}'".format(
                self.__class__.__name__))
        return dataset

    def libsvm_files(self):
        files = ["train", "test", "validation"]
        return [os.path.join(self.libsvm_path, "{}.{}.txt".format(self.model_name, filename))
                for filename in files]

    def prepare_model(self):
        files = self.libsvm_files()
        dtrain = xgb.DMatrix(files[0])
        dtest = xgb.DMatrix(files[1])
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        model = xgb.train(self.params, dtrain, self.num_round, watchlist)
        return model

    def train(self, batch_size=0, num_steps=0):
        model = self.prepare_model()
        if not isinstance(model, MLModel):
            self.model = MLModel(fit_fn=xgb.train, 
                                predictors=[model.predict],
                                load_fn=self.load_fn,
                                save_fn=model.save_model)
        else:
            self.model = model
        self.save_model()

    def load_fn(self):
        model = xgb.Booster(model_file=self.get_model_path())
        self.model = MLModel(fit_fn=xgb.train, 
                            predictors=[model.predict],
                            load_fn=self.load_fn,
                            save_fn=model.save_model)

    def _predict(self, data, raw=False):
        data = xgb.DMatrix(data, label=None)
        for prediction in self.model.predict(data):
            if not isinstance(prediction, np.ndarray):
                prediction = np.asarray(prediction)
            yield self.convert_label(prediction, raw=raw)
