from ml.utils.logger import log_config
from ml.models import SupervicedModel, MLModel
from ml import measures as metrics


log = log_config(__name__)


class RegModel(SupervicedModel):
    def __init__(self, **params):
        self.labels_dim = 1
        super(RegModel, self).__init__(**params)

    #def load(self, model_version="1"):
    #    self.model_version = model_version
    #    self.test_ds = self.get_dataset()
    #    self.get_train_validation_ds()
    #    self.load_model()

    #def scores(self, measures=None, chunks_size=2000):
    #    if measures is None or isinstance(measures, str):
    #        measures = metrics.Measure.make_metrics(measures="msle", name=self.model_name)
    #    with self.test_ds:
    #        test_data = self.test_ds.data[:]
    #        test_labels = self.test_ds.labels[:]
    #        length = test_labels.shape[0]

    #    for output in measures.outputs():
    #        predictions = self.predict(test_data, output=output,
    #            transform=False, chunks_size=chunks_size).to_memory(length)
    #        measures.set_data(predictions, test_labels, output=output)
    #    return measures.to_list()

    def convert_labels(self, labels, output=None):
        for chunk in labels:
            yield chunk

    def _predict(self, data, output=None):
        prediction = self.model.predict(data)
        return self.convert_labels(prediction, output=output)


class SKLP(RegModel):
    def __init__(self, *args, **kwargs):
        super(SKLP, self).__init__(*args, **kwargs)

    def ml_model(self, model):        
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit,
                        model=model,
                        predictors=model.predict,
                        load_fn=self.load_fn,
                        save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
        from sklearn.externals import joblib
        model = joblib.load('{}'.format(path))
        self.model = self.ml_model(model)


class LGB(RegModel):
    def ml_model(self, model, bst=None):
        self.bst = bst
        return MLModel(fit_fn=model.train, 
                            model=model,
                            predictors=self.bst.predict,
                            load_fn=self.load_fn,
                            save_fn=self.bst.save_model)

    def load_fn(self, path):
        import lightgbm as lgb
        bst = lgb.Booster(model_file=path)
        self.model = self.ml_model(lgb, bst=bst)

    def array2dmatrix(self, data):
        import lightgbm as lgb
        return lgb.Dataset(data)


class XGB(RegModel):
    def ml_model(self, model, bst=None):
        self.bst = bst
        return MLModel(fit_fn=model.train, 
                            predictors=self.bst.predict,
                            load_fn=self.load_fn,
                            save_fn=self.bst.save_model,
                            transform_data=self.array2dmatrix)

    def load_fn(self, path):
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(path)
        self.model = self.ml_model(xgb, bst=bst)

    def array2dmatrix(self, data):
        import xgboost as xgb
        return xgb.DMatrix(data)
