from sklearn.externals import joblib
from ml.utils.logger import log_config
from ml.models import SupervicedModel, MLModel
from ml import measures as metrics
from ml.data.it import Iterator


log = log_config(__name__)


class RegModel(SupervicedModel):
    def __init__(self, **params):
        self.labels_dim = 1
        super(RegModel, self).__init__(**params)

    def scores(self, measures="msle", batch_size: int=2000):
        if measures is None or isinstance(measures, str):
            measure = metrics.MeasureBatch(name=self.model_name, batch_size=batch_size)
            measures = measure.make_metrics(measures=measures, discrete=False)
        with self.ds:
            test_data = self.ds[self.data_groups["data_test_group"]]
            for measure_fn in measures:
                test_target = Iterator(self.ds[self.data_groups["target_test_group"]]).batchs(batch_size=batch_size)
                predictions = self.predict(test_data, output=measure_fn.output, batch_size=batch_size)
                for pred, target in zip(predictions, test_target):
                    measures.update_fn(pred, target, measure_fn)
        return measures.to_list()

    def output_format(self, prediction, output=None):
        if output == 'uncertain' or output == 'n_dim':
            return prediction
        else:
            return prediction


class SKLP(RegModel):

    def ml_model(self, model):        
        from sklearn.externals import joblib
        return MLModel(fit_fn=model.fit,
                        model=model,
                        predictors=model.predict,
                        load_fn=self.load_fn,
                        save_fn=lambda path: joblib.dump(model, '{}'.format(path)))

    def load_fn(self, path):
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


class XGB(RegModel):
    def ml_model(self, model, bst=None):
        self.bst = bst
        return MLModel(fit_fn=model.train, 
                            predictors=self.bst.predict,
                            load_fn=self.load_fn,
                            save_fn=self.bst.save_model,
                            input_transform=self.array2dmatrix)

    def load_fn(self, path):
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(path)
        self.model = self.ml_model(xgb, bst=bst)

    def array2dmatrix(self, data):
        import xgboost as xgb
        return xgb.DMatrix(data.to_ndarray())
