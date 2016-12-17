import tensorflow as tf

class MLModel:
    def __init__(self, fit_fn=None, predictors=None, load_fn=None, save_fn=None):
        self.fit_fn = fit_fn
        self.predictors = predictors
        self.load_fn = load_fn
        self.save_fn = save_fn

    def fit(self, *args, **kwargs):
        return self.fit_fn(*args, **kwargs)

    def predict(self, data):
        prediction = self.predictors[0](data)
        for predictor in self.predictors[1:]:
            prediction = predictor(prediction)
        return prediction

    def load(self, path):
        return self.load_fn(path)

    def save(self, path):
        return self.save_fn(path)

