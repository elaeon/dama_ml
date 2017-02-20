from ml.ae.wrappers import Keras
from ml.models import MLModel
import tensorflow as tf

class PTsne(Keras):
    def __init__(self, *args, **kwargs):
        if 'dim' in kwargs:
            self.dim = kwargs['dim']
            del kwargs['dim']
        super(PTsne, self).__init__(*args, **kwargs)

    def prepare_model(self):
        from keras.layers import Dense, Activation
        #from keras.layers import Dropout
        from keras.models import Sequential

        from ml.utils.tf_functions import KLdivergence

        model = Sequential()
        model.add(Dense(500, input_shape=(self.num_features,)))
        model.add(Activation('relu'))
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(2000))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.compile(optimizer='sgd', loss=KLdivergence)
        self.model = self.default_model(model)

    def train(self, batch_size=258, num_steps=50):
        from ml.utils.tf_functions import TSNe
        import numpy as np
        self.tsne = TSNe(batch_size=batch_size, perplexity=30., dim=self.dim)
        limit = int(round(self.dataset.data.shape[0] * .9))
        diff = limit % batch_size
        diff2 = (limit - self.dataset.data.shape[0]) % batch_size
        X = self.dataset.data[:limit]
        Z = self.dataset.data[limit:]
        x = self.tsne.calculate_P(X)
        z = self.tsne.calculate_P(Z)
        self.prepare_model()
        self.model.fit(x,
            batch_size,
            num_steps,
            validation_data=z,
            nb_val_samples=batch_size)
        self.save_model()
