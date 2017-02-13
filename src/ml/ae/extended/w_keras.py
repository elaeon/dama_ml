from ml.ae.wrappers import Keras
from ml.models import MLModel


class PTsne(Keras):
    def prepare_model(self):
        from keras.layers import Input, Dense, Activation
        from keras.models import Model
        from keras import regularizers
        from keras.layers import Dropout
        from keras.models import Sequential

        model = Sequential()
        model.add(Dense(500, input_shape=(self.num_features,)))
        model.add(Activation('relu'))
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(2000))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.compile(optimizer='sgd', loss=self.tsne.KLdivergence)
        self.model = MLModel(fit_fn=model.fit, 
                            predictors=[model.predict],
                            load_fn=self.load_fn,
                            save_fn=model.save)

    def train(self, batch_size=258, num_steps=50):
        from ml.utils.tf_functions import TSNe
        from ml.utils.numeric_functions import expand_matrix_row
        import numpy as np
        self.tsne = TSNe(batch_size=batch_size, perplexity=30.)
        limit = int(round(self.dataset.data.shape[0] * .9))
        diff = limit % batch_size
        diff2 = (limit - self.dataset.data.shape[0]) % batch_size
        X = expand_matrix_row(self.dataset.data[:limit], batch_size, diff)
        Z = expand_matrix_row(self.dataset.data[limit:], batch_size, diff2)
        x = self.tsne.calculate_P(X)
        z = self.tsne.calculate_P(Z)
        self.prepare_model()
        self.model.fit(X, 
            x,
            nb_epoch=num_steps,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(Z, z))
        self.save_model()
