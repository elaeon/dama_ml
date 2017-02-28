from ml.ae.wrappers import Keras
from ml.models import MLModel
import tensorflow as tf

class PTsne(Keras):
    
    def custom_objects(self):
        from ml.utils.tf_functions import KLdivergence
        return {'KLdivergence': KLdivergence}

    def prepare_model(self):
        from keras.layers import Dense, Activation
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
        self.tsne = TSNe(batch_size=batch_size, perplexity=30., dim=self.latent_dim)
        limit = int(round(self.dataset.data.shape[0] * .9))
        #diff = limit % batch_size
        #diff2 = (limit - self.dataset.data.shape[0]) % batch_size
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


class VAE(Keras):
    def __init__(self, *args, **kwargs):
        if 'intermediate_dim' in kwargs:            
            self.intermediate_dim = kwargs['intermediate_dim']
            del kwargs['intermediate_dim']
        self.epsilon_std = 1.0
        super(VAE, self).__init__(*args, **kwargs)

    def custom_objects(self):
        return {'vae_loss': self.vae_loss, 'sampling': self.sampling}

    def sampling(self, args):
        from keras import backend as K
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                              std=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def vae_loss(self, x, x_decoded_mean):
        from keras import objectives
        from keras import backend as K

        xent_loss = self.num_features * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss

    def prepare_model(self):
        from keras.layers import Input, Dense, Lambda
        from keras.models import Model

        x = Input(batch_shape=(self.batch_size, self.num_features))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)

        z = Lambda(self.sampling)([self.z_mean, self.z_log_var])

        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.num_features, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        model = Model(x, x_decoded_mean)
        model.compile(optimizer='rmsprop', loss=self.vae_loss)
        self.model = self.default_model(model)

    def calculate_batch(self, X):
        print("Computing batches...")
        while 1:
            n = X.shape[0]
            for i in xrange(0, n, self.batch_size):
                yield (X[i:i + self.batch_size], X[i:i + self.batch_size])

    def train(self, batch_size=100, num_steps=50):
        self.batch_size = batch_size
        self.prepare_model()
        limit = int(round(self.dataset.data.shape[0] * .9))
        x = self.calculate_batch(self.dataset.data[:limit])
        z = self.calculate_batch(self.dataset.data[limit:])
        self.model.fit(x,
            batch_size,
            num_steps,
            validation_data=z,
            nb_val_samples=batch_size)
        self.save_model()
        
