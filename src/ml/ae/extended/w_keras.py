from ml.ae.wrappers import Keras
from ml.models import MLModel
import tensorflow as tf
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras.layers import Lambda
from keras.losses import binary_crossentropy


class PTsne(Keras):
    def __init__(self, perplexity=30., epsilon_std=1.0, **kwargs):
        self.perplexity = perplexity
        self.epsilon_std = epsilon_std
        super(PTsne, self).__init__(**kwargs)

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
        self.model = self.default_model(model, self.load_fn)

    def train(self, batch_size=258, num_steps=50, num_epochs=50):
        from ml.utils.tf_functions import TSNe
        import numpy as np
        self.tsne = TSNe(batch_size=batch_size, perplexity=self.perplexity, dim=self.latent_dim)
        with self.train_ds:
            limit = int(round(self.train_ds.data.shape[0] * .9))
            X = self.train_ds.data[:limit]
            Z = self.train_ds.data[limit:]
        x = self.tsne.calculate_P(X)
        z = self.tsne.calculate_P(Z)
        self.prepare_model()
        self.model.fit(x,
            steps_per_epoch=num_steps,#batch_size,
            epochs=num_epochs,
            validation_data=z,
            nb_val_samples=num_steps)


def sampling(args):
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon


def vae_loss(num_features=None, z_log_var=None, z_mean=None):
    def vae_loss(x, outputs):
        reconstruction_loss = binary_crossentropy(x, outputs)# * num_features
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1) * -0.5
        return K.mean(reconstruction_loss + kl_loss)
    return vae_loss


class VAE(Keras):
    def __init__(self, intermediate_dim=5, epsilon_std=1.0,**kwargs):
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std
        super(VAE, self).__init__(**kwargs)

    def custom_objects(self):
        _, z_mean, z_log_var, _ = self.encoder()
        return {
            'sampling': sampling,
            'vae_loss': vae_loss(self.num_features, z_log_var, z_mean)
        }

    def encoder(self):
        x = Input(shape=(self.num_features,))
        h = Dense(self.intermediate_dim, activation='relu', name="intermedian_layer")(x)
        z_mean = Dense(self.latent_dim, name="z_mean")(h)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(h)
        z = Lambda(sampling)([z_mean, z_log_var])
        return x, z_mean, z_log_var, Model(x, z, name='encoder')
 
    def decoder(self):
        x = Input(shape=(self.latent_dim,), name='z_sampling')
        h = Dense(self.intermediate_dim, activation='relu')(x)
        outputs = Dense(self.num_features, activation='sigmoid')(x)
        decoder = Model(x, outputs, name='decoder')
        return decoder

    def intermedian_layer_model(self):
        model = self.model.model.get_layer('encoder')
        return self.default_model(model, self.load_fn)

    def prepare_model(self):
        x, z_mean, z_log_var, encoder = self.encoder()
        decoder = self.decoder()        

        outputs = decoder(encoder(x))
        model = Model(x, outputs, name='vae_mlp')
        model.compile(optimizer='adam', loss=vae_loss(self.num_features, z_log_var, z_mean))
        self.model = self.default_model(model, self.load_fn)

    def calculate_batch(self, X, batch_size=1):
        while 1:
            n = int(round(X.shape[0] / batch_size, 0))
            for i in range(0, n):
                yield (X[i:i + batch_size], X[i:i + batch_size])

    def train(self, batch_size=100, num_steps=50, num_epochs=50):
        with self.train_ds:
            limit = int(round(self.train_ds.data.shape[0] * .9))
            X = self.train_ds.data[:limit]
            Z = self.train_ds.data[limit:]
        batch_size_x = min(X.shape[0], batch_size)
        batch_size_z = min(Z.shape[0], batch_size)
        self.batch_size = min(batch_size_x, batch_size_z)
        self.prepare_model()
        x = self.calculate_batch(X, batch_size=self.batch_size)
        z = self.calculate_batch(Z, batch_size=self.batch_size)
        self.model.fit(x,
            steps_per_epoch=num_steps,
            epochs=num_epochs,
            validation_data=z,
            nb_val_samples=num_steps)
        
    def _metadata(self, keys={}):
        meta = super(VAE, self)._metadata(keys=keys)
        if "model" in meta:
            meta["model"]["intermediate_dim"] = self.intermediate_dim
        return meta

    def get_dataset(self):
        meta = self.load_meta()
        self.intermediate_dim = meta["model"]["intermediate_dim"]
        return super(VAE, self).get_dataset()


class SAE(Keras):
    def __init__(self, epsilon_std=1.0,**kwargs):
        self.epsilon_std = epsilon_std
        super(SAE, self).__init__(**kwargs)

    def custom_objects(self):
        return {}

    def encoder(self):
        input_ = Input(shape=(self.num_features,))
        encoder = Dense(self.latent_dim, activation='relu', 
                activity_regularizer=regularizers.l1(10e-5), name='intermedian_layer')(input_)

        return input_, encoder
 
    def decoder(self, encoded):
        decoder = Dense(self.num_features, activation='sigmoid')(encoded)
        return decoder

    def prepare_model(self):
        input_, encoded = self.encoder()
        decoded = self.decoder(encoded)

        model = Model(input_, decoded)
        model.compile(optimizer='sgd', loss='binary_crossentropy')
        self.model = self.default_model(model, self.load_fn)
        self.encoder_m = self.model

    def calculate_batch(self, X, batch_size=1):
        while 1:
            n = int(round(X.shape[0] / batch_size, 0))
            for i in range(0, n):
                yield (X[i:i + batch_size], X[i:i + batch_size])

    def train(self, batch_size=100, num_steps=50, num_epochs=50):
        with self.train_ds:
            limit = int(round(self.train_ds.data.shape[0] * .9))
            X = self.train_ds.data[:limit]
            Z = self.train_ds.data[limit:]
        batch_size_x = min(X.shape[0], batch_size)
        batch_size_z = min(Z.shape[0], batch_size)
        self.batch_size = min(batch_size_x, batch_size_z)
        self.prepare_model()
        x = self.calculate_batch(X, batch_size=self.batch_size)
        z = self.calculate_batch(Z, batch_size=self.batch_size)
        self.model.fit(x,
            steps_per_epoch=num_steps,
            epochs=num_epochs,
            validation_data=z,
            nb_val_samples=num_steps)
        
    def _metadata(self, keys={}):
        meta = super(SAE, self)._metadata(keys=keys)
        if "model" in meta:
            meta["model"]["latent_dim"] = self.latent_dim
        return meta

    def get_dataset(self):
        meta = self.load_meta()
        self.latent_dim = meta["model"]["latent_dim"]
        return super(SAE, self).get_dataset()
