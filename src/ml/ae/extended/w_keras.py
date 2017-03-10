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


def sampling(args):
    from keras import backend as K
    z_mean, z_log_var = args
    batch_size = 1
    latent_dim = 2
    epsilon_std = 1.0
    epsilon = K.random_normal(shape=(batch_size, latent_dim), 
        mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def vae_loss(num_features=None, z_log_var=None, z_mean=None):
    from keras import objectives
    from keras import backend as K

    def vae_loss(x, x_decoded_mean):
        xent_loss = num_features * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #print(xent_loss.get_shape(), kl_loss.get_shape(), z_log_var.get_shape(), z_mean.get_shape())
        return xent_loss + kl_loss 

    return vae_loss


class VAE(Keras):
    def __init__(self, *args, **kwargs):
        if 'intermediate_dim' in kwargs:            
            self.intermediate_dim = kwargs['intermediate_dim']
            del kwargs['intermediate_dim']
        self.epsilon_std = 1.0
        super(VAE, self).__init__(*args, **kwargs)

    def custom_objects(self):
        x, z_mean, z_log_var = self.encoder()
        return {'sampling': sampling, 
            'vae_loss': vae_loss(num_features=self.num_features, z_log_var=z_log_var, z_mean=z_mean)}

    def encoder(self):
        from keras.layers import Input, Dense
        #x = Input(batch_shape=(self.batch_size, self.num_features))
        x = Input(shape=(self.num_features,))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        return x, z_mean, z_log_var
 
    def decoder(self, z):
        from keras.layers import Dense
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.num_features, activation='sigmoid')

        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        return x_decoded_mean

    def prepare_model(self):
        from keras.layers import Lambda
        from keras.models import Model

        x, z_mean, z_log_var = self.encoder()
        z = Lambda(sampling)([z_mean, z_log_var])
        x_decoded_mean = self.decoder(z)        

        model = Model(x, x_decoded_mean)
        model.compile(optimizer='rmsprop', 
            loss=vae_loss(num_features=self.num_features, z_log_var=z_log_var, z_mean=z_mean))
        self.model = self.default_model(model)

    def calculate_batch(self, X, batch_size=1):
        print("Computing batches...")
        while 1:
            n = X.shape[0]
            for i in xrange(0, n, batch_size):
                yield (X[i:i + batch_size], X[i:i + batch_size])

    def train(self, batch_size=100, num_steps=50):
        limit = int(round(self.dataset.data.shape[0] * .9))
        X = self.dataset.data[:limit]
        Z = self.dataset.data[limit:]
        batch_size_x = min(X.shape[0], batch_size)
        batch_size_z = min(Z.shape[0], batch_size)
        self.batch_size = min(batch_size_x, batch_size_z)
        self.prepare_model()
        x = self.calculate_batch(X, batch_size=self.batch_size)
        z = self.calculate_batch(Z, batch_size=self.batch_size)
        self.model.fit(x,
            self.batch_size,
            num_steps,
            validation_data=z,
            nb_val_samples=batch_size_z)
        self.save_model()
        
    def _metadata(self):
        meta = super(VAE, self)._metadata()
        meta["intermediate_dim"] = self.intermediate_dim
        return meta

    def get_dataset(self):
        meta = self.load_meta()
        self.intermediate_dim = meta["intermediate_dim"]
        return super(VAE, self).get_dataset()
