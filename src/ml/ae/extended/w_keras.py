from ml.ae.wrappers import UnsupervisedModel
from keras import regularizers
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras import backend as K
from keras.layers import Lambda
from keras.losses import binary_crossentropy
from ml.utils.tf_functions import KLdivergence
from ml.models import MLModel
from ml.data.it import Iterator
from keras.models import Sequential


def clean_iter(it):
    for groups in it:
        row = []
        for group in groups:
            array = group.to_ndarray()
            if array.shape[1] > array.shape[0]:
                array = array[:, :array.shape[0]]
            row.append(array)
        yield row


class PTsne(UnsupervisedModel):
    def __init__(self, perplexity=30., epsilon_std=1.0, **kwargs):
        self.perplexity = perplexity
        self.epsilon_std = epsilon_std
        super(PTsne, self).__init__(**kwargs)

    def custom_objects(self):
        return {'KLdivergence': KLdivergence}

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path, custom_objects=self.custom_objects())
        self.model = self.ml_model(model)

    def ml_model(self, model) -> MLModel:
        return MLModel(fit_fn=model.fit_generator,
                       predictors=model.predict,
                       load_fn=self.load_fn,
                       save_fn=model.save,
                       to_json_fn=model.to_json)

    def prepare_model(self, obj_fn=None, num_steps: int = 0, model_params=None, batch_size: int = None) -> MLModel:
        with self.ds:
            input_shape = self.ds[self.data_groups["data_train_group"]].shape.to_tuple()

        model = Sequential()
        model.add(Dense(500, input_shape=input_shape[1:]))
        model.add(Activation('relu'))
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(2000))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.compile(optimizer='sgd', loss=KLdivergence)
        with self.ds:
            x_stc = self.ds[[self.data_groups["data_train_group"], self.data_groups["target_train_group"]]]
            z_stc = self.ds[[self.data_groups["data_validation_group"], self.data_groups["target_validation_group"]]]
            x_it = Iterator(x_stc).batchs(batch_size=batch_size, batch_type="structured").cycle().to_iter()
            z_it = Iterator(z_stc).batchs(batch_size=batch_size, batch_type="structured").cycle().to_iter()
            x_iter = clean_iter(x_it)
            z_iter = clean_iter(z_it)

        steps = round(len(x_stc)/batch_size/num_steps, 0)
        vsteps = round(len(z_stc)/batch_size/num_steps, 0)
        steps = 1 if steps == 0 else steps
        vsteps = 1 if vsteps == 0 else vsteps
        model.fit_generator(x_iter, steps_per_epoch=steps, epochs=num_steps, validation_data=z_iter,
                            validation_steps=vsteps, max_queue_size=1)
        return self.ml_model(model)



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


#class VAE(KerasAe):
#    def __init__(self, intermediate_dim=5, epsilon_std=1.0,**kwargs):
#        self.intermediate_dim = intermediate_dim
#        self.epsilon_std = epsilon_std
#        super(VAE, self).__init__(**kwargs)

#    def custom_objects(self):
#        _, z_mean, z_log_var, _ = self.encoder()
#        return {
#            'sampling': sampling,
#            'vae_loss': vae_loss(self.num_features, z_log_var, z_mean)
#        }

#    def encoder(self):
#        x = Input(shape=(self.num_features,))
#        h = Dense(self.intermediate_dim, activation='relu', name="intermedian_layer")(x)
#        z_mean = Dense(self.latent_dim, name="z_mean")(h)
#        z_log_var = Dense(self.latent_dim, name="z_log_var")(h)
#        z = Lambda(sampling)([z_mean, z_log_var])
#        return x, z_mean, z_log_var, Model(x, z, name='encoder')
 
#    def decoder(self):
#        x = Input(shape=(self.latent_dim,), name='z_sampling')
#       h = Dense(self.intermediate_dim, activation='relu')(x)
#        outputs = Dense(self.num_features, activation='sigmoid')(x)
#        decoder = Model(x, outputs, name='decoder')
#        return decoder

#    def intermedian_layer_model(self):
#        model = self.model.model.get_layer('encoder')
#        return self.default_model(model, self.load_fn)

#    def prepare_model(self):
#        x, z_mean, z_log_var, encoder = self.encoder()
#        decoder = self.decoder()

#        outputs = decoder(encoder(x))
#        model = Model(x, outputs, name='vae_mlp')
#        model.compile(optimizer='adamax', loss=vae_loss(self.num_features, z_log_var, z_mean))
#        self.model = self.default_model(model, self.load_fn)

#    def calculate_batch(self, X, batch_size=1):
#        while 1:
#            n = int(round(X.shape[0] / batch_size, 0))
#            for i in range(0, n):
#                yield (X[i:i + batch_size], X[i:i + batch_size])

#    def train(self, batch_size=100, num_steps=50, num_epochs=50):
#       with self.train_ds:
#            limit = int(round(self.train_ds.data.shape[0] * .9))
#            X = self.train_ds.data[:limit]
#            Z = self.train_ds.data[limit:]
#        batch_size_x = min(X.shape[0], batch_size)
#        batch_size_z = min(Z.shape[0], batch_size)
#        self.batch_size = min(batch_size_x, batch_size_z)
#       self.prepare_model()
#        x = self.calculate_batch(X, batch_size=self.batch_size)
#        z = self.calculate_batch(Z, batch_size=self.batch_size)
#        self.model.fit(x,
#            steps_per_epoch=num_steps,
#            epochs=num_epochs,
#            validation_data=z,
#            nb_val_samples=num_steps)
        
#    def _metadata(self, keys={}):
#        meta = super(VAE, self)._metadata(keys=keys)
#        if "model" in meta:
#            meta["model"]["intermediate_dim"] = self.intermediate_dim
#        return meta

#    def get_dataset(self):
#        meta = self.load_meta()
#        self.intermediate_dim = meta["model"]["intermediate_dim"]
#        return super(VAE, self).get_dataset()


#class SAE(KerasAe):
#    def __init__(self, epsilon_std=1.0,**kwargs):
#        self.epsilon_std = epsilon_std
#        super(SAE, self).__init__(**kwargs)

#    def custom_objects(self):
#        return {}

#    def encoder(self):
#        input_ = Input(shape=(self.num_features,))
#        encoder = Dense(self.latent_dim, activation='relu',
#                activity_regularizer=regularizers.l1(10e-5), name='intermedian_layer')(input_)

#        return input_, encoder
 
#    def decoder(self, encoded):
#        decoder = Dense(self.num_features, activation='sigmoid')(encoded)
#        return decoder

#    def prepare_model(self):
#        input_, encoded = self.encoder()
#        decoded = self.decoder(encoded)

#        model = Model(input_, decoded)
#        model.compile(optimizer='sgd', loss='binary_crossentropy')
#        self.model = self.default_model(model, self.load_fn)
#        self.encoder_m = self.model

#    def calculate_batch(self, X, batch_size=1):
#        while 1:
#            n = int(round(X.shape[0] / batch_size, 0))
#            for i in range(0, n):
#                yield (X[i:i + batch_size], X[i:i + batch_size])

#    def train(self, batch_size=100, num_steps=50, num_epochs=50):
#        with self.train_ds:
#            limit = int(round(self.train_ds.data.shape[0] * .9))
#            X = self.train_ds.data[:limit]
#            Z = self.train_ds.data[limit:]
#        batch_size_x = min(X.shape[0], batch_size)
#        batch_size_z = min(Z.shape[0], batch_size)
#        self.batch_size = min(batch_size_x, batch_size_z)
#        self.prepare_model()
#        x = self.calculate_batch(X, batch_size=self.batch_size)
#        z = self.calculate_batch(Z, batch_size=self.batch_size)
#        self.model.fit(x,
#            steps_per_epoch=num_steps,
#            epochs=num_epochs,
#            validation_data=z,
#            nb_val_samples=num_steps)
        
#    def _metadata(self, keys={}):
#        meta = super(SAE, self)._metadata(keys=keys)
#        if "model" in meta:
#            meta["model"]["latent_dim"] = self.latent_dim
#        return meta

#    def get_dataset(self):
#        meta = self.load_meta()
#        self.latent_dim = meta["model"]["latent_dim"]
#        return super(SAE, self).get_dataset()
