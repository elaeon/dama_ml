from dama.ae.wrappers import UnsupervisedModel
from keras.layers import Dense, Activation
from keras import backend as K
from keras.losses import binary_crossentropy
from dama.utils.tf_functions import KLdivergence
from dama.models import MLModel
from dama.data.it import Iterator
from keras.models import Sequential


def clean_iter(it):
    for slice_obj in it:
        row = []
        batch = slice_obj.batch
        for group in batch.groups:
            row.append(batch[group].to_ndarray())
        yield row


class PTsne(UnsupervisedModel):
    def __init__(self, perplexity=30., epsilon_std=1.0, **kwargs):
        self.perplexity = perplexity
        self.epsilon_std = epsilon_std
        super(PTsne, self).__init__(**kwargs)

    @staticmethod
    def custom_objects():
        return {'KLdivergence': KLdivergence}

    def load_fn(self, path):
        from keras.models import load_model
        model = load_model(path, custom_objects=PTsne.custom_objects())
        self.model = self.ml_model(model)

    def ml_model(self, model) -> MLModel:
        return MLModel(fit_fn=model.fit_generator,
                       predictors=model.predict,
                       load_fn=self.load_fn,
                       save_fn=model.save,
                       to_json_fn=model.to_json)

    def prepare_model(self, obj_fn=None, num_steps: int = 0, model_params=None, batch_size: int = None) -> MLModel:
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
        x_stc = self.ds[[self.data_groups["data_train_group"], self.data_groups["target_train_group"]]]
        z_stc = self.ds[[self.data_groups["data_validation_group"], self.data_groups["target_validation_group"]]]
        x_it = Iterator(x_stc).batchs(chunks=(batch_size, )).cycle()
        z_it = Iterator(z_stc).batchs(chunks=(batch_size, )).cycle()
        x_iter = clean_iter(x_it)
        z_iter = clean_iter(z_it)
        steps = round(x_stc.size/batch_size/num_steps, 0)
        vsteps = round(z_stc.size/batch_size/num_steps, 0)
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
