from ml.clf.wrappers import Keras
from ml.models import MLModel
from keras.models import load_model


class FCNet(Keras):
    def load_fn(self, path):
        self.net_model = load_model(path)
        self.model = MLModel(fit_fn=self.net_model.fit, 
                            predictors=[self.net_model.predict],
                            load_fn=self.load_fn,
                            save_fn=self.save_fn)

    def save_fn(self, path):
        self.net_model.save(path)

    def preload_model(self):
        self.model = MLModel(fit_fn=None, 
                            predictors=None,
                            load_fn=self.load_fn,
                            save_fn=self.save_fn)

    def prepare_model(self):
        from keras.layers import Input, Dense
        from keras.models import Model
        from keras import regularizers
        from keras.layers import Dropout

        self.layers = [128, 64]
        input_layer = Input(shape=(self.num_features,))
        layer_ = input_layer
        for layer_size in self.layers:
            dense = Dense(layer_size, activation='tanh', 
                activity_regularizer=regularizers.activity_l2(10e-5))(input_layer)
            layer_ = Dropout(0.5)(dense)

        softmax = Dense(self.num_labels, activation='softmax')(layer_)
        self.net_model = Model(input=input_layer, output=softmax)
        self.net_model.compile(optimizer='sgd', loss='categorical_crossentropy')
        self.model = MLModel(fit_fn=self.net_model.fit, 
                            predictors=[self.net_model.predict],
                            load_fn=self.load_fn,
                            save_fn=self.save_fn)

    def train(self, batch_size=258, num_steps=50):
        self.prepare_model()
        self.model.fit(self.dataset.train_data, 
            self.dataset.train_labels,
            nb_epoch=num_steps,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(self.dataset.valid_data, self.dataset.valid_labels))
        self.save_model()
