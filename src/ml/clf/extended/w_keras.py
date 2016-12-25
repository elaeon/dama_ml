from ml.clf.wrappers import Keras
from ml.models import MLModel


class FCNet(Keras):
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
        net_model = Model(input=input_layer, output=softmax)
        net_model.compile(optimizer='sgd', loss='categorical_crossentropy')
        self.model = MLModel(fit_fn=net_model.fit, 
                            predictors=[net_model.predict],
                            load_fn=self.load_fn,
                            save_fn=net_model.save)

    def train(self, batch_size=258, num_steps=50):
        self.prepare_model()
        self.model.fit(self.dataset.train_data, 
            self.dataset.train_labels,
            nb_epoch=num_steps,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(self.dataset.valid_data, self.dataset.valid_labels))
        self.save_model()


class easy(Keras):
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
        net_model = model
        net_model.compile(optimizer='sgd', loss=self.tsne.KLdivergence)
        self.model = MLModel(fit_fn=net_model.fit, 
                            predictors=[net_model.predict],
                            load_fn=self.load_fn,
                            save_fn=net_model.save)

    def train(self, batch_size=258, num_steps=50):
        from ml.utils.tf_functions import TSNe
        from ml.utils.numeric_functions import expand_matrix_row
        import numpy as np
        self.tsne = TSNe(batch_size=batch_size)
        diff = self.dataset.train_data.shape[0] % batch_size
        #y = expand_matrix_row(self.dataset.train_labels, batch_size, diff)
        #y = self.tsne.calculate_P(y)
        X = expand_matrix_row(self.dataset.train_data, batch_size, diff)
        y = self.tsne.calculate_P(X)
        #self.prepare_model()
        #self.model.fit(X, 
        #    y,
        #    nb_epoch=num_steps,
        #    batch_size=batch_size,
        #    shuffle=False)
            #validation_data=(self.dataset.valid_data, v))
        #self.save_model()
