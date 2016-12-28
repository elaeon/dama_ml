from ml.clf.wrappers import Keras
from ml.models import MLModel


class FCNet(Keras):
    def prepare_model(self):
        from keras.layers import Dense
        from keras.models import Sequential
        from keras import regularizers
        from keras.layers import Dropout

        self.layers = [128, 64]
        model = Sequential()
        model.add(Dense(self.layers[0], input_shape=(self.num_features,)))
        for layer_size in self.layers[1:]:
            model.add(Dense(layer_size, activation='tanh', 
                activity_regularizer=regularizers.activity_l2(10e-5)))
            model.add(Dropout(0.5))

        model.add(Dense(self.num_labels, activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        return model

