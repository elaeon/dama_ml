from dama.clf.wrappers import Keras
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dropout
import numpy as np


class FCNet(Keras):
    @staticmethod
    def int2vector(target):
        num_classes = 2
        return (np.arange(num_classes) == target[:, None]).astype(np.float)

    def prepare_model(self, obj_fn=None, num_steps: int = 0, model_params=None, batch_size: int = None):
        layers = [128, 64]
        model = Sequential()
        input_shape = self.ds[self.data_groups["data_train_group"]].shape.to_tuple()
        model.add(Dense(layers[0], input_shape=input_shape[1:]))
        for layer_size in layers[1:]:
            model.add(Dense(layer_size, activation='tanh',
                activity_regularizer=regularizers.l2(l=10e-5)))
            model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                  FCNet.int2vector(self.ds[self.data_groups["target_train_group"]].to_ndarray()),
                  epochs=num_steps,
                  batch_size=batch_size,
                  shuffle="batch",
                  validation_data=(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                                   FCNet.int2vector(self.ds[self.data_groups["target_validation_group"]].to_ndarray())))
        return self.ml_model(model)


