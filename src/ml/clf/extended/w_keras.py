from ml.clf.wrappers import Keras
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dropout
import numpy as np


class FCNet(Keras):
    def int2vector(self, target):
        num_classes = 2
        return (np.arange(num_classes) == target[:, None]).astype(np.float)

    def prepare_model(self, obj_fn=None, num_steps: int = 0, model_params=None, batch_size=None):
        self.layers = [128, 64]
        model = Sequential()
        with self.ds:
            input_shape = self.ds[self.data_groups["data_train_group"]].shape.to_tuple()
        model.add(Dense(self.layers[0], input_shape=input_shape[1:]))
        for layer_size in self.layers[1:]:
            model.add(Dense(layer_size, activation='tanh', 
                activity_regularizer=regularizers.l2(l=10e-5)))
            model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        with self.ds:
            model.fit(self.ds[self.data_groups["data_train_group"]].to_ndarray(),
                      self.int2vector(self.ds[self.data_groups["target_train_group"]].to_ndarray()),
                      epochs=num_steps,
                      batch_size=self.batch_size,
                      shuffle="batch",
                      validation_data=(self.ds[self.data_groups["data_validation_group"]].to_ndarray(),
                                       self.int2vector(self.ds[self.data_groups["target_validation_group"]].to_ndarray())))
        return self.ml_model(model)


