import tflearn
import tensorflow as tf


class BasicModel:
    def __init__(self):
        pass

    def fit(self):
        pass


class MMLP(tflearn.DNN):
    def __init__(self, num_features, layers, num_labels):
        self.num_features = num_features
        self.layers = layers
        self.num_labels = num_labels
        self.net = self.build()
        super(MMLP, self).__init__(self.net, tensorboard_verbose=3)

    def build(self):
        input_layer = tflearn.input_data(shape=[None, self.num_features])
        layer_ = input_layer
        for layer_size in self.layers:
            dense = tflearn.fully_connected(layer_, layer_size, activation='tanh',
                                             regularizer='L2', weight_decay=0.001)
            dropout = tflearn.dropout(dense, 0.5)
            layer_ = dropout

        softmax = tflearn.fully_connected(layer_, self.num_labels, activation='softmax')
        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        acc = tflearn.metrics.Accuracy()
        return tflearn.regression(softmax, optimizer=sgd, metric=acc,
                         loss='categorical_crossentropy')

