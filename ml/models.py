import tflearn
import tensorflow as tf


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
            layer_ = tflearn.dropout(dense, 0.5)

        softmax = tflearn.fully_connected(layer_, self.num_labels, activation='softmax')
        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        acc = tflearn.metrics.Accuracy()
        return tflearn.regression(softmax, optimizer=sgd, metric=acc,
                         loss='categorical_crossentropy')


class MConvNet(tflearn.DNN):
    def __init__(self, num_features, layers, num_labels):
        self.net = self.build()
        super(MMLP, self).__init__(self.net, tensorboard_verbose=3)

    def build(self):
        import tflearn
        network = tflearn.input_data(
            shape=[None, self.dataset.image_size, self.dataset.image_size, self.num_channels],
            name='input')
        network = tflearn.conv_2d(network, self.depth, self.patch_size, 
            activation='relu', regularizer="L2")
        network = tflearn.max_pool_2d(network, 2)
        network = tflearn.local_response_normalization(network)
        network = tflearn.conv_2d(network, self.depth*2, self.patch_size, 
            activation='relu', regularizer="L2")
        network = tflearn.max_pool_2d(network, 2)
        network = tflearn.local_response_normalization(network)
        network = tflearn.fully_connected(network, self.depth*4, activation='tanh')
        network = tflearn.dropout(network, 0.8)
        network = tflearn.fully_connected(network, self.depth*8, activation='tanh')
        network = tflearn.dropout(network, 0.8)
        network = tflearn.fully_connected(network, self.num_labels, activation='softmax')
        #top_k = tflearn.metrics.Top_k(5)
        acc = tflearn.metrics.Accuracy()
        return tflearn.regression(network, optimizer='adam', metric=acc, learning_rate=0.01,
                                 loss='categorical_crossentropy', name='target')
