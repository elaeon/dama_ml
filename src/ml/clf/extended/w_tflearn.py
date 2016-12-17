from ml.clf.wrappers import TFL

import tensorflow as tf
import tflearn


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


class MLP(TFL):
    def __init__(self, *args, **kwargs):
        if "layers" in kwargs:
            self.layers = kwargs["layers"]
            del kwargs["layers"]
        else:
            self.layers = [128, 64]
        super(MLP, self).__init__(*args, **kwargs)

    def prepare_model(self):
        self.model = MMLP(self.num_features, self.layers, self.num_labels)

    def train(self, batch_size=10, num_steps=1000):
        with tf.Graph().as_default():
            self.prepare_model()
            self.model.fit(self.dataset.train_data, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.valid_data, self.dataset.valid_labels),
                show_metric=True, 
                batch_size=batch_size,
                run_id="mlp_model")
            self.save_model()


class ConvNet(TFL):
    def __init__(self, *args, **kwargs):
        self.num_channels = kwargs.get("num_channels", 1)
        self.patch_size = 3
        self.depth = 32
        if "num_channels" in kwargs:
            del kwargs["num_channels"]
        super(ConvTensor, self).__init__(*args, **kwargs)

    def transform_shape(self, img):
        return img.reshape((-1, self.dataset.image_size, self.dataset.image_size,
            self.num_channels)).astype(np.float32)

    def prepare_model(self):
        from ml.models import MConvNet
        self.model = MConvNet()

    def train(self, batch_size=10, num_steps=1000):
        with tf.Graph().as_default():
            self.prepare_model()
            self.model.fit(self.dataset.train_data, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.valid_data, self.dataset.valid_labels),
                show_metric=True, 
                batch_size=batch_size,
                snapshot_step=100,
                run_id="conv_model")
            self.save_model()


class ResidualTensor(TFL):
    def __init__(self, *args, **kwargs):
        self.num_channels = kwargs.get("num_channels", 1)
        self.patch_size = 3
        self.depth = 32
        if "num_channels" in kwargs:
            del kwargs["num_channels"]
        super(ResidualTensor, self).__init__(*args, **kwargs)

    def transform_shape(self, img):
        return img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)

    def reformat_all(self):
        import tflearn.data_utils as du
        all_ds = np.concatenate((self.train_labels, self.valid_labels, self.test_labels), axis=0)
        self.labels_encode(all_ds)
        self.train_data, self.train_labels = self.reformat(
            self.train_data, self.le.transform(self.train_labels))
        self.valid_data, self.valid_labels = self.reformat(
            self.valid_data, self.le.transform(self.valid_labels))
        self.test_data, self.test_labels = self.reformat(
            self.test_data, self.le.transform(self.test_labels))

        self.train_data, mean = du.featurewise_zero_center(self.train_data)
        self.test_data = du.featurewise_zero_center(self.test_data, mean)

        print('RF-Training set', self.train_data.shape, self.train_labels.shape)
        print('RF-Validation set', self.valid_data.shape, self.valid_labels.shape)
        print('RF-Test set', self.test_data.shape, self.test_labels.shape)

    def prepare_model(self, dropout=False):
        import tflearn

        net = tflearn.input_data(shape=[None, self.image_size, self.image_size, self.num_channels])
        net = tflearn.conv_2d(net, self.depth, self.patch_size, activation='relu', bias=False)
        net = tflearn.batch_normalization(net)
        # Residual blocks
        net = tflearn.deep_residual_block(net, self.patch_size, self.depth*2)
        net = tflearn.deep_residual_block(net, 1, self.depth*4, downsample=True)
        net = tflearn.deep_residual_block(net, self.patch_size, self.depth*4)
        net = tflearn.deep_residual_block(net, 1, self.depth*8, downsample=True)
        net = tflearn.deep_residual_block(net, self.patch_size, self.depth*8)
        net_shape = net.get_shape().as_list()
        k_size = [1, net_shape[1], net_shape[2], 1]
        net = tflearn.avg_pool_2d(net, k_size, padding='valid', strides=1)
        # Regression
        net = tflearn.fully_connected(net, self.num_labels, activation='softmax')
        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=300)
        acc = tflearn.metrics.Accuracy()
        self.net = tflearn.regression(net, optimizer=sgd,
                                metric=acc,
                                loss='categorical_crossentropy',
                                learning_rate=0.1)
        self.model = tflearn.DNN(
            self.net,
            tensorboard_verbose=3,
            max_checkpoints=10)

    def train(self, num_steps=1000):
        import tflearn
        with tf.Graph().as_default():
            self.prepare_model()
            self.model.fit(self.train_data, 
                self.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.valid_data, self.valid_labels),
                show_metric=True, 
                snapshot_step=100,
                batch_size=self.batch_size,
                run_id="resnet_mnist")
            self.save_model()


class LSTM(TFL):
    def __init__(self, timesteps=1, **kwargs):
        self.timesteps = timesteps
        super(LSTM, self).__init__(**kwargs)

    def load_dataset(self, dataset):
        if dataset is None:
            self.dataset = self.get_dataset()
        else:
            self.dataset = dataset.copy()
            self.model_name = self.dataset.name
        if len(self.dataset.train_data.shape) > 2:
            raise ValueError("The data shape must be 2 dimensional")
        elif self.dataset.train_data.shape[1] % self.timesteps > 0:
            raise ValueError("The number of features is not divisible by {}".format(self.timesteps))
        self.num_features_t = self.dataset.train_data.shape[1] / self.timesteps
        self._original_dataset_md5 = self.dataset.md5()
        self.reformat_all()

    def convert(self, data):
        ndata = np.ndarray(
            shape=(data.shape[0], data.shape[1]-2, 3), dtype=np.float32)
        for i, row in enumerate(data):
            ndata[i] = np.array(list(zip(row, row[1:], row[2:])))
        return ndata

    def transform_shape(self, data):
        return data.reshape((-1, self.timesteps, self.num_features_t)).astype(np.float32)

    def prepare_model(self, dropout=False):
        import tflearn
        net = tflearn.input_data(shape=[None, self.timesteps, self.num_features_t])
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, self.num_labels, activation='softmax')
        #sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        acc = tflearn.metrics.Accuracy()
        self.net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, metric=acc,
            loss='categorical_crossentropy')
        self.model = tflearn.DNN(self.net, tensorboard_verbose=3)

    def train(self, batch_size=10, num_steps=1000):
        import tflearn
        with tf.Graph().as_default():
            self.prepare_model()
            self.model.fit(self.dataset.train_data, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.valid_data, self.dataset.valid_labels),
                show_metric=True, 
                batch_size=batch_size,
                run_id="lstm_model")
            self.save_model()
