from generic import *

class OneClassSVM(SKL):
    def __init__(self, *args, **kwargs):
        super(OneClassSVM, self).__init__(*args, **kwargs)
        self.label_ref = 1
        self.label_other = 0

    def prepare_model(self):
        from sklearn import svm
        self.dataset.dataset = self.dataset.train_data
        self.dataset.labels = self.dataset.train_labels
        dataset_ref, _ = self.dataset.only_labels([self.label_ref])
        reg = svm.OneClassSVM(nu=.2, kernel="rbf", gamma=0.5)
        reg.fit(dataset_ref)
        self.model = reg

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(self.transform_shape(data)):
            label = self.label_other if prediction == -1 else self.label_ref
            yield self.convert_label(label)


class SVC(SKL):
    def prepare_model(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn import svm
        reg = svm.LinearSVC(C=1, max_iter=1000)
        reg = reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class RandomForest(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        reg = RandomForestClassifier(n_estimators=25, min_samples_split=2)
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class LogisticRegression(SKLP):
    def prepare_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")#"newton-cg")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class TensorFace(TF):
    def prepare_model(self, batch_size, dropout=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(batch_size, self.num_features))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(self.dataset.valid_data)
            self.tf_test_dataset = tf.constant(self.dataset.test_data)

            # Variables.
            weights = tf.Variable(
                tf.truncated_normal([self.num_features, self.num_labels]))
            biases = tf.Variable(tf.zeros([self.num_labels]))

            # Training computation.
            if dropout is True:
                hidden = tf.nn.dropout(self.tf_train_dataset, 0.5, seed=66478)
                self.logits = tf.matmul(hidden, weights) + biases
            else:
                self.logits = tf.matmul(self.tf_train_dataset, weights) + biases
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

            regularizers = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            self.loss += 5e-4 * regularizers

            # Optimizer.
            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(
                tf.matmul(self.tf_valid_dataset, weights) + biases)
            self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, weights) + biases)

    def _predict(self, imgs):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.check_point + self.dataset.name + "/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("...no checkpoint found...")

            for img in imgs:
                img = self.transform_shape(img)
                feed_dict = {self.tf_train_dataset: img}
                classification = session.run(self.train_prediction, feed_dict=feed_dict)
                yield self.convert_label(classification)


class GPC(TFL):
    def __init__(self, kernel=None, k_params={}, **kwargs):
        super(GPC, self).__init__(**kwargs)
        import GPy
        self.dim = self.dataset.num_features()
        kernel_f = kernel if kernel is not None else GPy.kern.RBF
        self.k = kernel_f(self.dim, **k_params)

    def train(self, batch_size=128, num_steps=1):
        from tqdm import tqdm
        self.prepare_model()
        pbar = tqdm(range(1, num_steps + 1))
        for label in pbar:
            self.model.optimize('scg', max_iters=100, messages=False) #bfgs
            pbar.set_description("Processing {}".format(label))
        self.save_model()

    def transform_to_gpy_labels(self, labels):
        t_labels = np.ndarray(
            shape=(labels.shape[0], 1), dtype=np.float32)
        for i, label in enumerate(labels):
            t_labels[i] = self.convert_label(label, raw=False,)
        return t_labels

    def prepare_model(self):
        import GPy
        k = GPy.kern.RBF(self.dim, variance=7., lengthscale=0.2)
        self.model = GPy.core.GP(X=self.dataset.train_data,
                    Y=self.transform_to_gpy_labels(self.dataset.train_labels), 
                    kernel=self.k + GPy.kern.White(1), 
                    inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                    likelihood=GPy.likelihoods.Bernoulli())

        self.model.kern.white.variance = 1e-2
        self.model.kern.white.fix()

    def save_model(self):
        path = self.make_model_file()
        self.save_meta()
        np.save(path, self.model.param_array)

    def _predict(self, data, raw=False):
        for prediction in self.model.predict(data)[0]:
            p = [1 - prediction[0], prediction[0]]
            yield self.convert_label(p, raw=raw)

    def load_model(self):
        import GPy
        self.model = GPy.models.GPClassification(self.dataset.train_data, 
            self.dataset.train_labels.reshape(-1, 1), kernel=self.k, initialize=False)    
        r = np.load(self.check_point+self.dataset.name+"/"+self.dataset.name+".npy")
        self.model[:] = r[:2]
        self.model.initialize_parameter()


class MLP(TFL):
    def __init__(self, *args, **kwargs):
        if "layers" in kwargs:
            self.layers = kwargs["layers"]
            del kwargs["layers"]
        else:
            self.layers = [128, 64]
        super(MLP, self).__init__(*args, **kwargs)

    def prepare_model(self, dropout=False):
        import tflearn
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
        self.net = tflearn.regression(softmax, optimizer=sgd, metric=acc,
                         loss='categorical_crossentropy')

    def train(self, batch_size=10, num_steps=1000):
        import tflearn
        with tf.Graph().as_default():
            self.prepare_model()
            self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
            self.model.fit(self.dataset.train_data, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.valid_data, self.dataset.valid_labels),
                show_metric=True, 
                batch_size=batch_size,
                run_id="dense_model")
            self.save_model()


class ConvTensor(TFL):
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

    def prepare_model(self, dropout=False):
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
        self.net = tflearn.regression(network, optimizer='adam', metric=acc, learning_rate=0.01,
                                 loss='categorical_crossentropy', name='target')

    def train(self, batch_size=10, num_steps=1000):
        import tflearn
        with tf.Graph().as_default():
            self.prepare_model()
            self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
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

    def train(self, num_steps=1000):
        import tflearn
        with tf.Graph().as_default():
            self.prepare_model()
            self.model = tflearn.DNN(self.net, 
                checkpoint_path='model_resnet_mnist', 
                tensorboard_verbose=3,
                max_checkpoints=10)

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
        if len(self.dataset.data.shape) > 2:
            raise ValueError("The data shape must be 2 dimensional")
        elif self.dataset.data.shape[1] % self.timesteps > 0:
            raise ValueError("The number of features is not divisible by {}".format(self.timesteps))
        self.num_features_t = self.dataset.data.shape[1] / self.timesteps
        self.reformat_all()

    def convert(self, data):
        ndata = np.ndarray(
            shape=(data.shape[0], data.shape[1]-2, 3), dtype=np.float32)
        for i, row in enumerate(data):
            ndata[i] = np.array(list(zip(row, row[1:], row[2:])))
        return ndata

    def transform_shape(self, data):
        #return self.convert(data)
        return data.reshape((-1, self.timesteps, self.num_features_t)).astype(np.float32)

    def prepare_model(self, dropout=False):
        import tflearn
        net = tflearn.input_data(shape=[None, self.timesteps, self.num_features_t])
        #net = tflearn.input_data(shape=[None, 19, 3])
        #net = tflearn.embedding(net, input_dim=1000, output_dim=128)
        net = tflearn.lstm(net, 128, dropout=0.8)
        net = tflearn.fully_connected(net, self.num_labels, activation='softmax')
        #sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        acc = tflearn.metrics.Accuracy()
        self.net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, metric=acc,
            loss='categorical_crossentropy')

    def train(self, batch_size=10, num_steps=1000):
        import tflearn
        with tf.Graph().as_default():
            self.prepare_model()
            self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
            self.model.fit(self.dataset.train_data, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.valid_data, self.dataset.valid_labels),
                show_metric=True, 
                batch_size=batch_size,
                run_id="lstm_model")
            self.save_model()


class Tensor2LFace(TF):
    def __init__(self, *args, **kwargs):
        super(Tensor2LFace, self).__init__(*args, **kwargs)
        self.num_hidden = 1024

    def layers(self, dropout):
        W1 = tf.Variable(
            tf.truncated_normal([self.image_size * self.image_size, self.num_hidden], stddev=1.0 / 10), 
            name='weights')
        b1 = tf.Variable(tf.zeros([self.num_hidden]), name='biases')
        hidden = tf.nn.relu(tf.matmul(self.tf_train_dataset, W1) + b1)

        W2 = tf.Variable(
            tf.truncated_normal([self.num_hidden, self.num_labels]))
        b2 = tf.Variable(tf.zeros([self.num_labels]))

        if dropout is True:
            hidden = tf.nn.dropout(hidden, 0.5, seed=66478)
        self.logits = tf.matmul(hidden, W2) + b2
        return W1, b1, W2, b2

    def fit(self, dropout=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(self.batch_size, self.image_size * self.image_size))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(self.valid_dataset)
            self.tf_test_dataset = tf.constant(self.test_dataset)

            W1, b1, W2, b2 = self.layers(dropout)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

            regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)
            self.loss += 5e-4 * regularizers

            self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

            self.train_prediction = tf.nn.softmax(self.logits)
            hidden_valid =  tf.nn.relu(tf.matmul(self.tf_valid_dataset, W1) + b1)
            valid_logits = tf.matmul(hidden_valid, W2) + b2
            self.valid_prediction = tf.nn.softmax(valid_logits)
            hidden_test = tf.nn.relu(tf.matmul(self.tf_test_dataset, W1) + b1)
            test_logits = tf.matmul(hidden_test, W2) + b2
            self.test_prediction = tf.nn.softmax(test_logits)


class ConvTensorFace(TF):
    def __init__(self, *args, **kwargs):
        self.num_channels = 1
        self.patch_size = 5
        self.depth = 32
        super(ConvTensorFace, self).__init__(*args, **kwargs)        
        self.num_hidden = 64

    def transform_shape(self, img):
        return img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)

    def layers(self, data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, 
            layer3_weights, layer3_biases, dropout=False):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool = tf.nn.max_pool(hidden,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer2_weights) + layer2_biases)
        if dropout:
            hidden = tf.nn.dropout(hidden, 0.5, seed=66478)
        return tf.matmul(hidden, layer3_weights) + layer3_biases

    def fit(self, dropout=True):
        import math
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(self.batch_size, self.image_size, self.image_size, self.num_channels))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(self.valid_data)
            self.tf_test_dataset = tf.constant(self.test_data)

            # Variables.
            layer3_size = int(math.ceil(self.image_size / 4.))
            layer1_weights = tf.Variable(tf.truncated_normal(
                [self.patch_size, self.patch_size, self.num_channels, self.depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([self.depth]))
            layer2_weights = tf.Variable(tf.truncated_normal(
                [layer3_size * layer3_size * self.depth, self.num_hidden], stddev=0.1)) # 4 num of ksize
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
            layer3_weights = tf.Variable(tf.truncated_normal(
                [self.num_hidden, self.num_labels], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))

            self.logits = self.layers(self.tf_train_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases, dropout=dropout)

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))
            regularizers = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) +\
            tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) +\
            tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases)
            self.loss += 5e-4 * regularizers

            # Optimizer: set up a variable that's incremented once per batch and
            # controls the learning rate decay.
            batch = tf.Variable(0)
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
              0.01,                # Base learning rate.
              batch * self.batch_size,  # Current index into the dataset.
              self.train_labels.shape[0],          # train_labels.shape[0] Decay step.
              0.95,                # Decay rate.
              staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss,
                global_step=batch)

            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(self.logits)
            self.valid_prediction = tf.nn.softmax(self.layers(self.tf_valid_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases))
            self.test_prediction = tf.nn.softmax(self.layers(self.tf_test_dataset, layer1_weights, 
                layer1_biases, layer2_weights, layer2_biases, layer3_weights, 
                layer3_biases))

    def train(self, num_steps=0):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in xrange(int(15 * self.train_labels.shape[0]) // self.batch_size):
                offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
                batch_data = self.train_data[offset:(offset + self.batch_size), :, :, :]
                batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
                feed_dict = {self.tf_train_data : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 5000 == 0):
                    print "Minibatch loss at step", step, ":", l
                    print "Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels)
                    print "Validation accuracy: %.1f%%" % self.accuracy(
                    self.valid_prediction.eval(), self.valid_labels)
            score = self.accuracy(self.test_prediction.eval(), self.test_labels)
            #print('Test accuracy: %.1f' % score)
            self.save_model(saver, session, step)
            return score
