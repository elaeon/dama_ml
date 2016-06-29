import os
import numpy as np
import tensorflow as tf

from sklearn import preprocessing

CHECK_POINT_PATH = "/home/sc/data/face_recog/"
#np.random.seed(133)
    
class Measure(object):
    def __init__(self, predictions, labels):
        if len(predictions.shape) > 1:
            self.transform = lambda x: np.argmax(x, 1)
        else:
            self.transform = lambda x: x
        
        self.labels = labels
        self.predictions = predictions
        self.average = "macro"

    def accuracy(self):
        from sklearn.metrics import accuracy_score
        return accuracy_score(self.labels, self.transform(self.predictions))

    #false positives
    def precision(self):
        from sklearn.metrics import precision_score
        return precision_score(self.labels, self.transform(self.predictions), 
            average=self.average, pos_label=None)

    #false negatives
    def recall(self):
        from sklearn.metrics import recall_score
        return recall_score(self.labels, self.transform(self.predictions), 
            average=self.average, pos_label=None)

    #weighted avg presicion and recall
    def f1(self):
        from sklearn.metrics import f1_score
        return f1_score(self.labels, self.transform(self.predictions), 
            average=self.average, pos_label=None)

    def logloss(self):
        from sklearn.metrics import log_loss
        return log_loss(self.labels, self.predictions)

    def print_all(self):
        print("#############")
        print("Accuracy: {}%".format(self.accuracy()*100))
        print("Precision: {}%".format(self.precision()*100))
        print("Recall: {}%".format(self.recall()*100))
        print("F1: {}%".format(self.f1()*100))
        print("#############")

class BasicFaceClassif(object):
    def __init__(self, dataset, check_point_path=CHECK_POINT_PATH, pprint=True):
        #self.image_size = dataset.image_size
        self.model = None
        self.pprint = pprint
        self.le = preprocessing.LabelEncoder()
        self.load_dataset(dataset)
        self.check_point_path = check_point_path
        self.check_point = check_point_path + self.__class__.__name__ + "/"

    def detector_test_dataset(self, raw=False):
        predictions = self.predict(self.dataset.test_dataset, raw=raw)
        measure = Measure(np.asarray(list(predictions)), np.asarray([self.convert_label(label) 
            for label in self.dataset.test_labels]))
        return self.__class__.__name__, measure

    def reformat(self, dataset, labels):
        dataset = self.transform_img(dataset)
        return dataset, labels

    def transform_img(self, img):
        return img.reshape(img.shape[0], -1).astype(np.float32)

    def labels_encode(self, labels):
        self.le.fit(labels)
        self.num_labels = self.le.classes_.shape[0]

    def position_index(self, label):
        return label

    def convert_label(self, label, raw=False):
        if raw is True:
            return label
        else:
            return self.le.inverse_transform(self.position_index(label))

    def reformat_all(self):
        all_ds = np.concatenate((self.dataset.train_labels, 
            self.dataset.valid_labels, self.dataset.test_labels), axis=0)
        self.labels_encode(all_ds)
        self.dataset.train_dataset, self.dataset.train_labels = self.reformat(
            self.dataset.train_dataset, self.le.transform(self.dataset.train_labels))
        self.dataset.test_dataset, self.dataset.test_labels = self.reformat(
            self.dataset.test_dataset, self.le.transform(self.dataset.test_labels))
        self.num_features = self.dataset.test_dataset.shape[-1]

        if len(self.dataset.valid_labels) > 0:
            self.dataset.valid_dataset, self.dataset.valid_labels = self.reformat(
                self.dataset.valid_dataset, self.le.transform(self.dataset.valid_labels))

        if self.pprint:
            print('RF-Training set', self.dataset.train_dataset.shape, self.dataset.train_labels.shape)
            if self.dataset.valid_dataset.shape[0] > 0:
                print('RF-Validation set', self.dataset.valid_dataset.shape,
                    self.dataset.valid_labels.shape)
            print('RF-Test set', self.dataset.test_dataset.shape, self.dataset.test_labels.shape)

    def accuracy(self, predictions, labels):
        measure = Measure(predictions, labels)
        if self.pprint:
            measure.print_all()
        return measure.accuracy()

    def load_dataset(self, dataset):
        from ml.ds import DataSetBuilder
        self.dataset = DataSetBuilder(dataset.name)
        self.dataset.image_size = dataset.image_size
        self.dataset.train_dataset = dataset.train_dataset
        self.dataset.train_labels = dataset.train_labels
        self.dataset.valid_dataset = dataset.valid_dataset
        self.dataset.valid_labels = dataset.valid_labels
        self.dataset.test_dataset = dataset.test_dataset
        self.dataset.test_labels = dataset.test_labels
        self.reformat_all()

class Binary(BasicFaceClassif):
    def __init__(self, *args, **kwargs):
        super(Binary, self).__init__(*args, **kwargs)
        self.label_ref = 1
        self.label_other = 0

    def prepare_model(self):
        from sklearn import svm
        self.dataset.dataset = self.dataset.train_dataset
        self.dataset.labels = self.dataset.train_labels
        dataset_ref, _ = self.dataset.only_labels([self.label_ref])
        reg = svm.OneClassSVM(nu=.2, kernel="rbf", gamma=0.5)
        reg.fit(dataset_ref)
        self.model = reg

    def train(self, batch_size=0, num_steps=0):
        self.prepare_model()
        predictions = self.model.predict(self.dataset.test_dataset)
        score = self.accuracy(predictions, self.dataset.test_labels)
        self.save_model()
        return score

    def predict(self, data, raw=False):
        return self._predict(data)

    def _predict(self, data, raw=False):
        if self.model is None:
            self.load_model()

        if isinstance(data, list):
            data = self.transform_img(np.asarray(data))

        data = preprocessing.scale(data)
        for prediction in self.model.predict(self.transform_img(data)):
            label = self.label_other if prediction == -1 else self.label_ref
            yield self.convert_label(label)

    def save_model(self):
        from sklearn.externals import joblib
        if not os.path.exists(self.check_point):
            os.makedirs(self.check_point)
        if not os.path.exists(self.check_point + self.dataset.name + "/"):
            os.makedirs(self.check_point + self.dataset.name + "/")
        joblib.dump(self.model, '{}.pkl'.format(
            self.check_point+self.dataset.name+"/"+self.dataset.name))

    def load_model(self):
        from sklearn.externals import joblib
        self.model = joblib.load('{}.pkl'.format(
            self.check_point+self.dataset.name+"/"+self.dataset.name))


class SVCFace(Binary):
    def predict(self, data, raw=False):
        return self._predict(data, raw=raw)

    def _predict(self, data, raw=False):
        if self.model is None:
            self.load_model()

        if isinstance(data, list):
            data = np.asarray(data)

        data = preprocessing.scale(data)
        for prediction in self.model.predict(self.transform_img(data)):
            yield self.convert_label(prediction)

    def prepare_model(self):
        from sklearn import svm
        reg = svm.LinearSVC(C=1, max_iter=1000)
        reg = reg.fit(self.dataset.train_dataset, self.dataset.train_labels)
        self.model = reg


class RandomForest(Binary):
    def position_index(self, label):
        return np.argmax(label)

    def predict(self, data, raw=False):
        return self._predict(data, raw=raw)

    def _predict(self, data, raw=False):
        if self.model is None:
            self.load_model()

        if isinstance(data, list):
            data = np.asarray(data)

        data = preprocessing.scale(data)
        for prediction in self.model.predict_proba(self.transform_img(data)):
            yield self.convert_label(prediction, raw=raw)

    def prepare_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        reg = RandomForestClassifier(n_estimators=25, min_samples_split=2)
        reg.fit(self.dataset.train_dataset, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_dataset, self.dataset.valid_labels)
        self.model = sig_clf

class LogisticRegression(RandomForest):
    def prepare_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")#"newton-cg")
        reg.fit(self.dataset.train_dataset, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_dataset, self.dataset.valid_labels)
        self.model = sig_clf

class BasicTensor(BasicFaceClassif):
    def reformat(self, dataset, labels):
        dataset = self.transform_img(dataset)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels_m = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels_m

    def prepare_model(self, batch_size, dropout=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                            shape=(batch_size, self.num_features))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
            self.tf_valid_dataset = tf.constant(self.dataset.valid_dataset)
            self.tf_test_dataset = tf.constant(self.dataset.test_dataset)

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

    def train(self, batch_size=10, num_steps=3001):
        self.prepare_model(batch_size)
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            print "Initialized"
            for step in xrange(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (self.dataset.train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = self.dataset.train_dataset[offset:(offset + batch_size), :]
                batch_labels = self.train_labels[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print "Minibatch loss at step", step, ":", l
                    print "Minibatch accuracy: %.1f%%" % (self.accuracy(predictions, batch_labels)*100)
                    print "Validation accuracy: %.1f%%" % (self.accuracy(
                      self.valid_prediction.eval(), self.dataset.valid_labels)*100)
            score_v = self.accuracy(self.test_prediction.eval(), self.dataset.test_labels)
            self.save_model(saver, session, step)
            return score_v

    def save_model(self, saver, session, step):
        if not os.path.exists(self.check_point):
            os.makedirs(self.check_point)
        if not os.path.exists(self.check_point + self.dataset.name + "/"):
            os.makedirs(self.check_point + self.dataset.name + "/")
        
        saver.save(session, 
                '{}{}.ckpt'.format(self.check_point + self.dataset.name + "/", self.dataset.name), 
                global_step=step)

class TensorFace(BasicTensor):

    def position_index(self, label):
        return np.argmax(label)

    def predict(self, imgs):
        self.batch_size = 1
        self.fit(dropout=False)
        return self._predict(imgs)

    def _predict(self, imgs):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.check_point + self.dataset.name + "/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("...no checkpoint found...")

            for img in imgs:
                img = self.transform_img(img)
                feed_dict = {self.tf_train_dataset: img}
                classification = session.run(self.train_prediction, feed_dict=feed_dict)
                yield self.convert_label(classification)

class TfLTensor(TensorFace):
    def prepare_model(self, dropout=False):
        import tflearn
        input_layer = tflearn.input_data(shape=[None, self.num_features])
        dense1 = tflearn.fully_connected(input_layer, 128, activation='tanh',
                                         regularizer='L2', weight_decay=0.001)#128
        dropout1 = tflearn.dropout(dense1, 0.5)
        dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                        regularizer='L2', weight_decay=0.001)#64
        dropout2 = tflearn.dropout(dense2, 0.5)
        softmax = tflearn.fully_connected(dropout2, self.num_labels, activation='softmax')

        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        #top_k = tflearn.metrics.Top_k(5)
        acc = tflearn.metrics.Accuracy()
        self.net = tflearn.regression(softmax, optimizer=sgd, metric=acc,
                                 loss='categorical_crossentropy')

    def train(self, batch_size=10, num_steps=1000):
        import tflearn
        with tf.Graph().as_default():
            self.prepare_model()
            self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
            self.model.fit(self.dataset.train_dataset, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.valid_dataset, self.dataset.valid_labels),
                show_metric=True, 
                batch_size=batch_size,
                run_id="dense_model")
            self.save_model()
    
    def save_model(self):
        if not os.path.exists(self.check_point):
            os.makedirs(self.check_point)
        if not os.path.exists(self.check_point + self.dataset.name + "/"):
            os.makedirs(self.check_point + self.dataset.name + "/")

        self.model.save('{}{}.ckpt'.format(
            self.check_point + self.dataset.name + "/", self.dataset.name))

    def load_model(self):
        import tflearn
        self.prepare_model()
        self.model = tflearn.DNN(self.net, tensorboard_verbose=3)
        self.model.load('{}{}.ckpt'.format(
            self.check_point + self.dataset.name + "/", self.dataset.name))

    def predict(self, data, raw=False):
        return self._predict(data, raw=raw)

    def _predict(self, data, raw=False):
        with tf.Graph().as_default():
            if self.model is None:
                self.load_model()

            if isinstance(data, list):
                data = np.asarray(data)

            data = preprocessing.scale(data)
            for prediction in self.model.predict(self.transform_img(data)):
                    yield self.convert_label(prediction, raw=raw)

class ConvTensor(TfLTensor):
    def __init__(self, *args, **kwargs):
        self.num_channels = kwargs.get("num_channels", 1)
        self.patch_size = 3
        self.depth = 32
        if "num_channels" in kwargs:
            del kwargs["num_channels"]
        super(ConvTensor, self).__init__(*args, **kwargs)

    def transform_img(self, img):
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
            self.model.fit(self.dataset.train_dataset, 
                self.dataset.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.dataset.valid_dataset, self.dataset.valid_labels),
                show_metric=True, 
                batch_size=batch_size,
                snapshot_step=100,
                run_id="conv_model")
            self.save_model()

class ResidualTensor(TfLTensor):
    def __init__(self, *args, **kwargs):
        self.num_channels = kwargs.get("num_channels", 1)
        self.patch_size = 3
        self.depth = 32
        if "num_channels" in kwargs:
            del kwargs["num_channels"]
        super(ResidualTensor, self).__init__(*args, **kwargs)

    def transform_img(self, img):
        return img.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)

    def reformat_all(self):
        import tflearn.data_utils as du
        all_ds = np.concatenate((self.train_labels, self.valid_labels, self.test_labels), axis=0)
        self.labels_encode(all_ds)
        self.train_dataset, self.train_labels = self.reformat(
            self.train_dataset, self.le.transform(self.train_labels))
        self.valid_dataset, self.valid_labels = self.reformat(
            self.valid_dataset, self.le.transform(self.valid_labels))
        self.test_dataset, self.test_labels = self.reformat(
            self.test_dataset, self.le.transform(self.test_labels))

        self.train_dataset, mean = du.featurewise_zero_center(self.train_dataset)
        self.test_dataset = du.featurewise_zero_center(self.test_dataset, mean)

        print('RF-Training set', self.train_dataset.shape, self.train_labels.shape)
        print('RF-Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('RF-Test set', self.test_dataset.shape, self.test_labels.shape)

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

            self.model.fit(self.train_dataset, 
                self.train_labels, 
                n_epoch=num_steps, 
                validation_set=(self.valid_dataset, self.valid_labels),
                show_metric=True, 
                snapshot_step=100,
                batch_size=self.batch_size,
                run_id="resnet_mnist")
            self.save_model()


class Tensor2LFace(TensorFace):
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


class ConvTensorFace(TensorFace):
    def __init__(self, *args, **kwargs):
        self.num_channels = 1
        self.patch_size = 5
        self.depth = 32
        super(ConvTensorFace, self).__init__(*args, **kwargs)        
        self.num_hidden = 64

    def transform_img(self, img):
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
            self.tf_valid_dataset = tf.constant(self.valid_dataset)
            self.tf_test_dataset = tf.constant(self.test_dataset)

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
                batch_data = self.train_dataset[offset:(offset + self.batch_size), :, :, :]
                batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
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


class ClassifTest(object):
    def __init__(self, logloss=False):
        self.logloss = logloss
        self.headers = ["CLF", "Precision", "Recall", "F1"]
        if logloss is True:
            self.headers = self.headers + ["logloss"]

    def dataset_test(self, classifs, dataset, order_column):
        from utils.order import order_table_print
        table = []
        print("DATASET", dataset.name)
        for classif_name in classifs:
            classif = classifs[classif_name]["name"]
            params = classifs[classif_name]["params"]
            clf = classif(dataset, **params)
            name_clf, measure = clf.detector_test_dataset(raw=self.logloss)
            if self.logloss:
                table.append((name_clf, measure.precision(), measure.recall(), 
                    measure.f1(), measure.logloss()))
            else:
                table.append((name_clf, measure.precision(), measure.recall(), measure.f1()))
        order_table_print(self.headers, table, order_column)

    def classif_test(self, clf, order_column):
        from utils.order import order_table_print
        table = []
        name_clf, measure = clf.detector_test_dataset(raw=self.logloss)
        if self.logloss:
            table = [(name_clf, measure.precision(), measure.recall(), 
                measure.f1(), measure.logloss())]
        else:
            table = [(name_clf, measure.precision(), measure.recall(), measure.f1())]
        order_table_print(self.headers, table, order_column)
