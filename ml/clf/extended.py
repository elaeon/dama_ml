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
        reg = CalibratedClassifierCV(
            svm.LinearSVC(C=1, max_iter=1000), method="sigmoid")
        reg = reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class RandomForest(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        reg = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class ExtraTrees(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.calibration import CalibratedClassifierCV
        reg = CalibratedClassifierCV(
            ExtraTreesClassifier(n_estimators=25, min_samples_split=2), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class LogisticRegression(SKLP):
    def prepare_model(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        reg = CalibratedClassifierCV(
            LogisticRegression(solver="lbfgs", multi_class="multinomial")#"newton-cg")
            , method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class SGDClassifier(SKLP):
    def prepare_model(self):
        from sklearn.linear_model import SGDClassifier
        from sklearn.calibration import CalibratedClassifierCV
        reg = CalibratedClassifierCV(
            SGDClassifier(loss='log', penalty='elasticnet', 
            alpha=.0001, n_iter=100, n_jobs=-1), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class AdaBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.calibration import CalibratedClassifierCV
        reg = CalibratedClassifierCV(
            AdaBoostClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class GradientBoost(SKLP):
    def prepare_model(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        reg = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=25, learning_rate=1.0), method="sigmoid")
        reg.fit(self.dataset.train_data, self.dataset.train_labels)
        sig_clf = CalibratedClassifierCV(reg, method="sigmoid", cv="prefit")
        sig_clf.fit(self.dataset.valid_data, self.dataset.valid_labels)
        self.model = sig_clf


class GPC(SKLP):
    def __init__(self, kernel=None, optimizer='scg', 
                k_params={"variance":7., "lengthscale":0.2}, **kwargs):
        if not 'dataset_train_limit' in kwargs:
            kwargs['dataset_train_limit'] = 1000
        super(GPC, self).__init__(**kwargs)
        import GPy
        self.dim = self.dataset.num_features()
        kernel_f = kernel if kernel is not None else GPy.kern.RBF
        self.k = kernel_f(self.dim, **k_params)
        self.optimizer = optimizer
        #bfgs
        #Adadelta

    def train(self, batch_size=128, num_steps=1):
        from tqdm import tqdm
        self.prepare_model()
        pbar = tqdm(range(1, num_steps + 1))
        for label in pbar:
            self.model.optimize(self.optimizer, max_iters=100, messages=False) 
            pbar.set_description("Processing {}".format(label))
        self.save_model()
        self.load_model()

    def transform_to_gpy_labels(self, labels):
        t_labels = np.ndarray(
            shape=(labels.shape[0], 1), dtype=np.float32)
        for i, label in enumerate(labels):
            t_labels[i] = self.convert_label(label, raw=False,)
        return t_labels

    def prepare_model(self):
        import GPy
        self.model = GPy.core.GP(
                    X=self.dataset.train_data,
                    Y=self.transform_to_gpy_labels(self.dataset.train_labels), 
                    kernel=self.k + GPy.kern.White(1), 
                    inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                    likelihood=GPy.likelihoods.Bernoulli())

        self.model.kern.white.variance = 1e-2
        self.model.kern.white.fix()

    def save_model(self):
        if self.check_point_path is not None:
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
        if self.check_point_path is not None:
            path = self.make_model_file(check=False)
            r = np.load(path+".npy")
            self.model.update_model(False) # do not call the underlying expensive algebra on load
            self.model.initialize_parameter() # Initialize the parameters (connect the parameters up)
            self.model[:] = r[:2]
            self.model.update_model(True) # Call the algebra only once 


class SVGPC(GPC):
    def __init__(self, kernel=None, optimizer='Adadelta', 
                k_params={"variance":7., "lengthscale":0.2}, **kwargs):
        super(SVGPC, self).__init__(kernel=kernel, optimizer=optimizer, 
                                    k_params=k_params, **kwargs)

    def prepare_model(self):
        import GPy
        Z = np.random.rand(100, self.dataset.train_data.shape[1])
        self.model = GPy.core.SVGP(
            X=self.dataset.train_data, 
            Y=self.transform_to_gpy_labels(self.dataset.train_labels), 
            Z=Z, 
            kernel=self.k + GPy.kern.White(1), 
            likelihood=GPy.likelihoods.Bernoulli(),
            batchsize=100)
        self.model.kern.white.variance = 1e-5
        self.model.kern.white.fix()


class MLP(TFL):
    def __init__(self, *args, **kwargs):
        if "layers" in kwargs:
            self.layers = kwargs["layers"]
            del kwargs["layers"]
        else:
            self.layers = [128, 64]
        super(MLP, self).__init__(*args, **kwargs)

    def prepare_model(self):
        from ml.models import MMLP
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


