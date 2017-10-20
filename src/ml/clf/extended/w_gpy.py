from ml.clf.wrappers import SKLP
import GPy
import numpy as np


class GPC(SKLP):
    def __init__(self, kernel=None, optimizer='scg', 
                k_params={"variance":7., "lengthscale":0.2}, **kwargs):
        if not 'dataset_train_limit' in kwargs:
            kwargs['dataset_train_limit'] = 1000
        super(GPC, self).__init__(**kwargs)
        self.dim = self.dataset.num_features()
        kernel_f = kernel if kernel is not None else GPy.kern.RBF
        self.k = kernel_f(self.dim, **k_params)
        self.optimizer = optimizer
        self.ext = "npy"
        #bfgs
        #Adadelta

    def train(self, batch_size=128, num_steps=1, n_splits=None):
        from tqdm import tqdm
        
        self.batch_size = batch_size
        self.prepare_model()
        pbar = tqdm(range(1, num_steps + 1))
        for label in pbar:
            self.model.optimize(self.optimizer, max_iters=100, messages=False)
            #self.model.optimize_restarts(num_restarts=10) 
            pbar.set_description("Processing {}".format(label))
        self.save_model()
        self.load_model()
        self.save_model()

    def prepare_model(self, obj_fn=None):
        self.model = GPy.core.GP(
                    X=self.dataset.train_data,
                    Y=self.dataset.train_labels[:].reshape(-1, 1), 
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
        self.model = GPy.models.GPClassification(self.dataset.train_data[:1000], 
            self.dataset.train_labels[:1000].reshape(-1, 1), kernel=self.k, initialize=False)
        if self.check_point_path is not None:
            path = self.make_model_file()
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

    def prepare_model(self, obj_fn=None):
        Z = np.random.rand(100, self.dataset.train_data.shape[1])
        self.model = GPy.core.SVGP(
            X=self.dataset.train_data, 
            Y=self.dataset.train_labels[:].reshape(-1, 1),
            Z=Z, 
            kernel=self.k + GPy.kern.White(self.dataset.num_features(), variance=1e-5), 
            likelihood=GPy.likelihoods.Bernoulli(),
            batchsize=self.batch_size)
        self.model.kern.white.fix()
