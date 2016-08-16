import numpy as np
import ml
import GPy
from ml.processing import Preprocessing
from matplotlib import pyplot as plt

np.random.seed(1)
DIM = 21
SIZE = 1000
k = GPy.kern.RBF(DIM, variance=7., lengthscale=0.2)
X = np.random.rand(SIZE, DIM)
#f = np.random.multivariate_normal(np.zeros(SIZE), k.K(X))

lik = GPy.likelihoods.Bernoulli()
#p = lik.gp_link.transf(f)

#Y = lik.samples(f)
Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
print(Y.shape)
index = 3
print(Y[index])
print(Y[index+1])
print(Y[index+10])
print(Y[index+100])

def mll():
    dataset = ml.ds.DataSetBuilder("gpc_test", dataset_path="/home/sc/")
    dataset.build_from_data_labels(X, Y)
    classif = ml.clf_e.GPC(dataset=dataset, check_point_path="/home/sc/ml_data/checkpoints/")
    classif.train(batch_size=128, num_steps=1)
    classif.print_score()

def gp(X, Y):
    m = GPy.core.GP(X=X,
                    Y=Y, 
                    kernel=k + GPy.kern.White(1), 
                    inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                    likelihood=lik)

    m.kern.white.variance = 1e-5
    m.kern.white.fix()
    #for i in range(5):
    #    m.optimize('bfgs', max_iters=100)
    #    print('iteration:', i)
    #    print(m)
    #    print("")

    pred(m, X, index)
    np.save('model_save.npy', m.param_array)

def svgp(X, Y):
    import climin
    import sys
    #Z = np.random.rand(20, DIM)
    i = np.random.permutation(X.shape[0])[:10]
    Z = X[i].copy()
    batchsize = 50
    Y = Y.reshape(-1,1)
    m = GPy.core.SVGP(
        X, 
        Y, 
        Z, 
        k + GPy.kern.White(1), 
        lik,
        batchsize=batchsize)
    m.kern.white.variance = 1e-5
    m.kern.white.fix()
    #opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)
    for i in range(50):
        m.optimize('Adadelta', max_iters=100)
        print('iteration:', i)
    #    print(m)
    #    print("")

    #def callback(i):
        #Stop after 5000 iterations
    #    if i['n_iter'] > 5000:
    #        return True
    #    return False
    #info = opt.minimize_until(callback)
    pred(m, X, index)
    #np.save('model_save2.npy', m.param_array)

def pred(m, X, index):    
    #GPy.util.classification.conf_matrix(probs, Y[0])
    for index_ in [index, index+1, index+10, index+100]:
        probs = m.predict(np.asarray([X[index_]]))[0]
        print(probs)

if 1:
    #mll()
    #gp(X, Y)
    svgp(X, Y)
else:
    #pass
    #m = GPy.core.GP(X, Y, initialize=False)
    #m = GPy.models.SparseGPClassification(X, Y, kernel=k, initialize=True)
    m = GPy.models.GPClassification(X, Y, kernel=k, initialize=False)    
    r = np.load('model_save.npy')    
    print(r.shape)
    m[:] = r[:2]
    m.initialize_parameter()
    pred(m, X, index)
