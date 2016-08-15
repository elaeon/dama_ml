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
f = np.random.multivariate_normal(np.zeros(SIZE), k.K(X))

lik = GPy.likelihoods.Bernoulli()
p = lik.gp_link.transf(f)

Y = lik.samples(f)
print(Y.shape)
Y = Y.reshape(-1,1)
print(Y.shape)
index = 3
print(Y[index])
print(Y[index+1])
print(Y[index+10])
print(Y[index+100])

def r(X, Y):
    m = GPy.core.GP(X=X,
                    Y=Y, 
                    kernel=k, 
                    inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                    likelihood=lik)

    for i in range(5):
        m.optimize('bfgs', max_iters=100)
        print('iteration:', i)
        print(m)
        print("")

    probs = m.predict(np.asarray([X[index]]))[0]
    print(probs)
    #GPy.util.classification.conf_matrix(probs, Y[0])
    np.save('model_save.npy', m.param_array)

if 0:
    r(X, Y)
else:
    #pass
    #m = GPy.core.GP(X, Y, initialize=False)
    m = GPy.models.GPClassification(X, Y, initialize=True)
    m[:] = np.load('model_save.npy')
    #m.initialize_parameter()
    for index_ in [index, index+1, index+10, index+100]:
        probs = m.predict(np.asarray([X[index_]]))[0]
        print(probs)
