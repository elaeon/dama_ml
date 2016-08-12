import numpy as np
import ml
import GPy
from ml.processing import Preprocessing
from matplotlib import pyplot as plt

np.random.seed(1)
dataset_name = "crabs"
dataset_path = "/home/sc/"
check_point_path = "/home/sc/ml_data/checkpoints/"

class PreprocessingNull(Preprocessing):
    def scale(self):
        pass


dataset = ml.ds.DataSetBuilderFile.load_dataset(dataset_name, dataset_path=dataset_path, processing_class=PreprocessingNull)

DIM = 3
k = GPy.kern.RBF(DIM, variance=7., lengthscale=0.2)
X = np.random.rand(200, DIM)
f = np.random.multivariate_normal(np.zeros(200), k.K(X))

#plt.plot(X, f, 'bo')
#plt.title('latent function values')
#plt.xlabel('$x$')
#plt.ylabel('$f(x)$')
#plt.show()

lik = GPy.likelihoods.Bernoulli()
p = lik.gp_link.transf(f) # squash the latent function
#plt.plot(X, p, 'ro')
#plt.title('latent probabilities');plt.xlabel('$x$');plt.ylabel('$\sigma(f(x))$')
#plt.show()

Y = lik.samples(f)
print(Y.shape)
Y = Y.reshape(-1,1)
print(Y.shape)
index = 3
print(X[index], Y[index])
#plt.plot(X, Y, 'kx', mew=2);
#plt.ylim(-0.1, 1.1)
#plt.title('Bernoulli draws');
#plt.xlabel('$x$');plt.ylabel('$y \in \{0,1\}$')
#plt.show()

m = GPy.core.GP(X=X,
                Y=Y, 
                kernel=k, 
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik)

#print(m, '\n')
#for i in range(5):
#    m.optimize('bfgs', max_iters=100)
#    print('iteration:', i)
#    print(m)
#    print("")

probs = m.predict(np.asarray([X[index]]))[0]
print(probs)
#GPy.util.classification.conf_matrix(probs, Y[0])

#print(m)
#m.plot()
#plt.plot(X, p, 'ro')
#plt.ylabel('$y, p(y=1)$')
#plt.xlabel('$x$')
#plt.show()

#m.plot_f()
#plt.plot(X, f, 'bo')
#plt.ylabel('$f(x)$')
#plt.xlabel('$x$')
#plt.show()

#print(m, '\n')
#for i in range(5):
#    m.optimize('bfgs', max_iters=100)
#    print('iteration:', i)
#    print(m)
#    print("")

#m.plot()
#plt.plot(X, p, 'ro', label='Truth')
#plt.ylabel('$y, p(y=1)$')
#plt.xlabel('$x$')
#plt.legend()

#m.plot_f()
#plt.plot(X, f, 'bo', label='Truth')
#plt.ylabel('$f(x)$')
#plt.xlabel('$x$')
#plt.legend()
#plt.show()

