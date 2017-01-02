import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Bernoulli
from edward.stats import bernoulli, norm


class BernoulliC:
    def __init__(self, lik_std=0.5, prior_std=0.5, D=1):
        self.lik_std = lik_std
        self.prior_std = prior_std
        self.D = D

    def log_prob(self, xs, zs):
        x, y = xs['x'], xs['y']
        print(y.eval())
        w, b = zs['w'], zs['b']
        log_prior = tf.reduce_sum(norm.logpdf(w, tf.zeros(self.D), self.prior_std))
        log_prior += tf.reduce_sum(norm.logpdf(b, tf.zeros(self.D), self.prior_std))
        log_lik = tf.reduce_sum(bernoulli.logpmf(y, p=tf.sigmoid(ed.dot(x, w) + b)))
        return log_lik + log_prior

    def predict(self, xs, zs):
        """Return a prediction for each data point, latent variables z in zs."""
        x = xs['x']
        w, b = zs['w'], zs['b']
        return ed.dot(x, w) + b


def build_dataset(N):
    DIM = 21
    SIZE = N
    X = np.random.rand(SIZE, DIM)
    Y = np.asarray([1 if sum(row) > 0 else 0 for row in np.sin(6*X) + 0.1*np.random.randn(SIZE, 1)])
    return tf.convert_to_tensor(X.astype(np.float32)), tf.convert_to_tensor(Y.astype(np.float32))


def z():
    N = 10000  # num data points
    D = 21  # num features
    x_train, y_train = build_dataset(N)
    model = BernoulliC(D=D)#LinearModel()

    qw_mu = tf.Variable(tf.random_normal([D]))
    qw_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([D])))
    qb_mu = tf.Variable(tf.random_normal([1]))
    qb_sigma = tf.nn.softplus(tf.Variable(tf.random_normal([1])))

    qw = Normal(mu=qw_mu, sigma=qw_sigma)
    qb = Normal(mu=qb_mu, sigma=qb_sigma)

    data = {'x': x_train, 'y': y_train}
    qz = {'w': qw, 'b': qb}
    inference = ed.MFVI(qz, data, model)
    inference.run(n_iter=1000, n_samples=10, n_print=None)
    x_test, y_test = build_dataset(4)
    data_test = {'x': x_test, 'y': y_test}
    print(y_test.eval())
    print(model.predict({'x': x_test}, qz).eval())
    #print(ed.evaluate('log_loss', 
    #    data=data_test, latent_vars=qz, model_wrapper=model))


#ed.set_seed(42)


if __name__ == '__main__':
    z()
