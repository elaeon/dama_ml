### this code was copied from http://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
### and veryfied with the code in http://mubeta06.github.io/python/sp/_modules/sp/ssim.html
###
###

import tensorflow as tf
import numpy as np

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def ssim(img1, img2, cs_map=False, size=11, sigma=1.5):
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    return value

#Multi scale structural similarity image
def msssim(img1, img2, level=5, size=11):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map=True, size=size)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    return (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))


def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def KLdivergence(P, Y, low_dim=2):
    dtype = P.dtype
    with tf.Session() as sess:
        alpha = low_dim - 1.
        sum_Y = tf.reduce_sum(tf.square(Y), 1)
        eps = tf.Variable(10e-15, dtype=dtype, name="eps").initialized_value()
        Q = tf.reshape(sum_Y, [-1, 1]) + -2 * tf.matmul(Y, tf.transpose(Y))
        Q = sum_Y + Q / alpha
        Q = tf.pow(1 + Q, -(alpha + 1) / 2)
        #Q = Q * (1 - tf.diag(tf.ones(self.batch_size, dtype=dtype)))
        Q_d = tf.diag_part(Q)
        Q_d = Q_d - Q_d
        Q = tf.matrix_set_diag(Q, Q_d)
        Q = Q / tf.reduce_sum(Q)
        Q = tf.maximum(Q, eps)
        C = tf.log((P + eps) / (Q + eps))
        C = tf.reduce_sum(P * C)
        return C


class TSNe:
    def __init__(self, batch_size=100, perplexity=30., dim=2):
        self.batch_size = batch_size
        self.low_dim = dim
        self.perplexity = perplexity
        self.tol = 1e-4

    def x2p(self, X, max_tries=30):
        n = X.shape[0]                     # number of instances
        P = np.zeros([n, n])               # empty probability matrix
        beta = np.ones(n)                  # empty precision vector
        logU = np.log(self.perplexity)     # log of perplexity (= entropy)
        
        # Compute pairwise distances
        sum_X = np.sum(np.square(X), axis=1)
        D = sum_X + sum_X[:,None] + -2 * X.dot(X.T)

        for i in range(n):
            # Set minimum and maximum values for precision
            betamin = -np.inf
            betamax = np.inf

            # Compute the Gaussian kernel and entropy for the current precision
            indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
            Di = D[i, indices]
            H, thisP = Hbeta(Di, beta[i])
            
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while abs(Hdiff) > self.tol and tries < max_tries:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i]
                    if np.isinf(betamax):
                        beta[i] *= 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i]
                    if np.isinf(betamin):
                        beta[i] /= 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2
                
                # Recompute the values
                H, thisP = Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1
            
            # Set the final row of P
            P[i, indices] = thisP
        
        return P

    #joint_probabilities
    def calculate_P(self, X):
        print "Computing pairwise distances..."
        while 1:
            n = X.shape[0]
            P = np.zeros([n, self.batch_size])
            for i in xrange(0, n, self.batch_size):
                P_batch = self.x2p(X[i:i + self.batch_size])
                P_batch[np.isnan(P_batch)] = 0
                P_batch = (P_batch + P_batch.T) / 2.
                P_batch = P_batch / P_batch.sum()
                P_batch = np.maximum(P_batch, 1e-12)
                #P[i:i + self.batch_size] = P_batch
                yield (X[i:i + self.batch_size], P_batch)
        #return P

