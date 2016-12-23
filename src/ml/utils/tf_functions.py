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


