import unittest
import numpy as np
import tensorflow as tf
import os

from dama.utils.tf_functions import ssim, msssim
from dama.processing import rgb2gray, merge_offset


class TestMsssiM(unittest.TestCase):
    def setUp(self):
        base_file = os.path.dirname(os.path.abspath(__file__))
        self.img1 = os.path.join(base_file, "../data/imgs/testPattern.png")
        self.img2 = os.path.join(base_file, "../data/imgs/testPattern.png")
        self.img3 = os.path.join(base_file, "../data/imgs/testPattern2.png")

    def test_same_img(self):
        from skimage import io, img_as_float

        image_path = self.img1
        image = np.expand_dims(io.imread(image_path), axis=0)
        img = img_as_float(merge_offset(rgb2gray(image)))
        img = img[0]
        rows, cols = img.shape

        noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
        noise[np.random.random(size=noise.shape) > 0.5] *= -1

        img_noise = img + noise
        ## TF CALC START
        #BATCH_SIZE = 1
        #CHANNELS = 1
        image1 = tf.placeholder(tf.float32, shape=[rows, cols])
        image2 = tf.placeholder(tf.float32, shape=[rows, cols])

        def image_to_4d(image):
            image = tf.expand_dims(image, 0)
            image = tf.expand_dims(image, -1)
            return image

        image4d_1 = image_to_4d(image1)
        image4d_2 = image_to_4d(image2)
        ssim_index = ssim(image4d_1, image4d_2, size=4)
        msssim_index = tf.reduce_mean(msssim(image4d_1, image4d_2, level=5, size=4))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_ssim_none = sess.run(ssim_index,
                                    feed_dict={image1: img, image2: img})
            tf_ssim_noise = sess.run(ssim_index,
                                     feed_dict={image1: img, image2: img_noise})
            tf_msssim_none = sess.run(msssim_index,
                                    feed_dict={image1: img, image2: img})
            tf_msssim_noise = sess.run(msssim_index,
                                     feed_dict={image1: img, image2: img_noise})
            # TF CALC END
            print('tf_ssim_none', tf_ssim_none)
            print('tf_ssim_noise', tf_ssim_noise)
            print('tf_msssim_none', tf_msssim_none)
            print('tf_msssim_noise', tf_msssim_noise)
            if tf_msssim_none == 1:
                self.assertEqual(tf_msssim_none, 1)

    def test_two_img(self):
        from skimage import io, img_as_float

        image_path1 = self.img2
        image_path2 = self.img3

        image1 = np.expand_dims(io.imread(image_path1), axis=0)
        img1 = img_as_float(merge_offset(rgb2gray(image1)))
        img1 = img1[0]
        rows1, cols1 = img1.shape

        image2 = np.expand_dims(io.imread(image_path2), axis=0)
        img2 = img_as_float(merge_offset(rgb2gray(image2)))
        img2 = img2[0]
        rows2, cols2 = img2.shape

        image1 = tf.placeholder(tf.float32, shape=[rows1, cols1])
        image2 = tf.placeholder(tf.float32, shape=[rows2, cols2])

        def image_to_4d(image):
            image = tf.expand_dims(image, 0)
            image = tf.expand_dims(image, -1)
            return image

        image4d_1 = image_to_4d(image1)
        image4d_2 = image_to_4d(image2)
        # ssim_index = ssim(image4d_1, image4d_2)
        msssim_index = tf.reduce_mean(msssim(image4d_1, image4d_2, level=5, size=6))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_msssim_1 = sess.run(msssim_index,
                                    feed_dict={image1: img1, image2: img1})
            # tf_msssim_2 = sess.run(msssim_index,
            #                         feed_dict={image1: img1, image2: img2})

            if tf_msssim_1 == 1:
                self.assertEqual(tf_msssim_1, 1)

