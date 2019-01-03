import unittest
import numpy as np
from ml.data.ds import Data
from ml.data.drivers import HDF5
from ml.utils.model_selection import CV
from ml.ae.extended.w_keras import PTsne
from ml.models import Metadata
from ml.utils.tf_functions import TSNe
from ml.data.it import BatchIterator


class TestUnsupervicedModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(100)
        self.x = np.sin(6 * x).reshape(-1, 1)

    def train(self, ae, model_params=None):
        dataset = Data(name="tsne", dataset_path="/tmp", driver=HDF5(), clean=True)
        batch_size = 12
        tsne = TSNe(batch_size=batch_size, perplexity=ae.perplexity, dim=2)
        x_p = BatchIterator.from_batchs(tsne.calculate_P(self.x), length=len(self.x), from_batch_size=tsne.batch_size,
                                       dtypes=[("x", np.dtype(float)), ("y", np.dtype(float))])
        dataset.from_data(x_p)
        cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
        with dataset:
            stc = cv.apply(dataset)
            ds = Data(name="test", dataset_path="/tmp/", driver=HDF5(), clean=True)
            ds.from_data(stc)
            ae.train(ds, num_steps=5, data_train_group="train_x", target_train_group='train_y', batch_size=tsne.batch_size,
                      data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
            ae.save("tsne", path="/tmp/", model_version="1")
        dataset.destroy()
        return ae

    def test_parametric_tsne(self):
        ae = self.train(PTsne(), model_params=None)
        metadata = Metadata.get_metadata(ae.path_metadata, ae.path_metadata_version)
        self.assertEqual(len(metadata["train"]["model_json"]) > 0, True)
        dataset = Data(name="test", dataset_path="/tmp", driver=HDF5(), clean=True)
        dataset.from_data(self.x)
        ae = PTsne.load(model_version="1", model_name="tsne", path="/tmp/")
        with dataset:
            self.assertEqual(ae.predict(dataset).shape, (np.inf, 2))
        ae.destroy()
        dataset.destroy()

    def test_P(self):
        tsne = TSNe(batch_size=12, perplexity=30, dim=2)
        x_p = BatchIterator.from_batchs(tsne.calculate_P(self.x), length=len(self.x), from_batch_size=tsne.batch_size,
                                        dtypes=[("x", np.dtype(float)), ("y", np.dtype(float))])
        for i, e in enumerate(x_p):
            if i < 8:
                self.assertEqual(e[0].shape, (12, 1))
                self.assertEqual(e[1].shape, (12, 12))
            else:
                self.assertEqual(e[0].shape, (4, 1))
                self.assertEqual(e[1].shape, (4, 12))

# class TestAE(unittest.TestCase):
#    def setUp(self):
#        pass
        
#    def tearDown(self):
#        pass

#    def test_vae(self):
#        from ml.ae.extended.w_keras import VAE

#        X = np.random.rand(1000, 10)
#        X = (X * 10) % 2
#        X = X.astype(int)
#        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
#        with dataset:
#            dataset.from_data(X)

#        vae = VAE(
#            model_name="test",
#            check_point_path="/tmp/",
#            intermediate_dim=5)
#        vae.set_dataset(dataset)
#        vae.train(batch_size=1, num_steps=10)
#        vae.save(model_version="1")

#        vae = VAE(
#            model_name="test",
#            check_point_path="/tmp/")
#        vae.load(model_version="1")
#        encoder = vae.encode(X[0:1], chunks_size=10)
#        decoder = vae.predict(X[0:1], chunks_size=10)
#        self.assertEqual(encoder.shape, (None, 2))
#        self.assertEqual(decoder.shape, (None, 10))
#        dataset.destroy()
#        vae.destroy()

#    def test_dae(self):
#        from ml.ae.extended.w_keras import SAE
#        X = np.random.rand(1000, 10)
#        X = (X * 10) % 2
#        X = X.astype(int)
#        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
#        with dataset:
#            dataset.from_data(X)

#        dae = SAE(
#            model_name="test",
#            check_point_path="/tmp/",
#            latent_dim=5)
#        dae.set_dataset(dataset)
#        dae.train(batch_size=1, num_steps=10, num_epochs=3)
#        dae.save(model_version="1")

#        dae = SAE(
#            model_name="test",
#            check_point_path="/tmp/")
#        dae.load(model_version="1")
#        encoder = dae.encode(X[0:1], chunks_size=10)
#        self.assertEqual(encoder.shape, (None, 5))
#        decoder = dae.predict(X[0:1], chunks_size=10)
#        self.assertEqual(decoder.shape, (None, 10))
#        dataset.destroy()
#        dae.destroy()


if __name__ == '__main__':
    unittest.main()
