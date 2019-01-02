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
    def train(self, ae, model_params=None):
        np.random.seed(0)
        x = np.random.rand(100)
        x = np.sin(6 * x).reshape(-1, 1)
        dataset = Data(name="tsne", dataset_path="/tmp", driver=HDF5(), clean=True)
        tsne = TSNe(batch_size=3, perplexity=ae.perplexity, dim=2)
        x_p = BatchIterator.from_batchs(tsne.calculate_P(x), length=len(x), from_batch_size=tsne.batch_size,
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
            ae.save("test", path="/tmp/", model_version="1")
        dataset.destroy()
        return ae

    def test_parametric_tsne(self):
        ae = self.train(PTsne(), model_params=None)
        #self.assertEqual(len(clf.scores2table().measures[0]), 7)
        metadata = Metadata.get_metadata(ae.path_metadata, ae.path_metadata_version)
        print(metadata)
        #self.assertEqual(len(metadata["train"]["model_json"]) > 0, True)
        #clf.destroy()
        #classif = PTsne(model_name="tsne",
        #    check_point_path="/tmp/", latent_dim=2)
        #classif.set_dataset(dataset)
        #classif.train(batch_size=8, num_steps=2)
        #classif.save(model_version="1")

        #classif = PTsne(model_name="tsne", check_point_path="/tmp/")
        #classif.load(model_version="1")
        #self.assertEqual(classif.predict(X[:1]).shape, (None, 2))
        ae.destroy()


class TestAE(unittest.TestCase):
    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def test_vae(self):
        from ml.ae.extended.w_keras import VAE

        X = np.random.rand(1000, 10)
        X = (X * 10) % 2
        X = X.astype(int)
        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data(X)

        vae = VAE( 
            model_name="test", 
            check_point_path="/tmp/",
            intermediate_dim=5)
        vae.set_dataset(dataset)
        vae.train(batch_size=1, num_steps=10)
        vae.save(model_version="1")

        vae = VAE( 
            model_name="test",
            check_point_path="/tmp/")
        vae.load(model_version="1")
        encoder = vae.encode(X[0:1], chunks_size=10)
        decoder = vae.predict(X[0:1], chunks_size=10)
        self.assertEqual(encoder.shape, (None, 2))
        self.assertEqual(decoder.shape, (None, 10))
        dataset.destroy()
        vae.destroy()

    def test_dae(self):
        from ml.ae.extended.w_keras import SAE
        X = np.random.rand(1000, 10)
        X = (X * 10) % 2
        X = X.astype(int)
        dataset = Data(name="test", dataset_path="/tmp/", clean=True)
        with dataset:
            dataset.from_data(X)

        dae = SAE( 
            model_name="test", 
            check_point_path="/tmp/",
            latent_dim=5)
        dae.set_dataset(dataset)
        dae.train(batch_size=1, num_steps=10, num_epochs=3)
        dae.save(model_version="1")

        dae = SAE( 
            model_name="test", 
            check_point_path="/tmp/")
        dae.load(model_version="1")
        encoder = dae.encode(X[0:1], chunks_size=10)
        self.assertEqual(encoder.shape, (None, 5))
        decoder = dae.predict(X[0:1], chunks_size=10)
        self.assertEqual(decoder.shape, (None, 10))
        dataset.destroy()
        dae.destroy()


if __name__ == '__main__':
    unittest.main()
