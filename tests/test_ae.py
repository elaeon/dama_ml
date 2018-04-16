import unittest
import numpy as np
from ml.ds import Data


class TestUnsupervicedModel(unittest.TestCase):
    def test_parametric_tsne(self):
        from ml.ae.extended.w_keras import PTsne

        dataset = Data(
            name="tsne", 
            dataset_path="/tmp/")
        X = np.random.rand(100, 10)
        Y = np.sin(6*X)
        with dataset:
            dataset.build_dataset(Y)
        classif = PTsne(model_name="tsne", 
            check_point_path="/tmp/", latent_dim=2)
        classif.set_dataset(dataset)
        classif.train(batch_size=8, num_steps=2)
        classif.save(model_version="1")

        classif = PTsne(model_name="tsne", check_point_path="/tmp/")
        classif.load(model_version="1")
        self.assertEqual(classif.predict(X[:1]).shape, (1, 2))
        classif.destroy()
        dataset.destroy()


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
        dataset = Data(name="test", dataset_path="/tmp/", rewrite=True)
        with dataset:
            dataset.build_dataset(X)

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
        encoder = vae.predict(X[0:1], chunks_size=10, model_type="encoder")
        decoder = vae.predict(X[0:1], chunks_size=10, model_type="decoder")
        self.assertEqual(encoder.shape, (1, 2))
        self.assertEqual(decoder.shape, (1, 10))
        dataset.destroy()
        vae.destroy()

    def test_dae(self):
        from ml.ae.extended.w_keras import DAE
        X = np.random.rand(1000, 10)
        X = (X * 10) % 2
        X = X.astype(int)
        dataset = Data(name="test", dataset_path="/tmp/", rewrite=True)
        with dataset:
            dataset.build_dataset(X)

        dae = DAE( 
            model_name="test", 
            check_point_path="/tmp/",
            intermediate_dim=5)
        dae.set_dataset(dataset)
        dae.train(batch_size=1, num_steps=10)
        dae.save(model_version="1")

        dae = DAE( 
            model_name="test", 
            check_point_path="/tmp/")
        dae.load(model_version="1")
        encoder = dae.predict(X[0:1], chunks_size=10, model_type="encoder")
        self.assertEqual(encoder.shape, (1, 10))
        decoder = dae.predict(X[0:1], chunks_size=10, model_type="decoder")
        self.assertEqual(decoder.shape, (1, 10))
        dataset.destroy()
        dae.destroy()


if __name__ == '__main__':
    unittest.main()
