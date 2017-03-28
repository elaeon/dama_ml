import unittest
import numpy as np


class TestAE(unittest.TestCase):
    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def test_parametric_tsne(self):
        from ml.ds import Data
        from ml.ae.extended.w_keras import PTsne

        dataset = Data(
            name="tsne", 
            dataset_path="/tmp/")
        X = np.random.rand(100, 10)
        Y = np.sin(6*X)
        dataset.build_dataset(Y)
        classif = PTsne(model_name="tsne", model_version="1", 
            check_point_path="/tmp/", dataset=dataset, latent_dim=2)
        classif.train(batch_size=8, num_steps=2)

        classif = PTsne(model_name="tsne", model_version="1", 
            check_point_path="/tmp/")
        self.assertEqual(len(list(classif.predict([X[1]]))[0]), 2)

    def test_vae(self):
        from ml.ae.extended.w_keras import VAE
        from ml.ds import Data

        X = np.random.rand(1000, 10)
        X = (X * 10) % 2
        X = X.astype(int)
        dataset = Data(name="test", dataset_path="/tmp/", rewrite=True)
        dataset.build_dataset(X)

        vae = VAE(dataset=dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/",
            intermediate_dim=5)
        vae.train(batch_size=1, num_steps=10)

        vae = VAE( 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        encoder = np.asarray(list(vae.predict(X[0:1], chunk_size=10, model_type="encoder")))
        decoder = np.asarray(list(vae.predict(X[0:1], chunk_size=10, model_type="decoder")))
        self.assertEqual(encoder.shape, (1, 2))
        self.assertEqual(decoder.shape, (1, 10))

    def test_ensemble(self):
        from ml.clf.ensemble import Bagging
        from ml.ae.extended.w_keras import PTsne
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.ds import DataSetBuilder

        X = np.asarray([1, 0]*1000)
        Y = X*1
        dataset = DataSetBuilder("test_ae_ensemble", dataset_path="/tmp/", rewrite=False)
        dataset.build_dataset(X, Y)
        bagging = Bagging(RandomForest, {"0": [PTsne]},
                dataset=dataset,
                model_name="test_bag",
                model_version="1")
        bagging.train(batch_size=50, num_steps=1)
        bagging.scores().print_scores()
        dataset.destroy()

if __name__ == '__main__':
    unittest.main()
