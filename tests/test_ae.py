import unittest
import numpy as np
import os
from dama.data.ds import Data
from dama.drivers.core import HDF5
from dama.utils.model_selection import CV
from dama.ae.extended.w_keras import PTsne
from dama.models import MetadataX
from dama.utils.tf_functions import TSNe
from dama.data.it import BatchIterator
from dama.utils.files import check_or_create_path_dir
from dama.utils.core import Chunks


TMP_PATH = check_or_create_path_dir(os.path.dirname(os.path.abspath(__file__)), 'dama_data_test')


class TestUnsupervicedModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        x = np.random.rand(120)
        self.x = np.sin(6 * x).reshape(-1, 1)
        self.hash = None

    def tearDown(self):
        pass

    def train(self, ae, model_params=None):
        with Data(name="tsne", driver=HDF5(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as dataset, \
                Data(name="test", driver=HDF5(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as ds:
            batch_size = 12
            tsne = TSNe(batch_size=batch_size, perplexity=ae.perplexity, dim=2)
            x_p = BatchIterator.from_batchs(tsne.calculate_P(self.x), length=len(self.x), from_batch_size=tsne.batch_size,
                                       dtypes=np.dtype([("x", np.dtype(float)), ("y", np.dtype(float))]), to_slice=True)
            dataset.from_data(x_p)
            cv = CV(group_data="x", group_target="y", train_size=.7, valid_size=.1)
            stc = cv.apply(dataset)
            ds.from_data(stc)
            ae.train(ds, num_steps=5, data_train_group="train_x", target_train_group='train_y', batch_size=tsne.batch_size,
                      data_test_group="test_x", target_test_group='test_y', model_params=model_params,
                      data_validation_group="validation_x", target_validation_group="validation_y")
            ae.save("tsne", path=TMP_PATH, model_version="1")
            dataset.destroy()
        return ae

    def test_parametric_tsne(self):
        ae = self.train(PTsne(metadata_path=TMP_PATH), model_params=None)
        metadata = MetadataX.get_metadata(ae.path_metadata_version)
        self.assertEqual(len(metadata["train"]["model_json"]) > 0, True)
        with Data(name="test0", driver=HDF5(mode="w", path=TMP_PATH), metadata_path=TMP_PATH) as dataset:
            dataset.from_data(self.x)

        with PTsne.load(model_version="1", model_name="tsne", path=TMP_PATH, metadata_path=TMP_PATH) as ae,\
                Data(name="test0", driver=HDF5(mode="r", path=TMP_PATH),
                     chunks=Chunks({"g0": (10, 1)}), metadata_path=TMP_PATH) as dataset:
            self.assertEqual(ae.predict(dataset).shape, (120, 2))
            ae.destroy()
            dataset.destroy()
            self.hash = ae.ds.hash

    def test_P(self):
        tsne = TSNe(batch_size=12, perplexity=30, dim=2)
        x_p = BatchIterator.from_batchs(tsne.calculate_P(self.x), length=len(self.x), from_batch_size=tsne.batch_size,
                                        dtypes=np.dtype([("x", np.dtype(float)), ("y", np.dtype(float))]))
        for i, e in enumerate(x_p):
            if i < 10:
                self.assertEqual(e[0].shape, (12, 1))
                self.assertEqual(e[1].shape, (12, 12))
            else:
                self.assertEqual(e[0].shape, (4, 1))
                self.assertEqual(e[1].shape, (4, 12))


if __name__ == '__main__':
    unittest.main()
