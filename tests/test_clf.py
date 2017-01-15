import unittest
import numpy as np


def sizes(seq):
    return [len(list(row)) for row in seq]


class TestSeq(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(10, 10)

    def test_grouper_chunk_3(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(3, self.X)
        self.assertEqual(sizes(seq), [3, 3, 3, 1])

    def test_grouper_chunk_2(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(2, self.X)
        self.assertEqual(sizes(seq), [2, 2, 2, 2, 2])

    def test_grouper_chunk_10(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(10, self.X)
        self.assertEqual(sizes(seq), [10])

    def test_grouper_chunk_1(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(1, self.X)
        self.assertEqual(sizes(seq), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_grouper_chunk_7(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(7, self.X)
        self.assertEqual(sizes(seq), [7, 3])


class TestClf(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.extended.w_sklearn import RandomForest

        X = np.asarray([1, 0]*10)
        Y = X*1
        self.dataset = DataSetBuilder(name="test", dataset_path="/tmp/", ltype='int')
        self.dataset.build_dataset(X, Y)
        self.classif = RandomForest(dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif.train(num_steps=1)

    def test_only_is(self):
        self.assertEqual(self.classif.erroneous_clf(), None)
        c, _ , _ = self.classif.correct_clf()
        self.assertEqual(list(c.reshape(-1, 1)), [1, 0, 0, 1])

    def test_load_meta(self):
        self.assertEqual(type(self.classif.load_meta()), type({}))


class TestGrid(unittest.TestCase):
    def setUp(self):
        from ml.ds import DataSetBuilder
        from ml.clf.ensemble import Grid
        from ml.clf.extended.w_sklearn import RandomForest
        from ml.clf.extended.w_tflearn import MLP

        X = np.asarray([1, 0]*10)
        Y = X*1
        self.dataset = DataSetBuilder("test", dataset_path="/tmp/")
        self.dataset.build_dataset(X, Y)

        self.classif = Grid({0: [RandomForest, MLP]},
            dataset=self.dataset, 
            model_name="test", 
            model_version="1",
            check_point_path="/tmp/")
        self.classif.train(num_steps=1)

    def test_load_meta(self):
        self.assertEqual(type(self.classif.load_meta()), type({}))

if __name__ == '__main__':
    unittest.main()
