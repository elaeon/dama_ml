import unittest
import numpy as np
import pandas as pd
from ml.layers import IterLayer


class TestIterLayers(unittest.TestCase):

    def predict(self, data):
        for e in data:
            yield e + 1 

    def chunks(self, data, chunk_size=2):
        from ml.utils.seq import grouper_chunk
        for chunk in grouper_chunk(chunk_size, data):
            for p in self.predict(chunk):
                yield p

    def multi_round(self, X, *args):
        return np.asarray([round(x, *args) for x in X])

    def test_operations_lscalar(self):
        data = np.zeros((20, 2))
        predictor = IterLayer(data, shape=data.shape, dtype=data.dtype).to_chunks(chunks_size=2)
        predictor += 10.
        predictor -= 1.
        predictor *= 3.
        predictor /= 2.
        predictor **= 2
        result = predictor.to_narray()
        self.assertItemsEqual(result[0], [182.25, 182.25])

    def test_operations_rscalar(self):

        data = np.zeros((20, 2))
        predictor0 = IterLayer(self.chunks(data))
        predictor1 = IterLayer(self.chunks(data))
        predictor2 = IterLayer(self.chunks(data))
        predictor = .6*predictor0 + .3*predictor1 + .1*predictor2
        X = np.asarray(list(predictor)).reshape(-1).round(decimals=0)
        Y = np.zeros((40,)) + 1
        self.assertItemsEqual(X, Y)

    def test_operations_stream(self):

        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2)) + 2
        predictor_0 = IterLayer(data_0, shape=data_0.shape, dtype=data_0.dtype).to_chunks(chunks_size=3)
        predictor_1 = IterLayer(data_1, shape=data_1.shape, dtype=data_1.dtype).to_chunks(chunks_size=3)

        predictor = 4*predictor_0 + predictor_1**3
        result = predictor.to_narray()
        self.assertItemsEqual(result[0], [4, 4])

    def test_operations_list(self):
        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2)) + 1
        w = [1, 2]
        predictor_0 = IterLayer(data_0, length=20).to_chunks(chunks_size=3)
        predictor_1 = IterLayer(data_1, length=20).to_chunks(chunks_size=2)

        predictor = IterLayer([predictor_0, predictor_1], length=40)
        predictors = predictor * w
        predictors = predictors.to_narray()
        self.assertItemsEqual(predictors.reshape(-1)[:40], np.zeros((40)) - 1)
        self.assertItemsEqual(predictors.reshape(-1)[40:], np.zeros((40)) + 2)

    def test_operations(self):

        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2))
        data_2 = np.zeros((20, 2)) + 3
        predictor_0 = IterLayer(self.chunks(data_0, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(data_1, chunk_size=2))
        predictor_2 = IterLayer(self.chunks(data_2, chunk_size=3))

        predictor = ((predictor_0**.65) * (predictor_1**.35) * .85) + predictor_2 * .15
        self.assertItemsEqual(np.asarray(list(predictor)).reshape(-1), np.zeros((40,)) + .6)

    def test_avg(self):

        predictor_0 = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3), shape=(20, 2), dtype='float')
        predictor_1 = IterLayer(self.chunks(np.zeros((20, 2)) + 2, chunk_size=3), shape=(20, 2), dtype='float')
        predictor_2 = IterLayer(self.chunks(np.zeros((20, 2)) + 3, chunk_size=3), shape=(20, 2), dtype='float')

        predictor_avg = IterLayer.avg([predictor_0, predictor_1, predictor_2], 3)
        self.assertItemsEqual(predictor_avg.to_narray().reshape(-1), np.zeros((40,)) + 3)

        predictor_0 = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3), shape=(20, 2), dtype='float')
        predictor_1 = IterLayer(self.chunks(np.zeros((20, 2)) + 2, chunk_size=3), shape=(20, 2), dtype='float')
        predictor_2 = IterLayer(self.chunks(np.zeros((20, 2)) + 3, chunk_size=3), shape=(20, 2), dtype='float')

        predictor_avg = IterLayer.avg([predictor_0, predictor_1, predictor_2], 3, method="geometric")
        predictor_avg = predictor_avg.compose(self.multi_round, 2)
        self.assertItemsEqual(predictor_avg.to_narray().reshape(-1), np.zeros((40,)) + 2.88)

    def test_max_counter(self):

        predictor_0 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"], shape=(8,), dtype="int")
        predictor_1 = IterLayer(["1", "2", "2", "1", "2", "0", "0", "0"], shape=(8,), dtype="int")
        predictor_2 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"], shape=(8,), dtype="int")
        predictor_avg = IterLayer.max_counter([predictor_0, predictor_1, predictor_2])
        self.assertEqual(list(predictor_avg), ['0', '1', '0', '1', '2', '0', '1', '2'])

        weights = [1.5, 2, 1]
        predictor_0 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"], shape=(8,), dtype="int")
        predictor_1 = IterLayer(["1", "2", "2", "1", "2", "0", "0", "0"], shape=(8,), dtype="int")
        predictor_2 = IterLayer(["0", "1", "0", "1", "2", "0", "1", "2"], shape=(8,), dtype="int")        
        predictor_avg = IterLayer.max_counter([predictor_0, predictor_1, predictor_2], weights=weights)
        self.assertEqual(list(predictor_avg), ['0', '1', '0', '1', '2', '0', '1', '2'])

    def test_custom_fn(self):

        predictor = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3), shape=(20, 2), dtype="int")
        predictor = predictor.compose(self.multi_round, 2)
        self.assertItemsEqual(predictor.to_narray().reshape(-1), np.zeros((40,)) + 2)

    def test_concat_fn(self):

        l0 = np.random.rand(10, 2)
        l1 = np.random.rand(10, 2)
        predictor_0 = IterLayer(l0)
        predictor_1 = IterLayer(l1)
        predictor = predictor_0.concat(predictor_1)
        self.assertEqual(np.asarray(list(predictor)).shape, (20, 2))

    def test_concat_n(self):

        l0 = np.zeros((20, 2)) + 1
        l1 = np.zeros((20, 2)) + 2
        l2 = np.zeros((20, 2)) + 3
        fl = np.concatenate((l0.reshape(-1) + 1, l1.reshape(-1) + 1, l2.reshape(-1) + 1))
        predictor_0 = IterLayer(self.chunks(np.zeros((20, 2)) + 1, chunk_size=3))
        predictor_1 = IterLayer(self.chunks(np.zeros((20, 2)) + 2, chunk_size=3))
        predictor_2 = IterLayer(self.chunks(np.zeros((20, 2)) + 3, chunk_size=3))

        predictor = IterLayer.concat_n([predictor_0, predictor_1, predictor_2])
        self.assertItemsEqual(np.asarray(list(predictor)).reshape(-1), fl)

    def test_append_data_to_iter(self):

        data = [[0, 1, 0], [2, 3, 0], [4, 5, 0], [5, 6, 0]]
        data_i = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        iter_layer = IterLayer((e for e in data_i))
        iter_ce = iter_layer.concat_elems(data)

        for i, e in enumerate(iter_ce):
            self.assertItemsEqual(list(e), data_i[i] + data[i])

    def test_append_iter_to_iter(self):

        data_i2 = [[0, 1, 0], [2, 3, 0], [4, 5, 0], [5, 6, 0]]
        data_i1 = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        iter_layer_1 = IterLayer((e for e in data_i1))
        iter_layer_2 = IterLayer((e for e in data_i2))
        iter_ce = iter_layer_1.concat_elems(iter_layer_2)

        for i, e in enumerate(iter_ce):
            self.assertItemsEqual(list(e), data_i1[i] + data_i2[i])

    def test_flat(self):
        data = np.random.rand(10, 1)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.flat()
        data_flat = it_flat.to_narray()
        self.assertEqual(data_flat.shape, (10,))

        data = np.random.rand(10, 5)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.flat()
        data_flat = it_flat.to_narray()
        self.assertEqual(data_flat.shape, (50,))

        data = np.random.rand(10, 2 ,2)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.flat()
        data_flat = it_flat.to_narray()
        self.assertEqual(data_flat.shape, (40,))

        data = np.random.rand(1000, 2 ,2)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.to_chunks(chunks_size=100).flat()
        data_flat = it_flat.to_narray()
        self.assertEqual(data_flat.shape, (4000,))

    def test_shape(self):
        data = np.random.rand(10, 3)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        self.assertEqual(it.shape, (10, 3))
        it_c = it.to_chunks(4)
        self.assertEqual(it_c.shape, (10, 3))
        self.assertEqual(it_c.shape_w_chunks, (4, 3, 3))

    def test_chunks(self):
        chunks_size = 3
        data = np.random.rand(10, 1)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_0 = it.to_chunks(chunks_size)
        self.assertEqual(it_0.chunks_size, chunks_size)
        self.assertEqual(it_0.has_chunks, True)
        for smx in it_0:
            self.assertEqual(smx.shape[0] <= 3, True)

    def test_from_chunks(self):
        data = np.random.rand(10, 1)
        chunks_size = 3
        it = IterLayer(self.chunks(data, chunk_size=chunks_size), 
            has_chunks=True, chunks_size=chunks_size)
        for smx in it.to_chunks(chunks_size):
            self.assertEqual(smx.shape[0], 1)

    def test_chunks_to_array(self):
        chunks_size = 3
        data = np.random.rand(10, 1)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype).to_chunks(chunks_size)
        self.assertItemsEqual(it.to_narray(), data)

        chunks_size = 0
        data = np.random.rand(10, 1)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype).to_chunks(chunks_size)
        self.assertItemsEqual(it.to_narray(), data)

    def test_df_chunks(self):
        chunks_size = 3
        data = np.asarray([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]], dtype='float')
        it = IterLayer(data, shape=data.shape, dtype=[('x', 'float'), ('y', 'float')]).to_chunks(chunks_size)
        chunk = next(it)
        self.assertItemsEqual(chunk.x, [0, 2, 4])
        self.assertItemsEqual(chunk.y, [1, 3, 5])

        chunks_size = 2
        data = np.asarray([1,2,3,4,5,6,7,8,9], dtype='float')
        it = IterLayer(data, shape=data.shape, dtype=[('x', 'float')]).to_chunks(chunks_size)
        chunk = next(it)
        self.assertItemsEqual(chunk['x'].values, [1, 2])

    def test_chunk_taste(self):
        chunks_size = 2
        data = np.asarray([1,2,3,4,5,6,7,8,9], dtype='int')
        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertItemsEqual(it.to_narray(), data)

        data = np.asarray([1,2,3,'4',5,6,7,8,9], dtype='|O')
        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size)
        self.assertEqual(it.dtype, np.dtype('|O'))
        self.assertEqual(it.global_dtype, np.dtype('|O'))
        self.assertItemsEqual(it.to_narray(), data)
        
    def test_chunk_taste_2(self):
        chunks_size = 3
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertItemsEqual(it.flat().to_narray(), data.reshape(-1))

        data = pd.DataFrame(data, columns=['x', 'y'])
        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size)
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertEqual(isinstance(it.to_df(), pd.DataFrame), True)

    def test_chunk_df(self):
        chunks_size = 2
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = IterLayer(data, shape=data.shape, dtype=[('x', 'int'), ('y', 'float')]).to_chunks(chunks_size)
        self.assertEqual(isinstance(it.to_memory(), pd.DataFrame), True)


def chunk_sizes(seq):
    return [len(list(row)) for row in seq]


class TestSeq(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(10, 10)

    def test_grouper_chunk_3(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(3, self.X)
        self.assertEqual(chunk_sizes(seq), [3, 3, 3, 1])

    def test_grouper_chunk_2(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(2, self.X)
        self.assertEqual(chunk_sizes(seq), [2, 2, 2, 2, 2])

    def test_grouper_chunk_10(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(10, self.X)
        self.assertEqual(chunk_sizes(seq), [10])

    def test_grouper_chunk_1(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(1, self.X)
        self.assertEqual(chunk_sizes(seq), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_grouper_chunk_7(self):
        from ml.utils.seq import grouper_chunk
        seq = grouper_chunk(7, self.X)
        self.assertEqual(chunk_sizes(seq), [7, 3])
