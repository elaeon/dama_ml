import unittest
import numpy as np
import pandas as pd
from ml.layers import IterLayer
from ml.utils.numeric_functions import downsample


class TestIterLayers(unittest.TestCase):

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
        result = predictor.to_memory()
        self.assertItemsEqual(result[0], [182.25, 182.25])

    def test_operations_rscalar(self):
        data = np.zeros((20, 2)) + 1
        predictor0 = IterLayer(data, shape=(20, 2)).to_chunks(chunks_size=3)
        predictor1 = IterLayer(data, shape=(20, 2)).to_chunks(chunks_size=3)
        predictor2 = IterLayer(data, shape=(20, 2)).to_chunks(chunks_size=3)
        predictor = .6*predictor0 + .3*predictor1 + .1*predictor2
        X = predictor.flat().compose(self.multi_round, 0).to_memory()
        Y = np.zeros((40,)) + 1
        self.assertItemsEqual(X, Y)

    def test_operations_stream(self):
        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2)) + 2
        predictor_0 = IterLayer(data_0, shape=data_0.shape).to_chunks(chunks_size=3)
        predictor_1 = IterLayer(data_1, shape=data_1.shape).to_chunks(chunks_size=3)

        predictor = 4*predictor_0 + predictor_1**3
        result = predictor.to_memory()
        self.assertItemsEqual(result[0], [4, 4])

    def test_operations_list(self):
        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2)) + 1
        w = [1, 2]
        predictor_0 = IterLayer(data_0, length=20).to_chunks(chunks_size=3)
        predictor_1 = IterLayer(data_1, length=20).to_chunks(chunks_size=2)

        predictor = IterLayer([predictor_0, predictor_1], length=40)
        predictors = predictor * w
        predictors = predictors.to_memory()
        self.assertItemsEqual(predictors.reshape(-1)[:40], np.zeros((40)) - 1)
        self.assertItemsEqual(predictors.reshape(-1)[40:], np.zeros((40)) + 2)

    def test_operations(self):
        data_0 = np.zeros((20, 2)) + 1.2
        data_1 = np.zeros((20, 2)) + 1
        data_2 = np.zeros((20, 2)) + 3
        predictor_0 = IterLayer(data_0, shape=(20, 2)).to_chunks(chunks_size=3, dtype=[('x', float), ('y', float)])
        predictor_1 = IterLayer(data_1, shape=(20, 2)).to_chunks(chunks_size=3)
        predictor_2 = IterLayer(data_2, shape=(20, 2)).to_chunks(chunks_size=3)

        predictor = ((predictor_0**.65) * (predictor_1**.35) * .85) + predictor_2 * .15
        predictor = predictor.flat().compose(self.multi_round, 2).to_memory()
        self.assertItemsEqual(predictor, np.zeros((40,)) + 1.41)

    def test_raw_iter(self):
        data_0 = np.zeros((20, 3)) + 1.2
        predictor_0 = IterLayer(data_0, shape=data_0.shape)
        predictor_1 = IterLayer(data_0+1, shape=data_0.shape)
        predictor = predictor_0 + predictor_1
        predictor = predictor.flat().compose(round, 0).to_memory()
        self.assertItemsEqual(predictor, np.zeros((60,)) + 3)

    def test_avg(self):
        predictor_0 = IterLayer(np.zeros((20, 2)) + 1, shape=(20, 2), dtype='float').to_chunks(chunks_size=3)
        predictor_1 = IterLayer(np.zeros((20, 2)) + 2, shape=(20, 2), dtype='float').to_chunks(chunks_size=3)
        predictor_2 = IterLayer(np.zeros((20, 2)) + 3, shape=(20, 2), dtype='float').to_chunks(chunks_size=3)

        predictor_avg = IterLayer.avg([predictor_0, predictor_1, predictor_2], 3)
        self.assertItemsEqual(predictor_avg.to_memory().reshape(-1), np.zeros((40,)) + 2)

        predictor_0 = IterLayer(np.zeros((20, 2)) + 1, shape=(20, 2), dtype='float').to_chunks(chunks_size=3)
        predictor_1 = IterLayer(np.zeros((20, 2)) + 2, shape=(20, 2), dtype='float').to_chunks(chunks_size=3)
        predictor_2 = IterLayer(np.zeros((20, 2)) + 3, shape=(20, 2), dtype='float').to_chunks(chunks_size=3)

        predictor_avg = IterLayer.avg([predictor_0, predictor_1, predictor_2], 3, method="geometric")
        predictor_avg = predictor_avg.flat().compose(self.multi_round, 2).to_memory()
        self.assertItemsEqual(predictor_avg, np.zeros((40,)) + 1.82)

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
        predictor = IterLayer(np.zeros((20, 2)) + 1.6, shape=(20, 2), dtype="float").to_chunks(chunks_size=3)
        predictor = predictor.flat().compose(self.multi_round, 0).to_memory()
        self.assertItemsEqual(predictor, np.zeros((40,)) + 2)

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
        fl = np.concatenate((l0.reshape(-1), l1.reshape(-1), l2.reshape(-1)))
        predictor_0 = IterLayer(l0, shape=(20, 2)).to_chunks(chunks_size=3)
        predictor_1 = IterLayer(l1, shape=(20, 2)).to_chunks(chunks_size=3)
        predictor_2 = IterLayer(l2, shape=(20, 2)).to_chunks(chunks_size=3)

        predictor = IterLayer.concat_n([predictor_0, predictor_1, predictor_2])
        self.assertItemsEqual(predictor.flat().to_memory(), fl)

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

    def test_flat_shape(self):
        data = np.random.rand(10, 1)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.flat()
        data_flat = it_flat.to_memory()
        self.assertEqual(data_flat.shape, (10,))

        data = np.random.rand(10, 5)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.flat()
        data_flat = it_flat.to_memory()
        self.assertEqual(data_flat.shape, (50,))

        data = np.random.rand(10, 2 ,2)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.flat()
        data_flat = it_flat.to_memory()
        self.assertEqual(data_flat.shape, (40,))

        data = np.random.rand(1000, 2 ,2)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype)
        it_flat = it.to_chunks(chunks_size=100).flat()
        data_flat = it_flat.to_memory()
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
        self.assertEqual(it_0.shape_w_chunks, (chunks_size, (10/3)+1, 1))
        for smx in it_0:
            self.assertEqual(smx.shape[0] <= 3, True)

    def test_chunks_obj(self):
        chunks_size = 3
        m = [[1, '5.0'], [2, '3'], [4, 'C']]
        data = pd.DataFrame(m, columns=['A', 'B'])
        it = IterLayer(data, shape=data.shape)
        it_0 = it.to_chunks(chunks_size)
        self.assertEqual(it_0.chunks_size, chunks_size)
        self.assertEqual(it_0.has_chunks, True)
        self.assertEqual(it_0.shape_w_chunks, (chunks_size, 1, 2))
        for i, smx in enumerate(it_0):
            for i, row in enumerate(smx.values):
                self.assertItemsEqual(row, m[i])

    def test_from_chunks(self):
        data = np.random.rand(10, 1)
        chunks_size = 2
        it = IterLayer(data, shape=(10, 1))
        for smx in it.to_chunks(chunks_size):
            self.assertEqual(smx.shape[0], 2)

    def test_chunks_to_array(self):
        chunks_size = 3
        data = np.random.rand(10, 1)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype).to_chunks(chunks_size)
        self.assertItemsEqual(it.to_memory(), data)

        chunks_size = 0
        data = np.random.rand(10, 1)
        it = IterLayer(data, shape=data.shape, dtype=data.dtype).to_chunks(chunks_size)
        self.assertItemsEqual(it.to_memory(), data)

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
        self.assertItemsEqual(it.to_memory(), data)

        data = np.asarray([1,2,3,'4',5,6,7,8,9], dtype='|O')
        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size, dtype="|O")
        self.assertEqual(it.dtype, np.dtype('|O'))
        self.assertEqual(it.global_dtype, np.dtype('|O'))
        self.assertItemsEqual(it.to_memory(), data)
        
    def test_chunk_taste_2(self):
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = IterLayer(data, shape=data.shape)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))

        chunks_size = 3
        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size=chunks_size)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))

        data = pd.DataFrame(data, columns=['x', 'y'])
        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size)
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertEqual(isinstance(it.to_df(), pd.DataFrame), True)

    def test_chunk_df(self):
        chunks_size = 2
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = IterLayer(data, shape=data.shape, dtype=[('x', 'int'), ('y', 'float')]).to_chunks(chunks_size)
        self.assertEqual(isinstance(it.to_memory(), pd.DataFrame), True)

    def test_type_elem(self):
        iter_ = (float(i) for i in range(10))
        it = IterLayer(iter_, shape=(10,))
        self.assertEqual(it.type_elem, float)

        iter_ = ((i,) for i in range(10))
        it = IterLayer(iter_, shape=(10,))
        self.assertEqual(it.type_elem, tuple)
        
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = IterLayer(data, shape=data.shape)
        self.assertEqual(it.type_elem, np.ndarray)

        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = IterLayer(data, length=data.shape[0]).to_chunks(chunks_size=2, dtype=[('x', 'int'), ('y', 'int')])
        self.assertEqual(it.type_elem, pd.DataFrame)

        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size=2, dtype='int')
        self.assertEqual(it.type_elem, np.ndarray)

        it = IterLayer(data, shape=data.shape).to_chunks(chunks_size=2)
        self.assertEqual(it.type_elem, np.ndarray)

    def test_flat(self):
        result = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]
        iter_ = ((i,i+1) for i in range(10))
        it = IterLayer(iter_, shape=(10, 2))
        self.assertItemsEqual(it.flat().to_memory(), result)

        iter_ = ((i,i+1) for i in range(10))
        it = IterLayer(iter_, shape=(10, 2))
        self.assertItemsEqual(it.to_chunks(chunks_size=3).flat().to_memory(), result)

        iter_ = ((i,i+1) for i in range(10))
        it = IterLayer(iter_, shape=(10,2))
        self.assertItemsEqual(
            it.to_chunks(chunks_size=3, dtype=[('x', int), ('y', int)]).flat().to_memory(), result)

        iter_ = ((i,i+1) for i in range(10))
        it = IterLayer(iter_, shape=(10,2))
        self.assertItemsEqual(it.flat().to_chunks(chunks_size=3, dtype=int).to_memory(), result)

        result = range(10)
        iter_ = ((i,) for i in range(10))
        it = IterLayer(iter_, shape=(10,))
        self.assertItemsEqual(it.flat().to_memory(), result)

    def test_clean_chunks(self):
        it = IterLayer(((i, 'X', 'Z') for i in range(20)), shape=(20, 3))
        chunks_size = 2
        it_0 = it.to_chunks(chunks_size=chunks_size)
        for smx in it_0.clean_chunks():
            self.assertEqual(smx.shape[0] <=3, True)

    def test_sample(self):
        order = (i for i in range(20))
        it = IterLayer(order, shape=(20,))
        it_0 = it.to_chunks(chunks_size=2)
        it_s = it_0.sample(5)
        self.assertEqual(len(it_s.to_memory()), 5)

    def test_gen_weights(self):
        order = (i for i in range(4))
        it = IterLayer(order, shape=(4,))
        it_0 = it.weights_gen(it, None, lambda x: x%2+1)
        self.assertItemsEqual(list(it_0), [(0, 1), (1, 2), (2, 1), (3, 2)])

        def fn(v):
            if v == 0:
                return 1
            else:
                return 99

        data = np.zeros((20, 4)) + [1,2,3,0]
        data[:, 3] = np.random.rand(1,20) > .5
        it = IterLayer(data, shape=(20, 4))
        w_v = list(it.weights_gen(it, 3, fn))
        self.assertEqual(w_v[0][1], fn(data[0][3]))

    def test_sample_weight(self):
        import collections
        def fn(v):
            if v == 0:
                return 1
            else:
                return 90

        num_items = 200000
        num_samples = 20000
        data = np.zeros((num_items, 4)) + [1, 2, 3, 0]
        data[:, 3] = np.random.rand(1, num_items) > .5
        it = IterLayer(data, shape=(num_items, 4))
        it_s = it.sample(num_samples, col=3, weight_fn=fn)
        c = collections.Counter(it_s.to_memory()[:, 3])
        self.assertEqual(c[1]/float(num_samples) > .79, True)

    def test_split(self):
        it = IterLayer(((i, 'X', 'Z') for i in range(20)), shape=(20,))
        it0, it1 = it.split(2)
        self.assertItemsEqual(it0.to_memory()[0], [0, 'X'])
        self.assertItemsEqual(it1.flat().to_memory(), np.asarray(['Z']*20))

        it = IterLayer(((i, 'X', 'Z') for i in range(20)), shape=(20,))
        it_0, it_1 = it.split(2)
        self.assertItemsEqual(it_0.to_chunks(2).to_memory()[0], [0, 'X'])
        self.assertItemsEqual(it_1.to_chunks(2).flat().to_memory(), np.asarray(['Z']*20))

        data = ((i, 'X'+str(i), 'Z') for i in range(20))
        it = IterLayer(data, shape=(20, 3), 
            dtype=[('A', 'int'), ('B', '|O'), ('C', '|O')])
        it_0, it_1 = it.to_chunks(chunks_size=2).split(2)
        self.assertItemsEqual(it_0.to_memory().iloc[0, :].values, [0, 'X0'])
        self.assertItemsEqual(it_1.flat().to_memory(), np.asarray(['Z']*20))

        data = ((i, 'X', 'Z') for i in range(20))
        it = IterLayer(data, shape=(20, 3), 
            dtype=[('A', 'int'), ('B', 'str'), ('C', 'str')])
        self.assertItemsEqual(it.split(2)[0].to_memory().iloc[0, :], [0, 'X'])

    def test_raw(self):
        data = [1, 2, 3, 4, 5]
        it = IterLayer(data, shape=(len(data),))
        self.assertItemsEqual(it.to_memory(), data)

        data = np.random.rand(2, 3)
        it = IterLayer(data, shape=data.shape)
        for i, row in enumerate(it.to_memory()):
            self.assertItemsEqual(row, data[i])

        data = [[1,2], [3,4]]
        it = IterLayer(data, shape=(len(data), 2))
        for i, row in enumerate(it.to_memory()):
            self.assertItemsEqual(row, data[i])

    def test_downsample(self):
        size = 5000
        data = np.random.rand(size, 3)
        data[:, 2] = data[:, 2] <= .9
        v = downsample(data, {0: 200, 1:240}, 2, size)
        true_values = count_true_values(v.to_memory(), 2)
        self.assertEqual(true_values[0] > 50, True)
        self.assertEqual(true_values[1], 240)


    def test_downsample_small(self):
        size = 10
        data = np.random.rand(size, 3)
        data[:, 2] = data[:, 2] <= .9
        v = downsample(data, {0: 0, 1:3}, 2, size)
        self.assertItemsEqual(v.to_memory()[:, 2], [1,1,1])
        #v = downsample(data, {0: 20, 1:3}, 2, 10)
        #self.assertEqual(v.to_memory()

    def test_downsample_static(self):
        data = [0,0,0,1,1,1,1,1,2,2,2]
        size = len(data)
        v = downsample(data, {0: 2, 1: 4}, None, size)
        self.assertItemsEqual(v.to_memory(), [0,0,1,1,1,1])
        v = downsample(data, {1: 4}, None, size)
        self.assertItemsEqual(v.to_memory(), [1,1,1,1])
        v = downsample(data, {2: 2, 1: 4}, None, size)
        self.assertItemsEqual(v.to_memory(), [2,2,1,1,1,1])

    def test_empty(self):
        self.assertEqual(IterLayer([], shape=(0,)).to_memory().shape, (0,))

    def test_columns(self):
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        data = pd.DataFrame(data, columns=['x', 'y'])
        it = IterLayer(data, shape=data.shape)
        self.assertItemsEqual(it.columns(), ['x', 'y'])


def chunk_sizes(seq):
    return [len(list(row)) for row in seq]


def count_true_values(data, y):
    true_values = len([e for e in data[:, y] == 1 if e])
    return true_values*100 / float(data.shape[0]), true_values


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
