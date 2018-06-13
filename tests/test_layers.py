import unittest
import numpy as np
import pandas as pd
import datetime

from ml.layers import Iterator
from ml import ittools


class TestIterators(unittest.TestCase):

    def multi_round(self, X, *args):
        return np.asarray([round(x, *args) for x in X])

    def test_operations_lscalar(self):
        data = np.zeros((20, 2))
        predictor = Iterator(data).to_chunks(chunks_size=2)
        predictor += 10.
        predictor -= 1.
        predictor *= 3.
        predictor /= 2.
        predictor **= 2
        result = predictor.to_memory(20)
        self.assertItemsEqual(result[:, 0], np.asarray([[182.25, 182.25]]*20)[:, 0])

    def test_operations_rscalar(self):
        data = np.zeros((20, 2)) + 1
        predictor0 = Iterator(data).to_chunks(chunks_size=3)
        predictor1 = Iterator(data).to_chunks(chunks_size=3)
        predictor2 = Iterator(data).to_chunks(chunks_size=3)
        predictor = .6*predictor0 + .3*predictor1 + .1*predictor2
        X = predictor.flat().compose(self.multi_round, 0).to_memory(40)
        Y = np.zeros((40,)) + 1
        self.assertItemsEqual(X, Y)

    def test_operations_stream(self):
        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2)) + 2
        predictor_0 = Iterator(data_0).to_chunks(chunks_size=3)
        predictor_1 = Iterator(data_1).to_chunks(chunks_size=3)

        predictor = 4*predictor_0 + predictor_1**3
        result = predictor.to_memory(20)
        self.assertItemsEqual(result[:, 0], np.asarray([[4, 4]]*20)[:, 0])

    def test_operations(self):
        data_0 = np.zeros((20, 2)) + 1.2
        data_1 = np.zeros((20, 2)) + 1
        data_2 = np.zeros((20, 2)) + 3
        predictor_0 = Iterator(data_0).to_chunks(chunks_size=3, dtype=[('x', float), ('y', float)])
        predictor_1 = Iterator(data_1).to_chunks(chunks_size=3)
        predictor_2 = Iterator(data_2).to_chunks(chunks_size=3)

        predictor = ((predictor_0**.65) * (predictor_1**.35) * .85) + predictor_2 * .15
        predictor = predictor.flat().compose(self.multi_round, 2).to_memory(40)
        self.assertItemsEqual(predictor, np.zeros((40,)) + 1.41)

    def test_raw_iter(self):
        data_0 = np.zeros((20, 3)) + 1.2
        predictor_0 = Iterator(data_0)
        predictor_1 = Iterator(data_0+1)
        predictor = predictor_0 + predictor_1
        predictor = predictor.flat().compose(round, 0).to_memory(60)
        self.assertItemsEqual(predictor, np.zeros((60,)) + 3)

    def test_avg(self):
        predictor_0 = Iterator(np.zeros((20, 2)) + 1).to_chunks(chunks_size=3)
        predictor_1 = Iterator(np.zeros((20, 2)) + 2).to_chunks(chunks_size=3)
        predictor_2 = Iterator(np.zeros((20, 2)) + 3).to_chunks(chunks_size=3)

        predictor_avg = ittools.avg([predictor_0, predictor_1, predictor_2])
        self.assertItemsEqual(predictor_avg.flat().to_memory(40), np.zeros((40,)) + 2)

        predictor_0 = Iterator(np.zeros((20, 2)) + 1).to_chunks(chunks_size=3)
        predictor_1 = Iterator(np.zeros((20, 2)) + 2).to_chunks(chunks_size=3)
        predictor_2 = Iterator(np.zeros((20, 2)) + 3).to_chunks(chunks_size=3)

        predictor_avg = ittools.avg([predictor_0, predictor_1, predictor_2], method="geometric")
        predictor_avg = predictor_avg.flat().compose(self.multi_round, 2).to_memory(40)
        self.assertItemsEqual(predictor_avg, np.zeros((40,)) + 1.82)

    def test_max_counter(self):
        predictor_0 = Iterator(["0", "1", "0", "1", "2", "0", "1", "2"])
        predictor_1 = Iterator(["1", "2", "2", "1", "2", "0", "0", "0"])
        predictor_2 = Iterator(["0", "1", "0", "1", "2", "0", "1", "2"])
        predictor_mc = ittools.max_counter([predictor_0, predictor_1, predictor_2])
        self.assertItemsEqual(predictor_mc.to_memory(8), ['0', '1', '0', '1', '2', '0', '1', '2'])

        weights = [1.5, 2, 1]
        predictor_0 = Iterator(["0", "1", "0", "1", "2", "0", "1", "2"])
        predictor_1 = Iterator(["1", "2", "2", "1", "2", "0", "0", "0"])
        predictor_2 = Iterator(["0", "1", "0", "1", "2", "0", "1", "2"])        
        predictor_mc = ittools.max_counter([predictor_0, predictor_1, predictor_2], weights=weights)
        self.assertItemsEqual(predictor_mc.to_memory(8), ['0', '1', '0', '1', '2', '0', '1', '2'])

    def test_custom_fn(self):
        predictor = Iterator(np.zeros((20, 2)) + 1.6).to_chunks(chunks_size=3)
        predictor = predictor.flat().compose(self.multi_round, 0).to_memory(40)
        self.assertItemsEqual(predictor, np.zeros((40,)) + 2)

    def test_concat_fn(self):
        l0 = np.random.rand(10, 2)
        l1 = np.random.rand(10, 2)
        predictor_0 = Iterator(l0)
        predictor_1 = Iterator(l1)
        predictor = predictor_0.concat(predictor_1)
        self.assertEqual(predictor.to_memory(20).shape, (20, 2))

    def test_concat_n(self):
        l0 = np.zeros((20, 2)) + 1
        l1 = np.zeros((20, 2)) + 2
        l2 = np.zeros((20, 2)) + 3
        fl = np.concatenate((l0.reshape(-1), l1.reshape(-1), l2.reshape(-1)))
        predictor_0 = Iterator(l0).to_chunks(chunks_size=3)
        predictor_1 = Iterator(l1).to_chunks(chunks_size=3)
        predictor_2 = Iterator(l2).to_chunks(chunks_size=3)
        predictor = ittools.concat([predictor_0, predictor_1, predictor_2])
        self.assertItemsEqual(predictor.flat().to_memory(120), fl)

    def test_operations_concat_n_scalar(self):
        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2)) + 1
        w = 3
        predictor_0 = Iterator(data_0).to_chunks(chunks_size=2)
        predictor_1 = Iterator(data_1).to_chunks(chunks_size=2)

        predictor = ittools.concat([predictor_0, predictor_1])
        predictors = predictor * w
        predictors = predictors.flat().to_memory(80)
        self.assertItemsEqual(predictors.reshape(-1)[:40], np.zeros((40)) - 3)
        self.assertItemsEqual(predictors.reshape(-1)[40:], np.zeros((40)) + 3)

    def test_append_data_to_iter(self):
        data = [[0, 1, 0], [2, 3, 0], [4, 5, 0], [5, 6, 0]]
        data_i = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        iter_layer = Iterator((e for e in data_i))
        iter_ce = iter_layer.concat_elems(data)

        for i, e in enumerate(iter_ce):
            self.assertItemsEqual(list(e), data_i[i] + data[i])

    def test_append_iter_to_iter(self):
        data_i2 = [[0, 1, 0], [2, 3, 0], [4, 5, 0], [5, 6, 0]]
        data_i1 = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        iter_layer_1 = Iterator((e for e in data_i1))
        iter_layer_2 = Iterator((e for e in data_i2))
        iter_ce = iter_layer_1.concat_elems(iter_layer_2)

        for i, e in enumerate(iter_ce):
            self.assertItemsEqual(list(e), data_i1[i] + data_i2[i])

    def test_flat_shape(self):
        data = np.random.rand(10, 1)
        it = Iterator(data)
        data_flat = it.flat().to_memory(10)
        self.assertEqual(data_flat.shape, (10,))

        data = np.random.rand(10, 5)
        it = Iterator(data)
        data_flat = it.flat().to_memory(50)
        self.assertEqual(data_flat.shape, (50,))

        data = np.random.rand(10, 2 ,2)
        it = Iterator(data)
        data_flat = it.flat().to_memory(40)
        self.assertEqual(data_flat.shape, (40,))

        data = np.random.rand(1000, 2 ,2)
        it = Iterator(data)
        data_flat = it.to_chunks(chunks_size=100).flat().to_memory(4000)
        self.assertEqual(data_flat.shape, (4000,))

    def test_shape(self):
        data = np.random.rand(10, 3)
        it = Iterator(data)
        self.assertEqual(it.shape, (None, 3))
        self.assertEqual(it.features_dim, (3,))

        data = np.random.rand(10)
        it = Iterator(data)
        self.assertEqual(it.shape, (None,))
        self.assertEqual(it.features_dim, ())

        data = np.random.rand(10, 3, 3)
        it = Iterator(data)
        self.assertEqual(it.shape, (None, 3, 3))
        self.assertEqual(it.features_dim, (3, 3))

        data = np.random.rand(10)
        it = Iterator(data).to_chunks(2)
        self.assertEqual(it.shape, (None,))
        self.assertEqual(it.features_dim, ())

        data = np.random.rand(10, 3)
        it = Iterator(data).to_chunks(2)
        self.assertEqual(it.shape, (None, 3))
        self.assertEqual(it.features_dim, (3,))

        data = np.random.rand(10, 3, 3)
        it = Iterator(data).to_chunks(2)
        self.assertEqual(it.shape, (None, 3, 3))
        self.assertEqual(it.features_dim, (3, 3))

    def test_chunks(self):
        chunks_size = 3
        data = np.random.rand(10, 1)
        it = Iterator(data)
        it_0 = it.to_chunks(chunks_size)
        self.assertEqual(it_0.chunks_size, chunks_size)
        self.assertEqual(it_0.has_chunks, True)
        for smx in it_0:
            self.assertEqual(smx.shape[0] <= 3, True)

    def test_chunks_obj(self):
        chunks_size = 3
        m = [[1, '5.0'], [2, '3'], [4, 'C']]
        data = pd.DataFrame(m, columns=['A', 'B'])
        it = Iterator(data)
        it_0 = it.to_chunks(chunks_size)
        self.assertEqual(it_0.chunks_size, chunks_size)
        self.assertEqual(it_0.has_chunks, True)
        for i, smx in enumerate(it_0):
            for i, row in enumerate(smx.values):
                self.assertItemsEqual(row, m[i])

    def test_from_chunks(self):
        data = np.random.rand(10, 1)
        chunks_size = 2
        it = Iterator(data)
        for smx in it.to_chunks(chunks_size):
            self.assertEqual(smx.shape[0], 2)

    def test_chunks_to_array(self):
        chunks_size = 3
        data = np.random.rand(10, 1)
        it = Iterator(data).to_chunks(chunks_size)
        self.assertItemsEqual(it.to_memory(10), data)

        chunks_size = 0
        data = np.random.rand(10, 1)
        it = Iterator(data).to_chunks(chunks_size)
        self.assertItemsEqual(it.to_memory(10), data)

    def test_df_chunks(self):
        chunks_size = 3
        data = np.asarray([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]], dtype='float')
        it = Iterator(data, dtype=[('x', 'float'), ('y', 'float')]).to_chunks(chunks_size)
        chunk = next(it)
        self.assertItemsEqual(chunk.x, [0, 2, 4])
        self.assertItemsEqual(chunk.y, [1, 3, 5])

        chunks_size = 2
        data = np.asarray([1,2,3,4,5,6,7,8,9], dtype='float')
        it = Iterator(data, dtype=[('x', 'float')]).to_chunks(chunks_size)
        chunk = next(it)
        self.assertItemsEqual(chunk['x'].values, [1, 2])

    def test_chunk_taste_dtype_array(self):
        chunks_size = 2
        data = np.asarray([1,2,3,4,5,6,7,8,9], dtype='int')
        it = Iterator(data).to_chunks(chunks_size)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertItemsEqual(it.to_memory(9), data)

        data = np.asarray([1,2,3,'4',5,6,7,8,9], dtype='|O')
        it = Iterator(data).to_chunks(chunks_size, dtype="|O")
        self.assertEqual(it.dtype, np.dtype('|O'))
        self.assertEqual(it.global_dtype, np.dtype('|O'))
        self.assertItemsEqual(it.to_memory(9), data)

        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = Iterator(data)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertEqual(isinstance(it.to_memory(9), np.ndarray), True)
        
    def test_chunk_taste_dtype_df(self):
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        data = pd.DataFrame(data, columns=['x', 'y'])

        it = Iterator(data).to_chunks(chunks_size=3)
        self.assertEqual(it.dtype, [('x', np.dtype('int64')), ('y', np.dtype('int64'))])
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertEqual(isinstance(it.to_memory(5), pd.DataFrame), True)        

    def test_chunk_taste_no_chunks(self):
        data = np.asarray([1,2,3,4,5,6,7,8,9], dtype='int')
        it = Iterator(data)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))

        data = np.asarray(['1',2,3,4,5,6,7,8,9], dtype='|O')
        it = Iterator(data)
        self.assertEqual(it.dtype, np.dtype('|O'))
        self.assertEqual(it.global_dtype, np.dtype('|O'))

    def test_chunk_taste_no_chunks_df(self):
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = Iterator(data, dtype=[('x', 'int'), ('y', 'float')])
        self.assertEqual(it.dtype, [('x', 'int'), ('y', 'float')])
        self.assertEqual(isinstance(it.to_memory(5), pd.DataFrame), True)

    def test_chunk_taste_type_elem(self):
        iter_ = (float(i) for i in range(10))
        it = Iterator(iter_)
        self.assertEqual(it.type_elem, float)

        iter_ = ((i,) for i in range(10))
        it = Iterator(iter_)
        self.assertEqual(it.type_elem, tuple)

        iter_ = [[1, 2], [2,3]]
        it = Iterator(iter_)
        self.assertEqual(it.type_elem, list)
        
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        it = Iterator(data)
        self.assertEqual(it.type_elem, np.ndarray)

        it = Iterator(data).to_chunks(chunks_size=2, dtype=[('x', 'int'), ('y', 'int')])
        self.assertEqual(it.type_elem, pd.DataFrame)

        it = Iterator(data).to_chunks(chunks_size=2, dtype='int')
        self.assertEqual(it.type_elem, np.ndarray)

        it = Iterator(data).to_chunks(chunks_size=2)
        self.assertEqual(it.type_elem, np.ndarray)

    def test_flat(self):
        result = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]
        iter_ = ((i,i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertItemsEqual(it.flat().to_memory(20), result)

        iter_ = ((i,i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertItemsEqual(it.to_chunks(chunks_size=3).flat().to_memory(20), result)

        iter_ = ((i,i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertItemsEqual(
            it.to_chunks(chunks_size=3, dtype=[('x', int), ('y', int)]).flat().to_memory(20), result)

        iter_ = ((i,i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertItemsEqual(it.flat().to_chunks(chunks_size=3, dtype=int).to_memory(20), result)

        result = range(10)
        iter_ = ((i,) for i in range(10))
        it = Iterator(iter_)
        self.assertItemsEqual(it.flat().to_memory(10), result)

    def test_clean_chunks(self):
        it = Iterator(((i, 'X', 'Z') for i in range(20)))
        chunks_size = 2
        it_0 = it.to_chunks(chunks_size=chunks_size)
        for smx in it_0.clean_chunks():
            self.assertEqual(smx.shape[0] <=3, True)

    def test_sample(self):
        order = (i for i in range(20))
        it = Iterator(order)
        it_0 = it.to_chunks(chunks_size=2)
        it_s = it_0.sample(5)
        self.assertEqual(len(it_s.to_memory(5)), 5)

    def test_gen_weights(self):
        order = (i for i in range(4))
        it = Iterator(order)
        it_0 = it.weights_gen(it, None, lambda x: x%2+1)
        self.assertItemsEqual(list(it_0), [(0, 1), (1, 2), (2, 1), (3, 2)])

        def fn(v):
            if v == 0:
                return 1
            else:
                return 99

        data = np.zeros((20, 4)) + [1,2,3,0]
        data[:, 3] = np.random.rand(1,20) > .5
        it = Iterator(data)
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
        it = Iterator(data)
        it_s = it.sample(num_samples, col=3, weight_fn=fn)
        c = collections.Counter(it_s.to_memory(num_items)[:, 3])
        self.assertEqual(c[1]/float(num_samples) > .79, True)

    def test_split(self):
        it = Iterator(((i, 'X', 'Z') for i in range(20)))
        it0, it1 = it.split(2)
        self.assertItemsEqual(it0.to_memory(20)[0], [0, 'X'])
        self.assertItemsEqual(it1.flat().to_memory(20), np.asarray(['Z']*20))

        it = Iterator(((i, 'X', 'Z') for i in range(20)))
        it_0, it_1 = it.split(2)
        self.assertItemsEqual(it_0.to_chunks(2).to_memory(20)[0], [0, 'X'])
        self.assertItemsEqual(it_1.to_chunks(2).flat().to_memory(20), np.asarray(['Z']*20))

        data = ((i, 'X'+str(i), 'Z') for i in range(20))
        it = Iterator(data, dtype=[('A', 'int'), ('B', '|O'), ('C', '|O')])
        it_0, it_1 = it.to_chunks(chunks_size=2).split(2)
        self.assertItemsEqual(it_0.to_memory(20).iloc[0, :].values, [0, 'X0'])
        self.assertItemsEqual(it_1.flat().to_memory(20), np.asarray(['Z']*20))

        data = ((i, 'X', 'Z') for i in range(20))
        it = Iterator(data, dtype=[('A', 'int'), ('B', 'str'), ('C', 'str')])
        self.assertItemsEqual(it.split(2)[0].to_memory(1).values[0], [0, 'X'])

    def test_raw(self):
        data = [1, 2, 3, 4, 5]
        it = Iterator(data)
        self.assertItemsEqual(it.to_memory(len(data)), data)

        data = np.random.rand(2, 3)
        it = Iterator(data)
        for i, row in enumerate(it.to_memory(2)):
            self.assertItemsEqual(row, data[i])

        data = [[1,2], [3,4]]
        it = Iterator(data)
        for i, row in enumerate(it.to_memory(len(data))):
            self.assertItemsEqual(row, data[i])

    def test_empty(self):
        self.assertItemsEqual(Iterator([]).to_memory(0), np.asarray([]))
        self.assertEqual(Iterator([]).to_memory(0).shape, (0,))

    def test_columns(self):
        data = np.asarray([[1,2],[3,4],[5,6],[7,8],[9,0]], dtype='int')
        data = pd.DataFrame(data, columns=['x', 'y'])
        it = Iterator(data)
        self.assertItemsEqual(it.columns(), ['x', 'y'])

    def test_shape_memory(self):
        data = np.random.rand(10, 2)
        it = Iterator(data)
        self.assertItemsEqual(it.to_memory(10)[:, 0], data[:, 0])
        it = Iterator(data)
        self.assertItemsEqual(it.to_chunks(3).to_memory(10)[:, 0], data[:, 0])
        it = Iterator(data)
        self.assertItemsEqual(it.to_memory(5)[:, 0], data[:5, 0])
        it = Iterator(data)
        self.assertItemsEqual(it.to_chunks(3).to_memory(5)[:, 0], data[:5, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertItemsEqual(it.to_memory(10).values[:, 0], data[:, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertItemsEqual(it.to_chunks(3).to_memory(10).values[:, 0], data[:, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertItemsEqual(it.to_memory(5).values[:, 0], data[:5, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertItemsEqual(it.to_chunks(3).to_memory(5).values[:, 0], data[:5, 0])

    def test_to_memory(self):
        data = np.random.rand(10)
        it = Iterator(data)
        self.assertItemsEqual(it.to_memory(10), data[:])

        it = Iterator(data)
        self.assertItemsEqual(it.to_memory(12), data[:])

        it = Iterator(data)
        self.assertItemsEqual(it.to_memory(3), data[:3])

    def test_chunks_to_memory(self):
        data = np.random.rand(10)
        it = Iterator(data)
        self.assertItemsEqual(it.to_chunks(3).to_memory(10), data[:])

        it = Iterator(data)
        self.assertItemsEqual(it.to_memory(12), data[:])

        it = Iterator(data)
        self.assertItemsEqual(it.to_chunks(3).to_memory(3), data[:3])

    def test_datetime(self):
        chunks_size = 2
        m = [[datetime.datetime.today()], [datetime.datetime.today()]]
        data = pd.DataFrame(m, columns=['A'])
        it = Iterator(data)
        it_0 = it.to_chunks(chunks_size)

        data = np.asarray(m)
        it = Iterator(m, dtype=[("A", np.dtype('<M8[ns]'))])
        it_0 = it.to_chunks(chunks_size)

    def test_chunks_unique(self):
        it = Iterator([1,2,3,4,4,4,5,6,3,8,1])
        counter = it.to_chunks(3).unique()
        self.assertEqual(counter[1], 2)
        self.assertEqual(counter[2], 1)
        self.assertEqual(counter[3], 2)
        self.assertEqual(counter[4], 3)
        self.assertEqual(counter[5], 1)
        self.assertEqual(counter[6], 1)
        self.assertEqual(counter[8], 1)


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
