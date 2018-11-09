import unittest
import numpy as np
import pandas as pd
import datetime
import collections

from ml.data.it import Iterator
from ml.data.ds import Data
from ml import ittools


def multi_round(matrix, *args):
    return np.asarray([round(x, *args) for x in matrix])


def stream():
    i = 0
    while True:
        yield i
        i += 1


class TestIterator(unittest.TestCase):

    def test_length_array(self):
        data = np.zeros((20, 2))
        it = Iterator(data)
        result = it[:5].to_ndarray()
        self.assertEqual((result == data[:5]).all(), True)

    def test_length_batch(self):
        data = np.zeros((20, 2))
        it = Iterator(data).batchs(batch_size=3, df=False)
        result = it[:5].to_ndarray()
        self.assertEqual((result == data[:5]).all(), True)

    def test_flat_all(self):
        data = np.zeros((20, 2))
        it = Iterator(data)
        x = it.flat().to_ndarray()
        y = np.zeros((40,))
        self.assertCountEqual(x, y)

    def test_flat_all_batch(self):
        data = np.zeros((20, 2))
        it = Iterator(data).batchs(batch_size=3, df=False)
        x = it.flat().to_ndarray()
        y = np.zeros((40,))
        self.assertCountEqual(x, y)

    def test_it_attrs(self):
        it = Iterator(stream())
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), None)
        self.assertEqual(it.shape, (None,))
        self.assertEqual(it.num_splits(), None)
        self.assertEqual(it.type_elem, int)
        self.assertEqual(it.columns, ["c0"])

    def test_it_attrs_length(self):
        it = Iterator(stream())[:10]
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), 10)
        self.assertEqual(it.shape, (10,))
        self.assertEqual(it.num_splits(), 10)
        self.assertEqual(it.type_elem, int)
        self.assertEqual(it.columns, ["c0"])

    def test_batch_it_attrs(self):
        it = Iterator(stream()).batchs(batch_size=3, df=True)
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), None)
        self.assertEqual(it.shape, (None,))
        self.assertEqual(it.batch_size, 3)
        self.assertEqual(it.num_splits(), 0)
        self.assertEqual(it.batch_shape(), [3])
        self.assertEqual(it.type_elem, pd.DataFrame)
        self.assertEqual(it.columns, ["c0"])

    def test_batch_it_attrs_length(self):
        it = Iterator(stream()).batchs(batch_size=3, df=True)[:10]
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), 10)
        self.assertEqual(it.shape, (10,))
        self.assertEqual(it.batch_size, 3)
        self.assertEqual(it.num_splits(), 4)
        self.assertEqual(it.batch_shape(), [3])
        self.assertEqual(it.type_elem, pd.DataFrame)
        self.assertEqual(it.columns, ["c0"])

    def test_stream(self):
        it = Iterator(stream())
        data = Data(name="test", driver="memory")
        data.from_data(it[:10])
        self.assertCountEqual(data.to_ndarray(), np.arange(1, 11))
        self.assertCountEqual(data.to_df().values, pd.DataFrame(np.arange(1, 11)).values)

    def test_stream_batchs(self):
        it = Iterator(stream()).batchs(batch_size=3, df=True)
        data = Data(name="test", driver="memory")
        data.from_data(it[:10])
        print(data.to_ndarray())
        #print(data.to_df())

    def test_concat_it(self):
        l0 = np.random.rand(10, 2)
        l1 = np.random.rand(10, 2)
        predictor_0 = Iterator(l0)
        predictor_0.set_length(10)
        predictor_1 = Iterator(l1)
        predictor_1.set_length(10)
        predictor = predictor_0.concat(predictor_1)
        self.assertEqual(predictor.to_memory().shape, (20, 2))

    def test_concat_it_chunks(self):
        l0 = np.zeros((10, 2)) + 1
        l1 = np.zeros((10, 2)) + 2
        l2 = np.zeros((10, 2)) + 3
        predictor_0 = Iterator(l0).batchs(3)
        predictor_1 = Iterator(l1).batchs(3)
        predictor_2 = Iterator(l2).batchs(3)
        predictor_0 = predictor_0.concat(predictor_1)
        predictor = predictor_0.concat(predictor_2)
        m = predictor.flat().to_memory()
        self.assertCountEqual(m, np.concatenate([l0, l1, l2]).reshape(-1))

    def test_concat_it_chunks_df(self):
        l0 = np.zeros((10, 2)) + 1
        l1 = np.zeros((10, 2)) + 2
        l2 = np.zeros((10, 2)) + 3
        dtype = [("a", int), ("b", int)]
        predictor_0 = Iterator(l0, dtype=dtype).batchs(3)
        predictor_1 = Iterator(l1, dtype=dtype).batchs(3)
        predictor_2 = Iterator(l2, dtype=dtype).batchs(3)
        predictor_0 = predictor_0.concat(predictor_1)
        predictor = predictor_0.concat(predictor_2)
        m = predictor.flat().to_memory()
        self.assertCountEqual(m, np.concatenate([l0, l1, l2]).reshape(-1))

    def test_concat_n(self):
        l0 = np.zeros((10, 2)) + 1
        l1 = np.zeros((10, 2)) + 2
        l2 = np.zeros((10, 2)) + 3
        fl = np.concatenate((l0.reshape(-1), l1.reshape(-1), l2.reshape(-1)))
        predictor_0 = Iterator(l0).batchs(3)
        predictor_0.set_length(10)
        predictor_1 = Iterator(l1).batchs(3)
        predictor_1.set_length(10)
        predictor_2 = Iterator(l2).batchs(3)
        predictor_2.set_length(10)
        predictor = ittools.concat([predictor_0, predictor_1, predictor_2])
        self.assertEqual(predictor.shape, (30, 2))
        self.assertCountEqual(predictor.flat().to_memory(), fl)

    def test_operations_concat_n_scalar(self):
        data_0 = np.zeros((20, 2)) - 1 
        data_1 = np.zeros((20, 2)) + 1
        w = 3
        predictor_0 = Iterator(data_0).batchs(2)
        predictor_1 = Iterator(data_1).batchs(2)

        predictor = ittools.concat([predictor_0, predictor_1])
        predictors = predictor * w
        predictors = predictors.flat().to_memory()
        self.assertCountEqual(predictors.reshape(-1)[:40], np.zeros(40)-3)
        self.assertCountEqual(predictors.reshape(-1)[40:], np.zeros(40)+3)

    def test_append_iter_to_iter(self):
        data_i2 = [[0, 1], [2, 3], [4, 5], [6, 7]]
        data_i1 = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        iter_layer_1 = Iterator((e for e in data_i1))
        iter_layer_2 = Iterator((e for e in data_i2))
        iter_ce = iter_layer_1.concat(iter_layer_2)

        self.assertCountEqual(iter_ce.flat().to_memory(16),
                              ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 0, 1, 2, 3, 4, 5, 6, 7])

    def test_flat_shape(self):
        data = np.random.rand(10, 1)
        it = Iterator(data)
        data_flat = it.flat().to_memory(10)
        self.assertEqual(data_flat.shape, (10,))

        data = np.random.rand(10, 5)
        it = Iterator(data)
        data_flat = it.flat().to_memory(50)
        self.assertEqual(data_flat.shape, (50,))

        data = np.random.rand(10, 2, 2)
        it = Iterator(data)
        data_flat = it.flat().to_memory(40)
        self.assertEqual(data_flat.shape, (40,))

        data = np.random.rand(1000, 2, 2)
        it = Iterator(data)
        data_flat = it.batchs(batch_size=100).flat().to_memory(4000)
        self.assertEqual(data_flat.shape, (4000,))

    def test_shape(self):
        data = np.random.rand(10, 3)
        it = Iterator(data)
        self.assertEqual(it.shape, (10, 3))
        self.assertEqual(it.features_dim, (3,))

        data = np.random.rand(10)
        it = Iterator(data)
        self.assertEqual(it.shape, (10,))
        self.assertEqual(it.features_dim, ())

        data = np.random.rand(10, 3, 3)
        it = Iterator(data)
        self.assertEqual(it.shape, (10, 3, 3))
        self.assertEqual(it.features_dim, (3, 3))

        data = np.random.rand(10)
        it = Iterator(data).batchs(2)
        self.assertEqual(it.shape, (10,))
        self.assertEqual(it.features_dim, ())

        data = np.random.rand(10, 3)
        it = Iterator(data).batchs(2)
        self.assertEqual(it.shape, (10, 3))
        self.assertEqual(it.features_dim, (3,))

        data = np.random.rand(10, 3, 3)
        it = Iterator(data).batchs(2)
        self.assertEqual(it.shape, (10, 3, 3))
        self.assertEqual(it.features_dim, (3, 3))

    def test_chunks(self):
        batch_size = 3
        data = np.random.rand(10, 1)
        it = Iterator(data)
        it_0 = it.batchs(batch_size)
        self.assertEqual(it_0.batch_size, batch_size)
        self.assertEqual(it_0.has_batchs, True)
        for smx in it_0:
            self.assertEqual(smx.shape[0] <= 3, True)

    def test_chunks_obj(self):
        batch_size = 3
        m = [[1, '5.0'], [2, '3'], [4, 'C']]
        data = pd.DataFrame(m, columns=['A', 'B'])
        it = Iterator(data)
        it_0 = it.batchs(batch_size)
        self.assertEqual(it_0.batch_size, batch_size)
        self.assertEqual(it_0.has_batchs, True)
        for _, smx in enumerate(it_0):
            for i, row in enumerate(smx.values):
                self.assertCountEqual(row, m[i])

    def test_from_chunks(self):
        data = np.random.rand(10, 1)
        batch_size = 2
        it = Iterator(data)
        for smx in it.batchs(batch_size):
            self.assertEqual(smx.shape[0], 2)

    def test_chunks_to_array(self):
        batch_size = 3
        data = np.random.rand(10, 1)
        it = Iterator(data).batchs(batch_size)
        self.assertCountEqual(it.to_memory(10), data)

        batch_size = 0
        data = np.random.rand(10, 1)
        it = Iterator(data).batchs(batch_size)
        self.assertCountEqual(it.to_memory(10), data)

    def test_df_chunks(self):
        batch_size = 3
        data = np.asarray([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]], dtype='float')
        it = Iterator(data, dtype=[('x', 'float'), ('y', 'float')]).batchs(batch_size)
        chunk = next(it)
        self.assertCountEqual(chunk.x, [0, 2, 4])
        self.assertCountEqual(chunk.y, [1, 3, 5])

        batch_size = 2
        data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float')
        it = Iterator(data, dtype=[('x', 'float')]).batchs(batch_size)
        chunk = next(it)
        self.assertCountEqual(chunk['x'].values, [1, 2])

    def test_chunk_taste_dtype_array(self):
        batch_size = 2
        data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int')
        it = Iterator(data).batchs(batch_size)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertCountEqual(it.to_memory(9), data)

        data = np.asarray([1, 2, 3, '4', 5, 6, 7, 8, 9], dtype='|O')
        it = Iterator(data).batchs(batch_size, dtype="|O")
        self.assertEqual(it.dtype, np.dtype('|O'))
        self.assertEqual(it.global_dtype, np.dtype('|O'))
        self.assertCountEqual(it.to_memory(9), data)

        data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]], dtype='int')
        it = Iterator(data)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertEqual(isinstance(it.to_memory(9), np.ndarray), True)
        
    def test_chunk_taste_dtype_df(self):
        data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]], dtype='int')
        data = pd.DataFrame(data, columns=['x', 'y'])

        it = Iterator(data).batchs(batch_size=3)
        self.assertEqual(it.dtype, [('x', np.dtype('int64')), ('y', np.dtype('int64'))])
        self.assertEqual(it.global_dtype, np.dtype('int'))
        self.assertEqual(isinstance(it.to_memory(5), pd.DataFrame), True)        

    def test_chunk_taste_no_chunks(self):
        data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int')
        it = Iterator(data)
        self.assertEqual(it.dtype, np.dtype('int'))
        self.assertEqual(it.global_dtype, np.dtype('int'))

        data = np.asarray(['1', 2, 3, 4, 5, 6, 7, 8, 9], dtype='|O')
        it = Iterator(data)
        self.assertEqual(it.dtype, np.dtype('|O'))
        self.assertEqual(it.global_dtype, np.dtype('|O'))

    def test_chunk_taste_no_chunks_df(self):
        data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]], dtype='int')
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

        iter_ = [[1, 2], [2, 3]]
        it = Iterator(iter_)
        self.assertEqual(it.type_elem, list)
        
        data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]], dtype='int')
        it = Iterator(data)
        self.assertEqual(it.type_elem, np.ndarray)

        it = Iterator(data).batchs(batch_size=2, dtype=[('x', 'int'), ('y', 'int')])
        self.assertEqual(it.type_elem, pd.DataFrame)

        it = Iterator(data).batchs(batch_size=2, dtype='int')
        self.assertEqual(it.type_elem, np.ndarray)

        it = Iterator(data).batchs(batch_size=2)
        self.assertEqual(it.type_elem, np.ndarray)

    def test_flat(self):
        result = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10]
        iter_ = ((i, i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertCountEqual(it.flat().to_memory(20), result)

        iter_ = ((i, i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertCountEqual(it.batchs(batch_size=3).flat().to_memory(20), result)

        iter_ = ((i, i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertCountEqual(
            it.batchs(batch_size=3, dtype=[('x', int), ('y', int)]).flat().to_memory(20), result)

        iter_ = ((i, i+1) for i in range(10))
        it = Iterator(iter_)
        self.assertCountEqual(it.flat().batchs(batch_size=3, dtype=int).to_memory(20), result)

        result = range(10)
        iter_ = ((i,) for i in range(10))
        it = Iterator(iter_)
        self.assertCountEqual(it.flat().to_memory(10), result)

    def test_clean_chunks(self):
        it = Iterator(((i, 'X', 'Z') for i in range(20)))
        batch_size = 2
        it_0 = it.batchs(batch_size=batch_size)
        for smx in it_0.clean_chunks():
            self.assertEqual(smx.shape[0] <= 3, True)

    def test_sample(self):
        order = (i for i in range(20))
        it = Iterator(order)
        it_0 = it.batchs(batch_size=2)
        it_s = it_0.sample(5)
        self.assertEqual(len(it_s.to_memory(5)), 5)

    def test_gen_weights(self):
        order = (i for i in range(4))
        it = Iterator(order)
        it_0 = it.weights_gen(it, None, lambda x: x % 2 + 1)
        self.assertCountEqual(list(it_0), [(0, 1), (1, 2), (2, 1), (3, 2)])

        def fn(v):
            if v == 0:
                return 1
            else:
                return 99

        data = np.zeros((20, 4)) + [1, 2, 3, 0]
        data[:, 3] = np.random.rand(1, 20) > .5
        it = Iterator(data)
        w_v = list(it.weights_gen(it, 3, fn))
        self.assertEqual(w_v[0][1], fn(data[0][3]))

    def test_sample_weight(self):
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

    def test_raw(self):
        data = [1, 2, 3, 4, 5]
        it = Iterator(data)
        self.assertCountEqual(it.to_memory(len(data)), data)

        data = np.random.rand(2, 3)
        it = Iterator(data)
        for i, row in enumerate(it.to_memory(2)):
            self.assertCountEqual(row, data[i])

        data = [[1, 2], [3, 4]]
        it = Iterator(data)
        for i, row in enumerate(it.to_memory(len(data))):
            self.assertCountEqual(row, data[i])

    def test_empty(self):
        self.assertCountEqual(Iterator([]).to_memory(0), np.asarray([]))
        self.assertEqual(Iterator([]).to_memory(0).shape, (0,))

    def test_columns(self):
        data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]], dtype='int')
        data = pd.DataFrame(data, columns=['x', 'y'])
        it = Iterator(data)
        self.assertCountEqual(it.columns, ['x', 'y'])

    def test_shape_memory(self):
        data = np.random.rand(10, 2)
        it = Iterator(data)
        self.assertCountEqual(it.to_memory(10)[:, 0], data[:, 0])
        it = Iterator(data)
        self.assertCountEqual(it.batchs(3).to_memory(10)[:, 0], data[:, 0])
        it = Iterator(data)
        self.assertCountEqual(it.to_memory(5)[:, 0], data[:5, 0])
        it = Iterator(data)
        self.assertCountEqual(it.batchs(3).to_memory(5)[:, 0], data[:5, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertCountEqual(it.to_memory(10).values[:, 0], data[:, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertCountEqual(it.batchs(3).to_memory(10).values[:, 0], data[:, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertCountEqual(it.to_memory(5).values[:, 0], data[:5, 0])
        it = Iterator(data, dtype=[("a", float), ("b", float)])
        self.assertCountEqual(it.batchs(3).to_memory(5).values[:, 0], data[:5, 0])

    def test_to_memory(self):
        data = np.random.rand(10)
        it = Iterator(data)
        self.assertCountEqual(it.to_memory(10), data[:])

        it = Iterator(data)
        self.assertCountEqual(it.to_memory(12), data[:])

        it = Iterator(data)
        self.assertCountEqual(it.to_memory(3), data[:3])

    def test_chunks_to_memory(self):
        data = np.random.rand(10)
        it = Iterator(data)
        self.assertCountEqual(it.batchs(3).to_memory(10), data[:])

        it = Iterator(data)
        self.assertCountEqual(it.to_memory(12), data[:])

        it = Iterator(data)
        self.assertCountEqual(it.batchs(3).to_memory(3), data[:3])

    def test_datetime(self):
        batch_size = 2
        m = [[datetime.datetime.today()], [datetime.datetime.today()]]
        data = pd.DataFrame(m, columns=['A'])
        it = Iterator(data)
        it.batchs(batch_size)

        it = Iterator(m, dtype=[("A", np.dtype('<M8[ns]'))])
        it.batchs(batch_size)

    def test_chunks_unique(self):
        it = Iterator([1, 2, 3, 4, 4, 4, 5, 6, 3, 8, 1])
        counter = it.batchs(3).unique()
        self.assertEqual(counter[1], 2)
        self.assertEqual(counter[2], 1)
        self.assertEqual(counter[3], 2)
        self.assertEqual(counter[4], 3)
        self.assertEqual(counter[5], 1)
        self.assertEqual(counter[6], 1)
        self.assertEqual(counter[8], 1)

    def test_to_df(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [6, 7, 8, 9]})
        it = Iterator(df)
        self.assertCountEqual(it.to_memory()["a"], df["a"])

    def test_to_array(self):
        array = np.random.rand(10)
        it = Iterator(array)
        self.assertCountEqual(it.to_memory(), array)

    def test_df_index_chunks(self):
        array = np.random.rand(10, 2)
        it = Iterator(array, dtype=[("a", int), ("b", int)]).batchs(3)
        df = next(it)
        self.assertCountEqual(df.index.values, np.array([0, 1, 2]))
        df = next(it)
        self.assertCountEqual(df.index.values, np.array([0, 1, 2]) + 3)
        df = next(it)
        self.assertCountEqual(df.index.values, np.array([0, 1, 2]) + 6)
        df = next(it)
        self.assertCountEqual(df.index.values, np.array([0]) + 9)

    def test_buffer(self):
        v = list(range(100))
        it = Iterator(v)
        buffer_size = 7
        i = 0
        j = buffer_size
        for elems in it.buffer(buffer_size):
            self.assertCountEqual(elems, list(range(i, j)))
            i = j
            j += buffer_size
            if j > 100:
                j = 100

    def test_buffer_chunks(self):
        v = list(range(100))
        batch_size = 2
        it = Iterator(v).batchs(batch_size=batch_size)
        buffer_size = 7
        i = 0
        j = batch_size
        for elems in it.buffer(buffer_size):
            for x in elems:
                self.assertCountEqual(x, list(range(i, j)))
                i = j
                j += batch_size
                if j > 100:
                    j = 100

    def test_sliding_window(self):
        it = Iterator(range(100))
        i = 1
        j = 2
        for e in it.window():
            self.assertCountEqual(e, [i, j])
            i += 1
            j += 1

    def test_ds_batchs(self):
        data = Data(name="test", dataset_path="/tmp", clean=True)
        data.from_data(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        index_l = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        array_l = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        with data:
            it = Iterator(data).batchs(batch_size=3)
            for batch, index, array in zip(it, index_l, array_l):
                self.assertCountEqual(batch.values, array)
                self.assertCountEqual(batch.index, index)
        data.destroy()


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
