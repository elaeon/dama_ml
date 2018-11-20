import unittest
import numpy as np
import pandas as pd
import datetime
import collections

from ml.data.it import Iterator
from ml.data.ds import Data


def multi_round(matrix, *args):
    return np.asarray([round(x, *args) for x in matrix])


def stream():
    i = 0
    while True:
        yield i
        i += 1


class TestIterator(unittest.TestCase):

    def test_mixtype(self):
        array = [1, 2, 3, 4.0, 'xxx', 1, 3]
        it = Iterator(array, dtypes=[("c0", np.dtype("object"))])
        data = Data(name="test", driver="memory")
        data.from_data(it[:7])
        result = data.to_ndarray()
        self.assertCountEqual(result, array, True)

    def test_mixtype2(self):
        array = [1, 2, 3, 4.0, 'xxx', [1], [[2, 3]]]
        it = Iterator(array, dtypes=[("c0", np.dtype("object"))])
        data = Data(name="test", driver="memory")
        data.from_data(it[:7])
        result = data.to_ndarray()
        self.assertCountEqual(result, np.asarray([1, 2, 3, 4.0, 'xxx', 1, [2, 3]], dtype='object'), True)

    def test_length_array(self):
        array = np.zeros((20, 2)) + 1
        it = Iterator(array)
        data = Data(name="test", driver="memory")
        data.from_data(it[:10])
        result = data[:5].to_ndarray()
        self.assertEqual((result == array[:5]).all(), True)

    def test_length_array_batch(self):
        array = np.zeros((20, 2)) + 1
        it = Iterator(array).batchs(batch_size=3, batch_type="array")
        data = Data(name="test", driver="memory")
        data.from_data(it[:10])
        result = data[:5].to_ndarray()
        self.assertEqual((result == array[:5]).all(), True)

    def test_flat_all(self):
        array = np.zeros((20, 2))
        it = Iterator(array)
        x = it.flat()
        data = Data(name="test", driver="memory")
        data.from_data(x)
        y = np.zeros((40,))
        self.assertCountEqual(data.to_ndarray(), y)

    def test_flat_all_batch(self):
        data = np.zeros((20, 2))
        it = Iterator(data).batchs(batch_size=3, batch_type="array")
        x = it.flat()
        data = Data(name="test", driver="memory")
        data.from_data(x)
        y = np.zeros((40,))
        self.assertCountEqual(data.to_ndarray(), y)

    def test_it_attrs(self):
        it = Iterator(stream())
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), None)
        self.assertEqual(it.shape, (None,))
        self.assertEqual(it.num_splits(), None)
        self.assertEqual(it.type_elem, int)
        self.assertEqual(it.labels, ["c0"])

    def test_it_attrs_length(self):
        it = Iterator(stream())[:10]
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), 10)
        self.assertEqual(it.shape, (10,))
        self.assertEqual(it.num_splits(), 10)
        self.assertEqual(it.type_elem, int)
        self.assertEqual(it.labels, ["c0"])

    def test_batch_it_attrs(self):
        it = Iterator(stream()).batchs(batch_size=3, batch_type="df")
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), None)
        self.assertEqual(it.shape, (None,))
        self.assertEqual(it.batch_size, 3)
        self.assertEqual(it.num_splits(), 0)
        self.assertEqual(it.batch_shape(), [3])
        self.assertEqual(it.type_elem, pd.DataFrame)
        self.assertEqual(it.labels, ["c0"])

    def test_batch_it_attrs_length(self):
        it = Iterator(stream()).batchs(batch_size=3, batch_type="df")[:10]
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [('c0', np.dtype('int64'))])
        self.assertEqual(it.length(), 10)
        self.assertEqual(it.shape, (10,))
        self.assertEqual(it.batch_size, 3)
        self.assertEqual(it.num_splits(), 4)
        self.assertEqual(it.batch_shape(), [3])
        self.assertEqual(it.type_elem, pd.DataFrame)
        self.assertEqual(it.labels, ["c0"])

    def test_stream(self):
        it = Iterator(stream())
        data = Data(name="test", driver="memory")
        data.from_data(it[:10])
        self.assertCountEqual(data.to_ndarray(), np.arange(0, 10))
        self.assertCountEqual(data.to_df().values, pd.DataFrame(np.arange(0, 10)).values)

    def test_stream_batchs(self):
        it = Iterator(stream()).batchs(batch_size=3, batch_type="df")
        data = Data(name="test", driver="memory")
        data.from_data(it[:10])
        self.assertCountEqual(data.to_ndarray(), np.arange(0, 10))
        self.assertCountEqual(data.to_df().values, pd.DataFrame(np.arange(0, 10)).values)

    def test_multidtype(self):
        x0 = np.zeros(20) + 1
        x1 = (np.zeros(20) + 2).astype("int")
        x2 = np.zeros(20) + 3
        df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
        data = Data(name="test", driver="memory")
        data.from_data(df, batch_size=0)
        self.assertEqual(data[:5].to_ndarray().shape, (5, 3))
        df = data[:5].to_df()
        self.assertEqual(df.dtypes["x0"], float)
        self.assertEqual(df.dtypes["x1"], int)
        self.assertEqual(df.dtypes["x2"], float)

    def test_onedtype_batchs(self):
        x0 = np.zeros(20) + 1
        x1 = np.zeros(20) + 2
        x2 = np.zeros(20) + 3
        df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
        data = Data(name="test", driver="memory")
        data.from_data(df, batch_size=3)
        self.assertEqual(data[:5].to_ndarray().shape, (5, 3))
        df = data[:5].to_df()
        self.assertEqual(df.dtypes["x0"], float)
        self.assertEqual(df.dtypes["x1"], float)
        self.assertEqual(df.dtypes["x2"], float)

    def test_multidtype_batchs(self):
        x0 = np.zeros(20) + 1
        x1 = (np.zeros(20) + 2).astype("int")
        x2 = np.zeros(20) + 3
        df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
        data = Data(name="test", driver="memory")
        data.from_data(df, batch_size=3)
        self.assertEqual(data[:5].to_ndarray().shape, (5, 3))
        df = data[:5].to_df()
        self.assertEqual(df.dtypes["x0"], float)
        self.assertEqual(df.dtypes["x1"], int)
        self.assertEqual(df.dtypes["x2"], float)

    def test_shape(self):
        data = np.random.rand(10, 3)
        it = Iterator(data)
        self.assertEqual(it.shape, (10, 3))

        data = np.random.rand(10)
        it = Iterator(data)
        self.assertEqual(it.shape, (10,))

        data = np.random.rand(10, 3, 3)
        it = Iterator(data)
        self.assertEqual(it.shape, (10, 3, 3))

        data = np.random.rand(10)
        it = Iterator(data).batchs(2)
        self.assertEqual(it.shape, (10,))

        data = np.random.rand(10, 3)
        it = Iterator(data).batchs(2)
        self.assertEqual(it.shape, (10, 3))

        data = np.random.rand(10, 3, 3)
        it = Iterator(data).batchs(2)
        self.assertEqual(it.shape, (10, 3, 3))

    def test_batchs_values(self):
        batch_size = 3
        m = [[1, '5.0'], [2, '3'], [4, 'C']]
        data = pd.DataFrame(m, columns=['A', 'B'])
        it = Iterator(data)
        it_0 = it.batchs(batch_size)
        self.assertEqual(it_0.batch_size, batch_size)
        for smx in it_0:
            for i, row in enumerate(smx):
                self.assertCountEqual(row, m[i])

    def test_batch_type(self):
        data = np.random.rand(10, 1)
        batch_size = 2
        it = Iterator(data)
        for smx in it.batchs(batch_size, batch_type='structured'):
            self.assertEqual(smx.shape, (2,))
        it = Iterator(data)
        for smx in it.batchs(batch_size, batch_type='array'):
            self.assertEqual(smx.shape, (2, 1))
        it = Iterator(data)
        for smx in it.batchs(batch_size, batch_type='df'):
            self.assertEqual(smx.shape, (2, 1))

    def test_df_batchs(self):
        batch_size = 3
        data = np.asarray([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]], dtype='float')
        dtypes = [('x', np.dtype('float')), ('y', np.dtype('float'))]
        it = Iterator(data, dtypes=dtypes).batchs(batch_size, batch_type="df")
        batch = next(it)
        self.assertCountEqual(batch.x, [0, 2, 4])
        self.assertCountEqual(batch.y, [1, 3, 5])

        batch_size = 2
        data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float')
        it = Iterator(data, dtypes=[('x', np.dtype('float'))]).batchs(batch_size, batch_type="df")
        batch = next(it)
        self.assertCountEqual(batch['x'].values, [1, 2])

    def test_to_ndarray_dtype(self):
        batch_size = 2
        array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int')
        it = Iterator(array).batchs(batch_size)
        data = Data(name="test", driver="memory")
        data.from_data(it)
        with data:
            array = data.to_ndarray(dtype='float')
            self.assertEqual(array.dtype, np.dtype("float"))
        data.destroy()

    def test_clean_batchs(self):
        it = Iterator(((i, 'X', 'Z') for i in range(20)))
        batch_size = 2
        it_0 = it.batchs(batch_size=batch_size)
        for smx in it_0.clean_batchs():
            self.assertEqual(smx.shape[0] <= 3, True)

    def test_sample(self):
        order = (i for i in range(20))
        it = Iterator(order)
        it_s = it.sample(5)
        data = Data(name="test", driver="memory")
        data.from_data(it_s)
        with data:
            self.assertEqual(data.to_ndarray().shape, (5,))

    def test_sample_batch(self):
        order = (i for i in range(20))
        it = Iterator(order).batchs(batch_size=2)
        it_s = it.sample(5)
        data = Data(name="test", driver="memory")
        data.from_data(it_s)
        with data:
            self.assertEqual(data.to_ndarray().shape, (5,))

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
        array = np.zeros((num_items, 4)) + [1, 2, 3, 0]
        array[:, 3] = np.random.rand(1, num_items) > .5
        it = Iterator(array).sample(num_samples, col=3, weight_fn=fn)
        data = Data(name="test", driver="memory")
        data.from_data(it)
        with data:
            c = collections.Counter(data.to_ndarray()[:, 3])
        self.assertEqual(c[1]/float(num_samples) > .79, True)

    def test_empty(self):
        it = Iterator([])
        data = Data(name="test", driver="memory")
        data.from_data(it)
        with data:
            self.assertCountEqual(data.to_ndarray(), np.asarray([]))
            self.assertEqual(data.shape, (0,))
        data.destroy()

    def test_columns(self):
        data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]], dtype='int')
        data = pd.DataFrame(data, columns=['x', 'y'])
        it = Iterator(data)
        self.assertCountEqual(it.labels, ['x', 'y'])

    def test_datetime(self):
        m = [datetime.datetime.today(), datetime.datetime.today(), datetime.datetime.today()]
        df = pd.DataFrame(m, columns=['A'])
        data = Data(name="test", driver="memory")
        data.from_data(df)
        self.assertCountEqual(data.to_df().values, df.values)
        it = Iterator(m, dtypes=[("A", np.dtype('<M8[ns]'))]).batchs(batch_size=2, batch_type="df")
        data = Data(name="test", driver="memory")
        data.from_data(it)
        self.assertCountEqual(data.to_df().values, df.values)

    def test_unique(self):
        it = Iterator([1, 2, 3, 4, 4, 4, 5, 6, 3, 8, 1])
        counter = it.batchs(3).unique()
        self.assertEqual(counter[1], 2)
        self.assertEqual(counter[2], 1)
        self.assertEqual(counter[3], 2)
        self.assertEqual(counter[4], 3)
        self.assertEqual(counter[5], 1)
        self.assertEqual(counter[6], 1)
        self.assertEqual(counter[8], 1)

    def test_df_index_chunks(self):
        array = np.random.rand(10, 2)
        it = Iterator(array, dtypes=[("a", np.dtype("float")), ("b", np.dtype("float"))]).batchs(3, batch_type="df")
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
        for elems in it.batchs(batch_size=buffer_size):
            self.assertCountEqual(elems, list(range(i, j)))
            i = j
            j += buffer_size
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

    def test_ads(self):
        array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        data = Data(name="test", driver="memory")
        data.from_data(array)
        with data:
            it = Iterator(data)
            for it_array, array_elem in zip(it, array):
                self.assertEqual(it_array.to_ndarray(), array_elem)
        data.destroy()

    def test_batch_ads(self):
        data = Data(name="test", driver="memory")
        data.from_data(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        array_l = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        with data:
            it = Iterator(data).batchs(batch_size=3, batch_type="array")
            for batch, array in zip(it, array_l):
                self.assertCountEqual(batch, array)
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
