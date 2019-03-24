import unittest
import numpy as np
import pandas as pd
import datetime
import collections

from dama.data.it import Iterator, BatchIterator, Slice
from dama.data.ds import Data
from dama.abc.group import DaGroupDict
from dama.fmtypes import DEFAUL_GROUP_NAME
from dama.utils.core import Chunks
from dama.utils.seq import grouper_chunk
from dama.groups.core import ListConn
import numbers

def stream():
    i = 0
    while True:
        yield i
        i += 1


class TestIteratorIter(unittest.TestCase):
    def test_iteration(self):
        array = np.arange(0, 10)
        it = Iterator(array)
        for i, e in enumerate(it):
            self.assertEqual(e, i)

    def test_iteration_dtype(self):
        array = [1, 2, 3, 4.0, 'xxx', 1, 3, 4, 5]
        it = Iterator(array, dtypes=np.dtype([("x0", np.dtype("float"))]))
        self.assertEqual(it.groups, ("x0",))
        self.assertEqual(it.dtype, np.dtype("float"))
        for a, e in zip(array, it):
            self.assertEqual(a, e)

    def test_iteration_dtype2(self):
        array = [1, 2, 3, 4.0, 'xxx', [1], [[2, 3]]]
        it = Iterator(array, dtypes=np.dtype([(DEFAUL_GROUP_NAME, np.dtype("float"))]))
        for a, e in zip(array, it):
            self.assertEqual(a, e)

    def test_nshape(self):
        array = np.zeros((20, 2)) + 1
        it = Iterator(array)
        self.assertEqual(it.shape, (20, 2))
        self.assertEqual(it[:10].shape, (10, 2))

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

    def test_iterator_cut(self):
        array = np.arange(0, 100)
        it = Iterator(array)
        list_ = []
        for e in it[:10]:
            list_.append(e)
        self.assertEqual((list_ == array[:10]).all(), True)

    def test_flat_all(self):
        array = np.empty((20, 2), dtype=np.dtype(int))
        array[:, 0] = np.arange(0, 20)
        array[:, 1] = np.arange(0, 20) + 2
        it = Iterator(array)
        flat_array = array.reshape(-1)
        for i, e in enumerate(it.flat()):
            self.assertEqual(e, flat_array[i])

    def test_it_attrs(self):
        it = Iterator(stream())
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [(DEFAUL_GROUP_NAME, np.dtype('int64'))])
        self.assertEqual(it.length, np.inf)
        self.assertEqual(it.shape, (np.inf,))
        self.assertEqual(it.num_splits(), np.inf)
        self.assertEqual(it.type_elem, numbers.Number)
        self.assertEqual(it.groups, (DEFAUL_GROUP_NAME,))

    def test_it_attrs_length(self):
        it = Iterator(stream())[:10]
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [(DEFAUL_GROUP_NAME, np.dtype('int64'))])
        self.assertEqual(it.length, 10)
        self.assertEqual(it.shape, (10,))
        self.assertEqual(it.num_splits(), 10)
        self.assertEqual(it.type_elem, numbers.Number)
        self.assertEqual(it.groups, (DEFAUL_GROUP_NAME,))

    def test_sample(self):
        order = (i for i in range(20))
        array = np.arange(20)
        it = Iterator(order)
        samples = []
        samples_it = it.sample(5)
        self.assertEqual(isinstance(samples_it, Iterator), True)
        for e in samples_it:
            samples.append(e)
        self.assertEqual((samples == array[:5]).all(), False)

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

    def test_groups(self):
        data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]], dtype='int')
        data = pd.DataFrame(data, columns=['x', 'y'])
        it = Iterator(data)
        self.assertCountEqual(it.groups, ['x', 'y'])

    def test_sliding_window(self):
        it = Iterator(range(100))
        i = 1
        j = 2
        for e in it.window():
            self.assertCountEqual(e, [i, j])
            i += 1
            j += 1

    def test_shape_list(self):
        def _it():
            for x in range(100):
                e = np.random.rand(3, 3)
                yield (x, e, [1, 2])
        it = Iterator(_it(), dtypes=np.dtype([("x", np.dtype("float")), ("y", np.dtype("float")),
                                              ("z", np.dtype("int"))]))
        self.assertEqual(it.shape["x"], (np.inf, ))
        self.assertEqual(it.shape["y"], (np.inf, 3, 3))
        self.assertEqual(it.shape["z"], (np.inf, 2))

    def test_shape_list_one_group(self):
        def _it():
            for _ in range(100):
                yield ([1], [2], [3])
        it = Iterator(_it(), dtypes=np.dtype([("x", np.dtype("float"))]))
        self.assertEqual(it.shape["x"], (np.inf, 3, 1))

    def test_list_dtype(self):
        l = [["a0", 0, "c0", 0], ["a1", 1, "c1", 1], ["a2", 2, "c2", 0]]
        dtypes = np.dtype([("a", np.dtype(object)), ("b", np.dtype(int)), ("c", np.dtype(object)),
                           ("d", np.dtype(bool))])
        it = Iterator(l, dtypes=dtypes).batchs(3)
        for e in it:
            df_v = e.batch.to_df().values
            array = np.asarray(l)
            self.assertEqual((df_v[:, 0] == array[:, 0]).all(), True)
            self.assertEqual((df_v[:, 1] == array[:, 1].astype(int)).all(), True)
            self.assertEqual((df_v[:, 2] == array[:, 2]).all(), True)
            self.assertEqual((df_v[:, 3] == array[:, 3].astype(bool)).all(), True)

    def test_batch_iterator_from(self):
        x = np.random.rand(20)
        batch_size = 5
        dtypes = np.dtype([("x", np.dtype(float)), ("y", np.dtype(float))])
        def iterator(x):
            init = 0
            end = batch_size
            while end <= x.shape[0]:
                yield (x[init:end], x[init:end]+1)
                init = end
                end += batch_size

        def conn_it(iterator, dtypes):
            for it in iterator:
                list_conn = ListConn([], dtypes)
                list_conn[0] = it[0]
                list_conn[1] = it[1]
                yield list_conn

        b_it = BatchIterator.from_batchs(conn_it(iterator(x), dtypes), length=len(x), from_batch_size=batch_size,
                                      dtypes=dtypes, to_slice=True)
        init = 0
        end = batch_size
        for e in b_it:
            self.assertEqual((e.batch.to_ndarray()[:, 0] == x[init:end]).all(), True)
            self.assertEqual((e.batch.to_ndarray()[:, 1] == x[init:end]+1).all(), True)
            init = end
            end += batch_size



class TestIteratorBatch(unittest.TestCase):

    def test_iteration_batch(self):
        array = np.arange(0, 10)
        it = Iterator(array).batchs(chunks=(3, ))
        for slice_obj in it:
            self.assertEqual(type(slice_obj), Slice)
            self.assertEqual((slice_obj.batch[it.groups[0]].to_ndarray() == array[slice_obj.slice]).all(), True)

    def test_mixtype_batch(self):
        array = [1, 2, 3, 4.0, 'xxx', 1, 3, 4, 5]
        np_array = np.asarray(array, dtype=object)
        it = Iterator(array, dtypes=np.dtype([(DEFAUL_GROUP_NAME, np.dtype("object"))])).batchs(chunks=(3, ))
        for slice_obj in it:
            self.assertEqual((slice_obj.batch[it.groups[0]].to_ndarray() == np_array[slice_obj.slice]).all(), True)

    def test_mixtype_multidim_batch(self):
        array = [1, 2, 3, 4.0, 'xxx', [1], [[2, 3]]]
        np_array = np.asarray(array, dtype=object)
        it = Iterator(array, dtypes=np.dtype([(DEFAUL_GROUP_NAME, np.dtype("object"))])).batchs(chunks=(3, ))
        for slice_obj in it:
            self.assertEqual((slice_obj.batch[it.groups[0]].to_ndarray() == np_array[slice_obj.slice]).all(), True)

    def test_batch_dtype(self):
        array = np.random.rand(10, 2)
        dtypes = np.dtype([(DEFAUL_GROUP_NAME, np.dtype("float")), ("g1", np.dtype("float"))])
        it = Iterator(array, dtypes=dtypes).batchs(chunks=(3, ))
        for slice_obj in it:
            self.assertEqual((slice_obj.batch["g1"].to_ndarray() == array[:, 1][slice_obj.slice]).all(), True)
            self.assertEqual((slice_obj.batch[DEFAUL_GROUP_NAME].to_ndarray() == array[:, 0][slice_obj.slice]).all(), True)

    def test_batch_it_attrs(self):
        df = pd.DataFrame({"x": np.arange(0, 10), "y": np.arange(10, 20)})
        it = Iterator(df).batchs(chunks=(3, ))
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.length, 10)
        self.assertEqual(it.shape, (10, 2))
        self.assertEqual(it.batch_size, 3)
        self.assertEqual(it.num_splits(), 4)
        self.assertEqual(it.batch_shape(), [3, 2])
        self.assertEqual((it.groups == df.columns.values).all(), True)

    def test_batch_it_attrs_length(self):
        it = Iterator(stream()).batchs(chunks=(3, ))
        self.assertEqual(it.dtype, int)
        self.assertEqual(it.dtypes, [(DEFAUL_GROUP_NAME, np.dtype('int64'))])
        self.assertEqual(it.length, np.inf)
        self.assertEqual(it.shape, (np.inf,))
        self.assertEqual(it.batch_size, 3)
        self.assertEqual(it.num_splits(), 0)
        self.assertEqual(it.batch_shape(), [3])
        self.assertEqual(it.groups, (DEFAUL_GROUP_NAME,))

    def test_shape(self):
        data = np.random.rand(10)
        it = Iterator(data).batchs(chunks=(2, ))
        self.assertEqual(it.shape, (10,))

        data = np.random.rand(10, 3)
        it = Iterator(data).batchs(chunks=(2, ))
        self.assertEqual(it.shape, (10, 3))

        data = np.random.rand(10, 3, 3)
        it = Iterator(data).batchs(chunks=(2, ))
        self.assertEqual(it.shape, (10, 3, 3))

    def test_batchs_values(self):
        batch_size = 3
        m = np.asarray([[1, '5.0'], [2, '3'], [4, 'C']])
        data = pd.DataFrame(m, columns=['A', 'B'])
        it = Iterator(data).batchs(chunks=(batch_size, ))
        self.assertEqual(it.batch_size, batch_size)
        for smx in it:
            for i, row in enumerate(smx.batch):
                self.assertEqual((row.to_ndarray()[0] == m[i]).all(), True)

    def test_df_batchs(self):
        batch_size = 2
        data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float')
        it = Iterator(data, dtypes=np.dtype([('x', np.dtype('float'))])).batchs(chunks=(batch_size, ))
        batch = next(it).batch
        self.assertCountEqual(batch['x'].to_ndarray(), [1, 2])

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

    def test_buffer(self):
        v = list(range(100))
        it = Iterator(v)
        buffer_size = 7
        i = 0
        j = buffer_size
        for elems in it.batchs(chunks=(buffer_size, )):
            self.assertCountEqual(elems.batch[DEFAUL_GROUP_NAME].to_ndarray(), list(range(i, j)))
            i = j
            j += buffer_size
            if j > 100:
                j = 100

    def test_iterator_cut(self):
        array = np.arange(0, 100)
        it = Iterator(array).batchs(chunks=(3, ))
        for slice_obj in it[:10]:
            self.assertEqual((slice_obj.batch[it.groups[0]].to_ndarray() == array[slice_obj.slice]).all(), True)

    def test_flat_all_batch(self):
        array = np.empty((20, 2), dtype=np.dtype(int))
        array[:, 0] = np.arange(0, 20)
        array[:, 1] = np.arange(0, 20) + 1
        it = Iterator(array).batchs(chunks=(3, 2))
        flat_array = array.reshape(-1)
        for i, e in enumerate(it.flat()):
            self.assertEqual(e, flat_array[i])

    def test_clean_batchs(self):
        it = Iterator(((i, 'X', 'Z') for i in range(20))).batchs(chunks=(2, 3))
        for i, smx in enumerate(it.clean_batchs()):
            self.assertEqual((smx[DEFAUL_GROUP_NAME].to_ndarray() == np.asarray([i, 'X', 'Z'], dtype=object)).all(), True)

    def test_sample_batch(self):
        order = (i for i in range(20))
        array = np.arange(0, 20)
        it = Iterator(order).batchs(chunks=(2, ))
        samples = []
        samples_it = it.sample(5, col=DEFAUL_GROUP_NAME)
        self.assertEqual(isinstance(samples_it, Iterator), True)
        for e in samples_it:
            samples.append(e)
        self.assertEqual((samples == array[:5]).all(), False)

    def test_one_elem(self):
        data = [[1, 2, 'a', 's'], [2, 3, 'c', 'e']]
        dtypes = np.dtype([("a", int), ("b", int), ("c", str), ("s", str)])
        data = Iterator(data, dtypes=dtypes)
        ok_shape = {'a': (2, ), 'b': (2, ), 'c': (2, ), 's': (2, )}
        self.assertEqual(data.shape.items(), ok_shape.items())

        data = [[1, 2, 'a', 's'], [2, 3, 'c', 'e']]
        data = Iterator(data)
        ok_shape = {'g0': (2, 4)}
        self.assertEqual(data.shape.items(), ok_shape.items())

        data = [1, 2, 'a', 's']
        dtypes = np.dtype([("a", int), ("b", int), ("c", str), ("s", str)])
        data = Iterator(data, dtypes=dtypes)
        ok_shape = {'a': (1, ), 'b': (1, ), 'c': (1, ), 's': (1, )}
        self.assertEqual(data.shape.items(), ok_shape.items())

        data = [1, 2, 'a', 's']
        data = Iterator(data)
        ok_shape = {'g0': (4, )}
        self.assertEqual(data.shape.items(), ok_shape.items())


class TestIteratorToData(unittest.TestCase):

    def test_length_array_batch(self):
        array = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                            [13, 14], [15, 16], [17, 18], [19, 20]])
        it = Iterator(array).batchs(chunks=(3, 2))
        with Data(name="test") as data:
            data.from_data(it[:10])
            self.assertEqual((data[:5].to_ndarray() == array[:5]).all(), True)

    def test_stream(self):
        it = Iterator(stream())
        with Data(name="test", chunks=(3, )) as data:
            data.from_data(it[:10])
            self.assertCountEqual(data.to_ndarray(), np.arange(0, 10))

    def test_stream_batchs(self):
        it = Iterator(stream()).batchs(chunks=(3, ))
        with Data(name="test") as data:
            data.from_data(it[:10])
            self.assertCountEqual(data.to_ndarray(), np.arange(0, 10))

    def test_multidtype(self):
        x0 = np.arange(20)
        x1 = (x0 + 1).astype("float")
        x2 = x0 + 2
        df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
        with Data(name="test", chunks=Chunks({"x0": (10, ), "x1": (10, ), "x2": (10, )})) as data:
            data.from_data(df)
            self.assertEqual((data["x0"][:5].to_ndarray() == x0[:5]).all(), True)
            self.assertEqual((data["x1"][:5].to_ndarray() == x1[:5]).all(), True)
            self.assertEqual((data["x2"][:5].to_ndarray() == x2[:5]).all(), True)
            self.assertEqual(data["x0"].dtype, np.dtype(int))
            self.assertEqual(data["x1"].dtype, np.dtype(float))

    def test_structured_batchs(self):
        x0 = np.zeros(20) + 1
        x1 = np.zeros(20) + 2
        x2 = np.zeros(20) + 3
        df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
        with Data(name="test", chunks=(3, )) as data:
            data.from_data(df)
            self.assertEqual((data["x0"].to_ndarray() == x0).all(), True)
            self.assertEqual((data["x1"].to_ndarray() == x1).all(), True)
            self.assertEqual((data["x2"].to_ndarray() == x2).all(), True)

    def test_multidtype_batchs(self):
        x0 = np.zeros(20) + 1
        x1 = (np.zeros(20) + 2).astype("int")
        x2 = np.zeros(20) + 3
        df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
        with Data(name="test", chunks=(3, )) as data:
            data.from_data(df)
            self.assertEqual(data.shape, (20, 3))
            self.assertEqual(data["x0"].dtypes["x0"], float)
            self.assertEqual(data["x1"].dtypes["x1"], int)
            self.assertEqual(data["x2"].dtypes["x2"], float)

    def test_to_ndarray_dtype(self):
        batch_size = 2
        array = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int')
        it = Iterator(array).batchs(batch_size)
        with Data(name="test", chunks=(batch_size, )) as data:
            data.from_data(it)
            array = data.to_ndarray(dtype='float')
            self.assertEqual(array.dtype, np.dtype("float"))
            data.destroy()

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
        with Data(name="test", chunks=(258, 4)) as data:
            data.from_data(it)
            c = collections.Counter(data.to_ndarray()[:, 3])
            self.assertEqual(c[1]/float(num_samples) > .79, True)

    def test_empty(self):
        it = Iterator([])
        with Data(name="test", chunks=(1, )) as data:
            data.from_data(it)
            self.assertCountEqual(data.to_ndarray(), np.asarray([]))
            self.assertEqual(data.shape, (0,))
            data.destroy()

    def test_datetime(self):
        m = [datetime.datetime.today(), datetime.datetime.today(), datetime.datetime.today()]
        df = pd.DataFrame(m, columns=['A'])
        it = Iterator(m, dtypes=np.dtype([("A", np.dtype('<M8[ns]'))])).batchs(chunks=(2, ))
        with Data(name="test") as data:
            data.from_data(it)
            self.assertCountEqual(data.to_ndarray(), df.values)

    def test_abstractds(self):
        array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        with Data(name="test", chunks=(3, )) as data:
            data.from_data(array)
            it = Iterator(data)
            for it_array, array_elem in zip(it, array):
                self.assertEqual(it_array.to_ndarray(), array_elem)
            data.destroy()

    def test_batch_ads(self):
        with Data(name="test", chunks=(3, )) as data:
            data.from_data(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
            array_l = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
            it = Iterator(data).batchs(chunks=(3, 1))
            for batch, array in zip(it.only_data(), array_l):
                self.assertCountEqual(batch, array)
            data.destroy()


class TestIteratorFromData(unittest.TestCase):

    def test_da_group(self):
        x = np.random.rand(10)
        with Data(name="test", chunks=(5, )) as data:
            data.from_data(x)
            it = Iterator(data)
            self.assertEqual(it.shape, (10,))
            self.assertEqual([(g, d) for g, (d, _) in it.dtypes.fields.items()], [(DEFAUL_GROUP_NAME, np.dtype(float))])

    def test_da_group_it(self):
        x = np.random.rand(10)
        with Data(name="test", chunks=(5, )) as data:
            data.from_data(x)
            it = Iterator(data)
            for i, e in enumerate(it):
                self.assertEqual(e.to_ndarray(), x[i])

    def test_da_group_it_batch(self):
        x = np.random.rand(10)
        with Data(name="test", chunks=(5, )) as data:
            data.from_data(x)
            it = Iterator(data).batchs(chunks=(5, ))
            for e in it:
                self.assertEqual((e.batch.to_ndarray() == x[e.slice]).all(), True)


class TestIteratorLoop(unittest.TestCase):
    def test_cycle_it(self):
        array = np.arange(10)
        with Data(name="test", chunks=(3, )) as data:
            data.from_data(array)
            it = Iterator(data).cycle()[:20]
            elems = []
            for e in it:
                elems.append(e.to_ndarray())
        self.assertEqual(elems, list(range(10))*2)

    def test_it2iter(self):
        x_array = np.random.rand(10)
        y_array = np.random.rand(10)
        z_array = np.random.rand(10)
        dagroup_dict = DaGroupDict.convert({"x": x_array, "y": y_array, "z": z_array},
                        chunks=Chunks({"x": (5, ), "y": (5, ), "z": (5, )}))
        with Data(name="test") as data:
            data.from_data(dagroup_dict)
            it = Iterator(data).batchs(chunks=(1, )).cycle().to_iter()
            for i, x_y_z in enumerate(it):
                self.assertEqual(x_y_z[0][0], x_array[i])
                self.assertEqual(x_y_z[0][1], y_array[i])
                self.assertEqual(x_y_z[0][2], z_array[i])
                break

    def test_cycle_it_batch_cut(self):
        x = range(10)
        with Data(name="test", chunks=(3, )) as data:
            data.from_data(x)
            it = Iterator(data).batchs(chunks=(3, )).cycle()[:22]
            elems = []
            for e in it:
                elems.append(e.batch.to_ndarray())

        self.assertCountEqual(elems[0], [0, 1, 2])
        self.assertCountEqual(elems[3], [9])
        self.assertCountEqual(elems[4], [0, 1, 2])
        self.assertCountEqual(elems[7], [9])
        self.assertCountEqual(elems[8], [0, 1])

    def test_from_batchs_to_iterator(self):
        def _it():
            for _ in range(100):
                e = np.random.rand(3, 3)
                yield (e, e)

        it = BatchIterator.from_batchs(_it(), length=100,
                                       dtypes=np.dtype([("x", np.dtype("float")), ("y", np.dtype("float"))]),
                                       from_batch_size=3)
        self.assertEqual(it.shape["x"], (100, 3))
        self.assertEqual(it.shape["y"], (100, 3))


def chunk_sizes(seq):
    return [len(list(row)) for row in seq]


class TestSeq(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(10, 10)

    def test_grouper_chunk_3(self):
        seq = grouper_chunk(3, self.X)
        self.assertEqual(chunk_sizes(seq), [3, 3, 3, 1])

    def test_grouper_chunk_2(self):
        seq = grouper_chunk(2, self.X)
        self.assertEqual(chunk_sizes(seq), [2, 2, 2, 2, 2])

    def test_grouper_chunk_10(self):
        seq = grouper_chunk(10, self.X)
        self.assertEqual(chunk_sizes(seq), [10])

    def test_grouper_chunk_1(self):
        seq = grouper_chunk(1, self.X)
        self.assertEqual(chunk_sizes(seq), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_grouper_chunk_7(self):
        seq = grouper_chunk(7, self.X)
        self.assertEqual(chunk_sizes(seq), [7, 3])
