import unittest
import numpy as np
import datetime
import dask.array as da
import os

from ml.data.csv import CSVDataset
from ml.data.it import Iterator
from ml.data.etl import Pipeline
from ml.utils.basic import Hash
from ml.utils.files import rm


def str_it(it):
    for x in it:
        yield str(x)


def inc_it(it):
    for x in it:
        yield x + 1


def inc(x):
    return x + 1


def dec(x):
    return x - 1


def add(x, y):
    return x + y


def sum_it(x):
    return sum(x)


def op_da_array(da):
    x = da + 1
    y = x*2
    return y


def ident(x):
    return x


def square(x):
    return x**2


def stream():
    i = 0
    while True:
        yield i
        i += 1

def temperature_sensor_stream():
    while True:
        temp = np.random.uniform(20, 3)
        t_stamp = datetime.datetime.today().timestamp()
        yield (t_stamp, temp)


def mov_avg_it(values):
    for window in values:
        avg = 0
        n = len(window)
        for t_stamp, temp in window:
            avg += temp / n
        yield avg


def mov_avg_fn(window):
    avg = 0
    n = len(window)
    for t_stamp, temp in window:
        avg += temp / n
    return avg


def calc_hash(it):
    h = Hash()
    for batch in it:
        h.update(batch)
    return str(h)


def write_csv(it):
    filepath = "/tmp/test.csv"
    print("IT", it, type(it))
    csv_writer = CSVDataset(filepath=filepath, delimiter=",")
    csv_writer.from_data(it, header=["A", "B", "C", "D", "F"])
    return True


class TestETL(unittest.TestCase):
    def test_pipeline_01(self):
        pipeline = Pipeline(4)
        a = pipeline.map(inc)
        c = a.map(dec)
        f = c.map(lambda x: x*4)
        b = pipeline.map(ident)
        d = pipeline.zip(c, b).map(add).map(str).map(float)
        g = pipeline.zip(d, f).map(add)

        for result in pipeline.compute():
            self.assertEqual(result, 24.0)

    def test_pipeline_02(self):
        it = [4, 1, 3]
        pipeline = Pipeline(it)
        a = pipeline.map(inc_it)
        d = pipeline.map(sum_it).map(str)

        results = pipeline.compute()
        self.assertEqual(list(results[0]), [5, 2, 4])
        self.assertEqual(results[1], '8')

    def test_pipeline_03(self):
        it = Iterator([1, 2, 3, 4, 5])
        pipeline = Pipeline(it)
        a = pipeline.map(str_it)
        results = list(pipeline.compute()[0])
        self.assertCountEqual(results, ['1', '2', '3', '4', '5'])

    def test_stream(self):
        it = Iterator(stream())
        pipeline = Pipeline(it[:10])
        a = pipeline.map(str_it)
        results = list(pipeline.compute()[0])
        self.assertCountEqual(results, [e for e in list(map(str, np.arange(0, 10)))])

    def test_temperature_sensor_stream(self):
        it = Iterator(temperature_sensor_stream()).window(200)
        pipeline = Pipeline(it)
        a = pipeline.map(mov_avg_it)
        counter = 0
        for avg in pipeline.compute()[0]:
            self.assertEqual(10 <= avg <= 13, True)
            if counter == 1:
                break
            counter += 1

    def test_dask_graph(self):
        data = 4
        pipeline = Pipeline(data)
        a = pipeline.map(inc)
        b = pipeline.map(dec)
        c = b.map(ident).map(square)
        d = pipeline.zip(a, b).map(add)
        self.assertEqual(a.compute(), 5)
        self.assertEqual(b.compute(), 3)
        self.assertEqual(c.compute(), 9)
        self.assertEqual(d.compute(), 8)
        self.assertEqual(pipeline.compute(), (9, 8))

    def test_dask_graph_map_values(self):
        data = 4
        values = np.asarray([1, 2, 3])
        pipeline = Pipeline(data)
        a = pipeline.map(inc)
        b = pipeline.map(dec, with_values=values)
        c = pipeline.zip(a, b).map(add)
        self.assertEqual((c.compute() == (values + 4)).all(), True)

    def test_dask_graph_da(self):
        x = np.array(range(1000))
        darray = da.from_array(x, chunks=(100,))
        pipeline = Pipeline(darray)
        a = pipeline.map(op_da_array)
        shape = a.compute().compute().shape
        self.assertEqual(shape, (1000,))

    def test_graph(self):
        pipeline = Pipeline(None)
        a = pipeline.map(ident)
        rm("/tmp/stream.svg")
        pipeline.visualize(filename="/tmp/stream", format="svg")
        self.assertEqual(os.path.exists("/tmp/stream.svg"), True)


