import unittest
import numpy as np
import datetime

from dask.threaded import get
from ml.data.csv import CSVDataset
from ml.data.it import Iterator
from ml.data.etl import Pipeline
from ml.utils.basic import Hash


def inc(x):
    return x + 1


def dec(x):
    return x - 1


def add(x, y):
    return x + y


def sum_(*x):
    return sum(x)


def op_da_array(da):
    x = da + 1
    y = x*2
    return y


def ident(x):
    return x


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
        it = Iterator([4])
        pipeline = Pipeline(it)
        a = pipeline.map(inc)
        c = a.map(dec)
        f = c.map(lambda x: x*4)
        b = pipeline.map(ident)
        d = pipeline.zip(c, b).map(add).map(str).map(float)
        g = pipeline.zip(d, f).map(add)

        for r in pipeline.compute_delays():
            self.assertEqual(r[0], 24.0)

    def test_pipeline_02(self):
        it = Iterator([4])
        pipeline = Pipeline(it)
        a = pipeline.map(inc)
        d = pipeline.zip(1, a, 2).map(sum_).map(str)

        for r in pipeline.compute_delays():
            self.assertEqual(r[0], '8')


    def test_pipeline_03(self):
        def stream():
            t = 0
            while True:
                t += 1
                yield t
                if t > 5:
                    break
        it = Iterator(stream())
        pipeline = Pipeline(it)
        a = pipeline.map(str)
        results = list(pipeline.compute_delays())
        self.assertCountEqual(results, [['1'], ['2'], ['3'], ['4'], ['5'], ['6']])

    def test_stream(self):
        it = Iterator(stream())
        pipeline = Pipeline(it[:10])
        a = pipeline.map(str)
        results = list(pipeline.compute_delays())
        self.assertCountEqual(results, [[e] for e in list(map(str, np.arange(0, 10)))])

    def test_temperature_sensor_stream(self):
        it = Iterator(temperature_sensor_stream()).window(200)
        pipeline = Pipeline(it)
        a = pipeline.map(mov_avg_it)
        dask_graph = pipeline.to_dask_graph()
        counter = 0
        for avg in get(dask_graph, 'mov_avg_it-fn'):
            self.assertEqual(10 <= avg <= 13, True)
            if counter == 1:
                break
            counter += 1

    def test_temperature_sensor_stream_delays(self):
        it = Iterator(temperature_sensor_stream()).window(200)
        pipeline = Pipeline(it)
        a = pipeline.map(mov_avg_fn)
        counter = 0
        for avg in pipeline.compute_delays():
            self.assertEqual(10 <= avg[0] <= 13, True)
            if counter == 1:
                break
            counter += 1

    def test_pipeline_write_calc_hash(self):
        iterator = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [0, 9, 8, 7, 9]]

        it = Iterator(iterator).batchs(2)
        pipeline = Pipeline(it)
        a = pipeline.map(calc_hash)
        b = pipeline.map(write_csv)
        for r in pipeline.compute_delays():
            print("--", r)

        #filepath = "/tmp/test.csv"
        #csv_writer = CSVDataset(filepath=filepath, delimiter=",")
        #for e in csv_writer:
        #    print(e)
        #csv_writer.destroy()

    def test_dask_graph(self):
        data = 4
        pipeline = Pipeline(data)
        a = pipeline.map(inc).map(ident)
        b = pipeline.map(dec)
        c = pipeline.zip(a, b).map(add)
        dask_graph = pipeline.to_dask_graph()
        self.assertEqual(get(dask_graph, 'dec-fn'), 3)
        self.assertEqual(get(dask_graph, 'ident-fn'), 5)
        self.assertEqual(get(dask_graph, 'add-fn'), 8)

    def test_dask_graph_map_values(self):
        data = 4
        values = np.asarray([1, 2, 3])
        pipeline = Pipeline(data)
        a = pipeline.map(inc)
        b = pipeline.map(dec, with_values=values)
        c = pipeline.zip(a, b).map(add)
        dask_graph = pipeline.to_dask_graph()
        self.assertEqual((get(dask_graph, 'add-fn') == (values + 4)).all(), True)

    def test_dask_graph_da(self):
        import dask.array as da
        x = np.array(range(1000))
        darray = da.from_array(x, chunks=(100,))
        print(darray)
        pipeline = Pipeline(darray)
        a = pipeline.map(op_da_array)
        dask_graph = pipeline.to_dask_graph()
        print(get(dask_graph, 'op_da_array-fn').compute())

    def test_graph(self):
        pipeline = Pipeline(None)
        a = pipeline.map(ident)
        pipeline.visualize(filename="stream", format="svg")
        #print(list(pipeline.compute_delays()))
        #s.visualize_graph()


