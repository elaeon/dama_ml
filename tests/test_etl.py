import unittest
import numpy as np
import datetime
import dask.array as da
import os
import pandas as pd

from ml.data.it import Iterator
from ml.data.etl import Pipeline
from ml.utils.files import check_or_create_path_dir
from ml.data.ds import Data
from ml.data.drivers.core import Zarr, HDF5
from ml.data.csv import ZIPFile

# from dask import get # single thread
# from dask.multiprocessing import get
# from dask.threaded import get


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


def add_many(x, y, z):
    return x + y + z


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


def line(x, a=0, b=1):
    return a*x + b


TMP_PATH = check_or_create_path_dir(os.path.dirname(os.path.abspath(__file__)), 'softstream_data_test')


class TestETL(unittest.TestCase):
    def test_pipeline_01(self):
        pipeline = Pipeline(4)
        a = pipeline.map(inc)
        c = a.map(dec)
        f = c.map(lambda x: x*4)
        b = pipeline.map(ident)
        d = pipeline.fold(c, b).map(add).map(str).map(float)
        pipeline.fold(d, f).map(add)

        for result in pipeline.compute():
            self.assertEqual(result, 24.0)

    def test_pipeline_02(self):
        it = [4, 1, 3]
        pipeline = Pipeline(it)
        pipeline.map(inc_it)
        pipeline.map(sum_it).map(str)

        results = pipeline.compute()
        self.assertEqual(list(results[0]), [5, 2, 4])
        self.assertEqual(results[1], '8')

    def test_pipeline_03(self):
        it = Iterator([1, 2, 3, 4, 5])
        pipeline = Pipeline(it)
        pipeline.map(str_it)
        results = list(pipeline.compute()[0])
        self.assertCountEqual(results, ['1', '2', '3', '4', '5'])

    def test_stream(self):
        it = Iterator(stream())
        pipeline = Pipeline(it[:10])
        pipeline.map(str_it)
        results = list(pipeline.compute()[0])
        self.assertCountEqual(results, [e for e in list(map(str, np.arange(0, 10)))])

    def test_temperature_sensor_stream(self):
        it = Iterator(temperature_sensor_stream()).window(200)
        pipeline = Pipeline(it)
        pipeline.map(mov_avg_it)
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
        d = pipeline.fold(a, b).map(add)
        self.assertEqual(a.compute(), 5)
        self.assertEqual(b.compute(), 3)
        self.assertEqual(c.compute(), 9)
        self.assertEqual(d.compute(), 8)
        self.assertEqual(pipeline.compute(), (9, 8))

    def test_pipeline_distinct_sources(self):
        pipeline = Pipeline(4)
        pipeline.map(inc)
        self.assertEqual(pipeline.compute()[0], 5)
        pipeline.feed(5)
        self.assertEqual(pipeline.compute()[0], 6)

    def test_dask_graph_map_values(self):
        data = 4
        values = np.asarray([1, 2, 3])
        pipeline = Pipeline(data)
        a = pipeline.map(inc)
        b = pipeline.map(dec, values)
        c = pipeline.fold(a, b).map(add)
        self.assertEqual((c.compute() == (values + 4)).all(), True)

    def test_dask_graph_da(self):
        x = np.array(range(1000))
        darray = da.from_array(x, chunks=(100,))
        pipeline = Pipeline(darray)
        a = pipeline.map(op_da_array)
        shape = a.compute().compute().shape
        self.assertEqual(shape, (1000,))

    #def test_graph(self):
    #    pipeline = Pipeline(None)
    #    pipeline.map(ident)
    #    rm("/tmp/stream.svg")
    #    pipeline.visualize(filename="/tmp/stream", format="svg")
    #    self.assertEqual(os.path.exists("/tmp/stream.svg"), True)

    def test_to_json(self):
        pipeline = Pipeline(1)
        a = pipeline.map(ident)
        b = pipeline.map(ident)
        c = pipeline.fold(a, b).map(add)
        pre_json_value = pipeline.compute()
        json_stc = pipeline.to_json()
        pipeline = Pipeline.load(json_stc, os.path.dirname(__file__))
        self.assertEqual(pre_json_value, pipeline.compute())

    def test_static_kwargs(self):
        pipeline = Pipeline(1)
        a = pipeline.map(line, kwargs=dict(a=2, b=1)).map(inc)
        b = pipeline.map(inc)
        c = pipeline.fold(a, b).map(add)
        self.assertEqual(a.compute(), 4)
        self.assertEqual(b.compute(), 2)
        self.assertEqual(c.compute(), 6)

    def test_to_json_kwargs(self):
        pipeline = Pipeline(1)
        a = pipeline.map(line, kwargs=dict(a=2, b=1)).map(inc)
        b = pipeline.map(inc)
        pipeline.fold(a, b).map(add)
        json_stc = pipeline.to_json()
        pipeline_cl = Pipeline.load(json_stc, os.path.dirname(__file__))
        self.assertEqual(pipeline.compute()[0], pipeline_cl.compute()[0])

    def test_store_da_array(self):
        x = np.asarray(range(10))
        array = da.from_array(x, chunks=(2,))
        array = array + 1
        with Data(name="test", dataset_path="/tmp/", driver=Zarr()) as data:
            data.from_data(array)
            self.assertEqual((data[:5].to_ndarray() == (x[:5] + 1)).all(), True)
            data.destroy()

    def test_pipeline_store_array(self):
        x = np.random.rand(5, 2)
        with Data(name="test_store_pipeline", dataset_path=TMP_PATH, driver=HDF5()) as data:
            pipeline = Pipeline(x, shape=x.shape)
            a = pipeline.map(inc)
            b = pipeline.map(dec)
            pipeline.fold(a, b).map(add)
            pipeline.store(data)
            self.assertEqual(data.to_ndarray().shape, x.shape)
            data.destroy()

    def test_pipeline_store_df(self):
        x = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 1, 1]})
        with Data(name="test_store_pipeline", dataset_path=TMP_PATH, driver=HDF5()) as data:
            pipeline = Pipeline(x, shape=x.shape)
            a = pipeline.map(inc)
            b = pipeline.map(dec)
            pipeline.fold(a, b).map(add)
            pipeline.store(data)
            self.assertEqual(data.to_ndarray().shape, x.shape)
            data.destroy()

    def test_short_etl(self):
        data = np.asarray([
            ["a", 1, 0.1],
            ["b", 2, 0.2],
            ["c", 3, 0.3],
            ["d", 4, 0.4],
            ["e", 5, 0.5],
            ["f", 6, 0.6],
            ["g", 7, 0.7],
            ["h", 8, 0.8],
            ["i", 9, 0.9],
            ["j", 10, 1],
            ["k", 11, 1.1],
            ["l", 12, 1.2],
        ], dtype="O")

        self.filepath = "/tmp/test.zip"
        pipeline = Pipeline(data)
        # csv_write = pipeline.map(write_to_csv, kwargs=dict(filepath=self.filepath, header=["letra", "nat", "real"], delimiter=","))
        # csv = pipeline.compute()[0]
        # for e in csv.read(batch_size=5):
        #    self.assertEqual((e.iloc[4].values == data[4]).all(), True)
        #    break
        #csv.destroy()


def write_to_csv(data, filepath=None, header=None, delimiter=None):
    csv = ZIPFile(filepath)
    csv.write(data, header=header, delimiter=delimiter)
    return csv
