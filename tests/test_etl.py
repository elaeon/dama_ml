import unittest
import numpy as np
import datetime

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


def mov_avg(values):
    avg = 0
    n = len(values)
    for t_stamp, temp in values:
        avg += temp / n
    return avg


def calc_hash(batch):
    h = Hash()
    h.update(batch)
    return str(h)


def write_csv(it):
    filepath = "/tmp/test.csv"
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

        for r in pipeline.compute():
            self.assertEqual(r[0], 24.0)

    def test_pipeline_02(self):
        it = Iterator([4])
        pipeline = Pipeline(it)
        a = pipeline.map(inc)
        d = pipeline.zip(1, a, 2).map(sum_).map(str)

        for r in pipeline.compute():
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
        results = list(pipeline.compute())
        self.assertCountEqual(results, [['1'], ['2'], ['3'], ['4'], ['5'], ['6']])

    def test_stream(self):
        it = Iterator(stream())
        pipeline = Pipeline(it[:10])
        a = pipeline.map(str)
        results = list(pipeline.compute())
        self.assertCountEqual(results, [[e] for e in list(map(str, np.arange(0, 10)))])

    def test_temperature_sensor_stream(self):
        it = Iterator(temperature_sensor_stream()).window(200)
        pipeline = Pipeline(it)
        a = pipeline.map(mov_avg)
        counter = 0
        for e in pipeline.compute():
            self.assertEqual(10 <= e[0] <= 13, True)
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
        for r in pipeline.compute():
            print("--", r)

        filepath = "/tmp/test.csv"
        csv_writer = CSVDataset(filepath=filepath, delimiter=",")
        #for e in csv_writer:
        #    print(e)
        csv_writer.destroy()

    #def test_stream2(self):
    #    csv = CSVDataset(self.filepath)
    #    csv.map(plus)
    #    data = Data(name="test", dataset_path="/tmp")
    #    data.from_data(csv, chunksize=3, length=2)
        #reg = Xgboost(name="test")
        #reg.set_dataset(csv)
        #reg.train()
        #reg.save(model_version="1")

    #def test_graph(self):
        #s.visualize_task_graph(filename="stream", format="svg")
        #s.visualize_graph()


