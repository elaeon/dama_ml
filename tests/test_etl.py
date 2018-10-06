import unittest
import numpy as np

from ml.data.ds import Data, DataLabel
from ml.extractors.file import CSVDataset
from ml.data.it import Iterator
from ml.processing import Transforms
#from ml.processing import Transforms
#from ml.random import sampling_size
#from ml.data.ds import Memory


def plus(x):
    return x + 1

def rest(x):
    return x - 1


class TestETL(unittest.TestCase):
    def setUp(self):
        self.iterator = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [0, 9, 8, 7, 6]]
        self.filepath = "/tmp/test.zip"
        self.filename = "test.csv"
        csv_writer = CSVDataset(filepath=self.filepath, delimiter=",")
        csv_writer.from_data(self.iterator, header=["A", "B", "C", "D", "F"])

    def tearDown(self):
        csv = CSVDataset(filepath=self.filepath)
        csv.destroy()

    def test_stream(self):
        csv = CSVDataset(self.filepath)
        it = csv.reader(nrows=2)#, chunksize=3)
        data = Data(name="test", dataset_path="/tmp")
        data.transforms.add(plus)
        data.from_data(it)
        with data:
            print(data.data[:])
        data.destroy()

    def test_stream2(self):
        csv = CSVDataset(self.filepath)
        csv.map(plus)
        data = Data(name="test", dataset_path="/tmp")
        data.from_data(csv, chunksize=3, length=2)
        #reg = Xgboost(name="test")
        #reg.set_dataset(csv)
        #reg.train()
        #reg.save(model_version="1")

    def test_branch_stream(self):
        stream = CSVDataset(self.filepath)
        a = stream.map(plus)
        b = stream.map(rest)
        print(stream._graph)
        stream.destroy()