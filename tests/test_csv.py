import unittest

from ml.data.csv import CSVDataset, ZIPFile, File, get_compressed_file_manager_ext
from ml.data.csv import DaskEngine


class TestCSVZip(unittest.TestCase):
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

    def test_csv_ext(self):
        fm = get_compressed_file_manager_ext(self.filepath, 'pandas')
        self.assertEqual(fm.mime_type, ZIPFile.mime_type)

    def test_reader(self):
        csv = CSVDataset(self.filepath)
        for r0, r1 in zip(self.iterator, csv.reader(nrows=2)):
            self.assertCountEqual(r0, r1)

        step = 0
        for r1 in csv.reader(nrows=None, chunksize=2):
            for base_r, r1r in zip(self.iterator[step:step+2], r1.values):
                self.assertCountEqual(base_r, r1r)
            step += 2

    def test_reader_another_file(self):
        csv = CSVDataset(self.filepath)
        it = csv.reader(nrows=2, filename="test.csv")
        self.assertCountEqual(it.labels, ["A", "B", "C", "D", "F"])
        csv.destroy()

    def test_labels(self):
        csv = CSVDataset(self.filepath)
        self.assertCountEqual(csv.labels, ["A", "B", "C", "D", "F"])

    def test_shape(self):
        csv = CSVDataset(self.filepath)
        self.assertCountEqual(csv.shape, (None, 5))

    def test_only_columns(self):
        csv = CSVDataset(self.filepath)
        it = csv.reader(columns=["B", "C"])
        #self.assertCountEqual(it.columns, ["B", "C"])


class TestCSV(unittest.TestCase):
    def setUp(self):
        self.iterator = [
            ["A", "B", "C", "D", "F"], 
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [0, 9, 8, 7, 6]]
        self.filepath = "/tmp/test.csv"
        csv_writer = CSVDataset(filepath=self.filepath)
        csv_writer.from_data(self.iterator, delimiter=",")

    def tearDown(self):
        csv = CSVDataset(filepath=self.filepath)
        csv.destroy()

    def test_csv_ext(self):
        fm = get_compressed_file_manager_ext(self.filepath, 'pandas')
        self.assertEqual(fm.mime_type, File.mime_type)

    def test_reader(self):
        csv = CSVDataset(self.filepath)
        for r0, r1 in zip(self.iterator[1:], csv.reader(nrows=2)):
            self.assertCountEqual(r0, r1)

        step = 1
        for r1 in csv.reader(nrows=None, chunksize=2):
            for base_r, r1r in zip(self.iterator[step:step+2], r1.values):
                self.assertCountEqual(base_r, r1r)
            step += 2

    def test_labels(self):
        csv = CSVDataset(self.filepath)
        self.assertCountEqual(csv.labels, ["A", "B", "C", "D", "F"])

    def test_shape(self):
        csv = CSVDataset(self.filepath)
        self.assertCountEqual(csv.shape, (None, 5))

    def test_only_columns(self):
        csv = CSVDataset(self.filepath)
        #it = csv.reader(columns=["B", "C"])
        #self.assertCountEqual(it.columns, ["B", "C"])

    def test_engine(self):
        csv = CSVDataset(self.filepath, engine="dask")
        it = csv.reader(columns=["A", "C"], exclude=True)
        self.assertEqual(csv.file_manager.engine, DaskEngine)
        df = it.to_memory()
        self.assertCountEqual(df.columns, ["B", "D", "F"])

    def test_to_df(self):
        csv = CSVDataset(self.filepath)
        #self.assertEqual(csv.to_df().shape, (3, 5))

    def test_to_ndarray(self):
        csv = CSVDataset(self.filepath)
        #array = np.asarray(self.iterator[1:]).astype("float32")
        #self.assertEqual((csv.to_ndarray(dtype="float32") == array).all(), True)


if __name__ == '__main__':
    unittest.main()
