import unittest
import zipfile

from ml.extractors.file import CSV, ZIPFile, File, get_compressed_file_manager_ext


class TestCSVZip(unittest.TestCase):
    def setUp(self):
        self.iterator = [
            ["A", "B", "C", "D", "F"], 
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [0, 9, 8, 7, 6]]
        self.filepath = "/tmp/test.zip"
        self.filename = "test.csv"
        csv_writer = CSV(filepath=self.filepath, delimiter=",")
        csv_writer.writer(self.iterator)

    def tearDown(self):
        csv = CSV(filepath=self.filepath)
        csv.destroy()

    def test_csv_ext(self):
        fm = get_compressed_file_manager_ext(self.filepath)
        self.assertEqual(fm.mime_type, ZIPFile.mime_type)

    def test_reader(self):
        csv = CSV(self.filepath)
        for r0, r1 in zip(self.iterator[1:], csv.reader(nrows=2)):
            self.assertCountEqual(r0, r1)

        step = 1
        for r1 in csv.reader(nrows=None, chunksize=2):
            for base_r, r1r in zip(self.iterator[step:step+2], r1.values):
                self.assertCountEqual(base_r, r1r)
            step += 2

    def test_reader_another_file(self):
        csv = CSV(self.filepath)
        it = csv.reader(nrows=2, filename="test.csv")
        print(it.to_memory(3))
        print(it.columns())

    def test_columns(self):
        csv = CSV(self.filepath)
        self.assertCountEqual(csv.columns(), ["A", "B", "C", "D", "F"])

    def test_shape(self):
        csv = CSV(self.filepath)
        self.assertCountEqual(csv.shape, (3, 5))

    def test_only_columns(self):
        csv = CSV(self.filepath)
        it = csv.reader(columns=["B", "C"])
        self.assertCountEqual(it.columns(), ["B", "C"])

    def test_exclude_columns(self):
        csv = CSV(self.filepath)
        it = csv.reader(columns=["A", "C"], exclude=True)
        self.assertCountEqual(it.columns(), ["B", "D", "F"])


class TestCSV(unittest.TestCase):
    def setUp(self):
        self.iterator = [
            ["A", "B", "C", "D", "F"], 
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [0, 9, 8, 7, 6]]
        self.filepath = "/tmp/test.csv"
        csv_writer = CSV(filepath=self.filepath)
        csv_writer.writer(self.iterator, delimiter=",")

    def tearDown(self):
        csv = CSV(filepath=self.filepath)
        csv.destroy()

    def test_csv_ext(self):
        fm = get_compressed_file_manager_ext(self.filepath)
        self.assertEqual(fm.mime_type, File.mime_type)

    def test_reader(self):
        csv = CSV(self.filepath)
        for r0, r1 in zip(self.iterator[1:], csv.reader(nrows=2)):
            self.assertCountEqual(r0, r1)

        step = 1
        for r1 in csv.reader(nrows=None, chunksize=2):
            for base_r, r1r in zip(self.iterator[step:step+2], r1.values):
                self.assertCountEqual(base_r, r1r)
            step += 2

    def test_columns(self):
        csv = CSV(self.filepath)
        self.assertCountEqual(csv.columns(), ["A", "B", "C", "D", "F"])

    def test_shape(self):
        csv = CSV(self.filepath)
        self.assertCountEqual(csv.shape, (3, 5))

    def test_only_columns(self):
        csv = CSV(self.filepath)
        it = csv.reader(columns=["B", "C"])
        self.assertCountEqual(it.columns(), ["B", "C"])

    def test_exclude_columns(self):
        csv = CSV(self.filepath)
        it = csv.reader(columns=["A", "C"], exclude=True)
        self.assertCountEqual(it.columns(), ["B", "D", "F"])


if __name__ == '__main__':
    unittest.main()
