import unittest
import zipfile

from ml.extractors.file import CSV, ZIPFile, File, get_compressed_file_manager_ext


class TestCSVZip(unittest.TestCase):
    def setUp(self):
        self.iterator = [["A", "B", "C", "D", "F"], [1, 2, 3, 4, 5]]
        self.filepath = "/tmp/test.zip"
        self.filename = "test.csv"
        csv_writer = CSV(filepath=self.filepath, delimiter=",", filename=self.filename)
        csv_writer.writer(self.iterator)

    def test_csv_ext(self):
        fm = get_compressed_file_manager_ext(self.filepath)
        self.assertEqual(fm.mime_type, ZIPFile.mime_type)

    def test_csv(self):
        csv = CSV(self.filepath)
        for r0, r1 in zip(self.iterator, csv.reader(limit=1)):
            self.assertItemsEqual(map(str, r0), r1)

    def test_header(self):
        csv = CSV(self.filepath)
        self.assertItemsEqual(csv.header(), ["A", "B", "C", "D", "F"])

    def test_shape(self):
        csv = CSV(self.filepath)
        self.assertItemsEqual(csv.shape(), (1, 5))
        csv = CSV(self.filepath, has_header=False)
        self.assertItemsEqual(csv.shape(), (2, 5))

    def test_columns(self):
        schema = [("A", "int"), ("C", "int")]
        csv = CSV(self.filepath, schema=schema)
        self.assertItemsEqual(csv.columns(), ["A", "C"])
        self.assertItemsEqual(csv.columns_index(), [0, 2])
        data = list(csv.reader(columns=csv.columns_index()))
        self.assertItemsEqual(data, [('A', 'C'), ('1', '3')])

    def test_to_iter(self):
        schema = [("A", "int"), ("C", "int")]
        csv = CSV(self.filepath, schema=schema)
        it = csv.to_iter()
        it.set_length(csv.shape()[0])
        data = it.to_memory()
        self.assertItemsEqual(data.values[0], [1, 3])


class TestCSV(unittest.TestCase):
    def setUp(self):
        self.iterator = [["A", "B", "C", "D", "F"], [1, 2, 3, 4, 5]]
        self.filepath = "/tmp/test.csv"
        csv_writer = CSV(filepath=self.filepath, delimiter=",")
        csv_writer.writer(self.iterator)

    def test_csv_ext(self):
        fm = get_compressed_file_manager_ext(self.filepath)
        self.assertEqual(fm.mime_type, File.mime_type)

    def test_csv(self):
        csv = CSV(self.filepath)
        for r0, r1 in zip(self.iterator, csv.reader(limit=1)):
            self.assertItemsEqual(map(str, r0), r1)

    def test_header(self):
        csv = CSV(self.filepath)
        self.assertItemsEqual(csv.header(), ["A", "B", "C", "D", "F"])

    def test_shape(self):
        csv = CSV(self.filepath)
        self.assertItemsEqual(csv.shape(), (1, 5))
        csv = CSV(self.filepath, has_header=False)
        self.assertItemsEqual(csv.shape(), (2, 5))


if __name__ == '__main__':
    unittest.main()
