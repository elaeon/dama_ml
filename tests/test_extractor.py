import unittest
import zipfile

from ml.extractors.file import CSV, ZIPFile, File, get_compressed_file_manager_ext


class TestCSVZip(unittest.TestCase):
    def setUp(self):
        self.iterator = [["A", "B", "C", "D", "F"], [1, 2, 3, 4, 5]]
        self.filepath = "/tmp/test.zip"
        self.filename = "test.csv"
        csv_writer = CSV(filepath=self.filepath)
        csv_writer.writer(self.iterator, filename=self.filename, delimiter=",")

    def test_csv_ext(self):
        fm = get_compressed_file_manager_ext(self.filepath)
        self.assertEqual(fm.mime_type, ZIPFile.mime_type)

    def test_csv(self):
        csv = CSV(self.filepath)
        for r0, r1 in zip(self.iterator, csv.reader(limit=1, filename=self.filename)):
            self.assertItemsEqual(map(str, r0), r1)

    def test_header(self):
        csv = CSV(self.filepath)
        self.assertItemsEqual(csv.header(), ["A", "B", "C", "D", "F"])

    def test_shape(self):
        csv = CSV(self.filepath)
        self.assertItemsEqual(csv.shape(), (1, 5))
        self.assertItemsEqual(csv.shape(header=True), (2, 5))


class TestCSV(unittest.TestCase):
    def setUp(self):
        self.iterator = [["A", "B", "C", "D", "F"], [1, 2, 3, 4, 5]]
        self.filepath = "/tmp/test.csv"
        csv_writer = CSV(filepath=self.filepath)
        csv_writer.writer(self.iterator, delimiter=",")

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
        self.assertItemsEqual(csv.shape(header=True), (2, 5))


if __name__ == '__main__':
    unittest.main()
