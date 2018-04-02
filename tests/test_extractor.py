import unittest
import zipfile

from ml.extractors.file import CSV


class TestCSV(unittest.TestCase):
    def setUp(self):
        self.iterator = [["A", "B", "C", "D", "F"], [1, 2, 3, 4, 5]]
        csv_writer = CSV(filepath="/tmp/test.zip", filename="test.csv")
        csv_writer.writer(self.iterator)

    def test_csv(self):
        csv = CSV("/tmp/test.zip")
        for r0, r1 in zip(self.iterator, csv.reader(limit=1)):
            self.assertItemsEqual(map(str, r0), r1)


if __name__ == '__main__':
    unittest.main()
