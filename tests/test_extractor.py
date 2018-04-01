import unittest
import zipfile
import csv
import StringIO

from ml.extractors.file import CSV


class TestCSV(unittest.TestCase):
    def setUp(self):
        with zipfile.ZipFile("/tmp/test.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            output = StringIO.StringIO()
            csv_writer = csv.writer(output, delimiter=",")
            header = ["A", "B", "C", "D", "F"]
            csv_writer.writerow(header)
            row = [1, 2, 3, 4, 5]
            csv_writer.writerow(row)
            zf.writestr("test.csv", output.getvalue())
            output.close()

    def test_csv(self):
        csv = CSV("/tmp/test.zip")
        for row in csv.stream(limit=1):
            print(row)


if __name__ == '__main__':
    unittest.main()
