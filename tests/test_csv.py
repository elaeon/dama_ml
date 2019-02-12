import unittest

#from ml.data.csv import ZIPFile, get_compressed_file_manager_ext


#class TestCSVZip(unittest.TestCase):
#    def setUp(self):
#        self.iterator = [
#            [1, 2, 3, 4, 5],
#            [6, 7, 8, 9, 0],
#            [0, 9, 8, 7, 6]]
#        self.filepath = "/tmp/test.zip"
#        csv = ZIPFile(self.filepath)
#        csv.write(self.iterator, header=["A", "B", "C", "D", "F"], delimiter=",")

#    def tearDown(self):
#        csv = ZIPFile(filepath=self.filepath)
#       csv.destroy()

#    def test_csv_ext(self):
#        fm = get_compressed_file_manager_ext(self.filepath)
#        self.assertEqual(fm.mime_type, ZIPFile.mime_type)

#    def test_reader(self):
#        csv = ZIPFile(filepath=self.filepath)
#        for r0, r1 in zip(self.iterator, csv.read()):
#            self.assertCountEqual(r0, r1)

#    def test_reader_another_file(self):
#        csv = ZIPFile(self.filepath)
#        it = csv.read(filename="test.csv", delimiter=",")
#        for row0, row1 in zip(self.iterator, it):
#            self.assertEqual(list(map(str, row0)), row1)
#        self.assertCountEqual(it.groups, ["A", "B", "C", "D", "F"])
#        csv.destroy()

#    def test_only_columns(self):
#        csv = ZIPFile(self.filepath)
#        it = csv.read(columns=["B", "C"])
#        self.assertCountEqual(it.groups, ["B", "C"])


if __name__ == '__main__':
    unittest.main()
