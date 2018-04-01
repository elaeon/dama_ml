import csv
from zipfile import ZipFile


class CSV(object):
    def __init__(self, filepath, filename=None, delimiter=","):
        self.filepath = filepath
        self.delimiter = delimiter

    def stream(self, limit=None):
        with ZipFile(self.filepath, 'r') as zf:
            files = zf.namelist()
            if len(files) == 1:
                filename = files[0]
            else:
                pass

            with zf.open(filename, 'r') as f:
                csv_reader = csv.reader(f, delimiter=self.delimiter)
                for i, row in enumerate(csv_reader):
                    print(row)
                    if limit is not None and i > limit:
                        break
