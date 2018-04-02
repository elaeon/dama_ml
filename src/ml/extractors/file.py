import csv
import zipfile
import StringIO


class CSV(object):
    def __init__(self, filepath, filename=None, delimiter=","):
        self.filepath = filepath
        self.delimiter = delimiter
        self.filename = filename

    def reader(self, limit=None):
        with zipfile.ZipFile(self.filepath, 'r') as zf:
            files = zf.namelist()
            if len(files) == 1:
                filename = files[0]
            else:
                filename = self.filename

            with zf.open(filename, 'r') as f:
                csv_reader = csv.reader(f, delimiter=self.delimiter)
                for i, row in enumerate(csv_reader):
                    if limit is not None and i > limit:
                        break
                    yield row

    def writer(self, iterator):
        with zipfile.ZipFile(self.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            output = StringIO.StringIO()
            csv_writer = csv.writer(output, delimiter=self.delimiter)
            for row in iterator:
                csv_writer.writerow(row)
            zf.writestr(self.filename, output.getvalue())
            output.close()
