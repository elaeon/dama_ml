import csv
from ml import fmtypes
from ml.utils.files import filename_n_ext_from_path


class CSV(object):
    def __init__(self, file_paths=None, delimiter=None, merge_field="id"):
        self.file_paths = file_paths
        self.delimiter = delimiter
        self.merge_field = merge_field

    def get_sep_path(self):
        if isinstance(self.file_paths, list):
            if not isinstance(self.delimiter, list):
                delimiters = [self.delimiter for _ in self.file_paths]
            else:
                delimiters = [self.delimiter]
            return self.file_paths, delimiters
        else:
            return [self.file_paths], [self.sep]

    def stream(self, fmtypes=None):
        from ml.db.utils import build_schema, insert_rows
        file_path, delimiters = self.get_sep_path()
        if fmtypes is None:
            fmtypes = [fmtypes.TEXT]*len(header)

        for data_path, delimiter in zip(file_path, delimiters):
            table_name = filename_n_ext_from_path(data_path)
            with open(data_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=delimiter)
                for row in csv_reader:
                    header = row
                    break
                build_schema(table_name, header, 
                    fmtypes,
                    self.merge_field)
                insert_rows(csv_reader, table_name, header)
