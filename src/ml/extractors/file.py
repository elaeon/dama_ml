#https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress
import csv
import zipfile
#import bz2
#import gzip
from io import StringIO, TextIOWrapper
from ml.utils.files import rm
from operator import itemgetter
from ml.data.it import Iterator


def get_compressed_file_manager(filepath):
    with file(filepath, 'r') as f:
        start_of_file = f.read(1024)
        f.seek(0)
        for cls in (ZIPFile,):
            if cls.is_magic(start_of_file):
                return cls(filepath)
        return File(filepath)


def get_compressed_file_manager_ext(filepath):
    ext = filepath.split(".").pop()
    for cls in (File, ZIPFile):
        if cls.proper_extension == ext:
            return cls(filepath)


class File(object):
    magic = None
    file_type = 'csv'
    mime_type = 'text/plain'
    proper_extension = 'csv'

    def __init__(self, filepath, mode='r'):
        self.filepath = filepath

    def read(self, header=True, nrows=None, delimiter=",", columns=None, 
            exclude=False, chunksize=None) -> Iterator:
        import pandas as pd
        if exclude is True:
            cols = lambda col: col not in columns
        elif exclude is False and columns:
            cols = lambda col: col in columns
        else:
            cols = None
        df = pd.read_csv(self.filepath, chunksize=chunksize, nrows=nrows, 
            usecols=cols, delimiter=delimiter)
        return Iterator(df, chunks_size=chunksize)

    def write(self, iterator: Iterator, delimiter=",") -> None:
        with open(self.filepath, 'w') as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            for row in iterator:
                csv_writer.writerow(row)

    @classmethod
    def is_magic(self, data):
        if self.magic is not None:
            return data.startswith(self.magic)
        return True

    
class ZIPFile(File):
    magic = '\x50\x4b\x03\x04'
    file_type = 'zip'
    mime_type = 'compressed/zip'
    proper_extension = 'zip'

    def read(self, header=True, nrows=None, filename=None, delimiter=",", 
        columns=None, exclude=False, chunksize=None) -> Iterator:
        import pandas as pd
        if filename is None:
            return super(ZIPFile, self).read(header=header, nrows=nrows, delimiter=delimiter,
            columns=columns, exclude=exclude, chunksize=chunksize)
        else:
            iter_ = self._read_another_file(filename, columns, delimiter)
            dtype = [(col, object) for col in next(iter_)] 
            it = Iterator(iter_, 
                chunks_size=chunksize, dtype=dtype)
            if nrows is not None:
                it.set_length(nrows)
            return it

    def _read_another_file(self, filename, columns, delimiter, dtype=None):
        with zipfile.ZipFile(self.filepath, 'r') as zf:
            files = zf.namelist()
            with zf.open(filename, 'r') as f:
                csv_reader = csv.reader(TextIOWrapper(f, encoding="utf8"), delimiter=delimiter)
                yield next(csv_reader)
                if columns is None:
                    for row in csv_reader:
                        yield row
                else:
                    for row in csv_reader:
                        yield itemgetter(*columns)(row)

    def write(self, iterator, filename=None, delimiter=","):
        with zipfile.ZipFile(self.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            output = StringIO()
            csv_writer = csv.writer(output, delimiter=delimiter)
            for row in iterator:
                csv_writer.writerow(row)
            if filename is None:
                filename = self.filepath.split("/")[-1]
                filename = filename.split(".")[:-1]
                if len(filename) == 1:
                    filename = "{}.csv".format(filename[0])
                else:
                    filename = ".".join(filename)
            zf.writestr(filename, output.getvalue())
            output.close()


#class BZ2File (CompressedFile):
#    magic = '\x42\x5a\x68'
#    file_type = 'bz2'
#    mime_type = 'compressed/bz2'

#    def open(self):
#        return bz2.BZ2File(self.f)


#class GZFile (CompressedFile):
#    magic = '\x1f\x8b\x08'
#    file_type = 'gz'
#    mime_type = 'compressed/gz'

#    def open(self):
#        return gzip.GzipFile(self.f)


class CSV(object):
    def __init__(self, filepath, delimiter=","):
        self.filepath = filepath
        self.file_manager = get_compressed_file_manager_ext(self.filepath)
        self.delimiter = delimiter

    def columns(self):
        return self.reader(nrows=1).columns()

    @property
    def shape(self):
        size = sum(1 for _ in self.reader(nrows=None, delimiter=self.delimiter))
        return size, len(self.columns())

    def reader(self, *args, **kwargs):
        kwargs["delimiter"] = self.delimiter
        return self.file_manager.read(*args, **kwargs)
    
    def writer(self, *args, **kwargs):
        return self.file_manager.write(*args, **kwargs)

    def destroy(self):
        rm(self.filepath)
