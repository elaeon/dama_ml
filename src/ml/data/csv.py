# https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress
import csv
import zipfile
import os
# import bz2
# import gzip
import numpy as np
import pandas as pd
from io import StringIO, TextIOWrapper
from ml.utils.files import rm
from operator import itemgetter
from ml.data.it import Iterator, BatchIterator
from tqdm import tqdm
from ml.utils.decorators import cache



def get_compressed_file_manager_ext(filepath):
    ext = filepath.split(".").pop()
    for cls in (File, ZIPFile):
        if cls.proper_extension == ext:
            return cls(filepath)


class PandasEngine:
    def read_csv(*args, **kwargs):
        if "batch_size" in kwargs:
            kwargs['chunksize'] = kwargs['batch_size']
            batch_size = kwargs['batch_size']
            del kwargs['batch_size']
        else:
            batch_size = 0
        df = pd.read_csv(*args, **kwargs)
        it = Iterator(df)
        if batch_size == 0:
            return it
        else:
            return BatchIterator(it, batch_size=batch_size)


class File(object):
    magic = None
    file_type = 'csv'
    mime_type = 'text/plain'
    proper_extension = 'csv'

    def __init__(self, filepath):
        self.filepath = filepath
        self.engine = PandasEngine

    def read(self, columns=None, exclude: bool=False, df: bool=True, filename: str=None, **kwargs) -> Iterator:
        if exclude is True:
            cols = lambda col: col not in columns
        elif exclude is False and columns:
            cols = lambda col: col in columns
        else:
            cols = None
        return self.engine.read_csv(self.filepath, usecols=cols, **kwargs)

    def write(self, iterator, header=None, delimiter: str=",") -> None:
        with open(self.filepath, 'w') as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            if header is not None:
                csv_writer.writerow(header)
            for row in tqdm(iterator):
                csv_writer.writerow(row)

    @property
    @cache
    def dtypes(self):
        return self.engine.read_csv(self.filepath).dtypes

    @classmethod
    def is_magic(self, data):
        if self.magic is not None:
            return data.startswith(self.magic)
        return True

    def destroy(self):
        rm(self.filepath)

    
class ZIPFile(File):
    magic = '\x50\x4b\x03\x04'
    file_type = 'zip'
    mime_type = 'compressed/zip'
    proper_extension = 'zip'

    def read(self, filename=None, columns=None, exclude=False, batch_type="df", **kwargs) -> Iterator:
        if filename is None and batch_type == "df":
            return super(ZIPFile, self).read(columns=columns, exclude=exclude, **kwargs)
        else:
            iter_ = self._read_another_file(filename, columns, kwargs.get("delimiter"))
            dtype = [(col, np.dtype("object")) for col in next(iter_)]
            nrows = kwargs.get("nrows", None)
            if nrows is None:
                it = Iterator(iter_, dtypes=dtype).batchs(batch_size=kwargs.get("batch_size", 0), batch_type=batch_type)
            else:
                it = Iterator(iter_, dtypes=dtype)[:nrows].batchs(batch_size=kwargs.get("batch_size", 0), batch_type=batch_type)
            return it

    def _read_another_file(self, filename, columns, delimiter):
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

    def write(self, iterator, header=None, filename=None, delimiter=",") -> None:
        from tqdm import tqdm
        with zipfile.ZipFile(self.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            output = StringIO()
            csv_writer = csv.writer(output, delimiter=delimiter)
            if header is not None:
                csv_writer.writerow(header)
            for row in tqdm(iterator):
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
