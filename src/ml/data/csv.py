# https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress
import csv
import zipfile
import os
# import bz2
# import gzip
import numpy as np
from io import StringIO, TextIOWrapper
from ml.utils.files import rm
from operator import itemgetter
from ml.data.it import Iterator, BatchIterator
from ml.data.abc import AbsDataset
from ml.utils.decorators import cache
from tqdm import tqdm



def get_compressed_file_manager_ext(filepath, engine):
    ext = filepath.split(".").pop()
    for cls in (File, ZIPFile):
        if cls.proper_extension == ext:
            return cls(filepath, engine)


class PandasEngine:
    def read_csv(*args, **kwargs):
        import pandas as pd
        if "batch_size" in kwargs:
            kwargs['chunksize'] = kwargs['batch_size']
            batch_size = kwargs['batch_size']
            del kwargs['batch_size']
        else:
            batch_size = 0
        df = pd.read_csv(*args, **kwargs)
        it = Iterator(df)
        print(it.shape)
        if batch_size == 0:
            return it
        else:
            return BatchIterator(it, batch_size=batch_size)


class DaskEngine:
    def read_csv(*args, **kwargs):
        import dask.dataframe as dd
        df = dd.read_csv(*args, **kwargs)
        return df # DaskIterator(df)


class File(object):
    magic = None
    file_type = 'csv'
    mime_type = 'text/plain'
    proper_extension = 'csv'

    def __init__(self, filepath, engine):
        self.filepath = filepath
        if engine == "pandas":
            self.engine = PandasEngine
        else:
            self.engine = DaskEngine

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

    def read(self, filename=None, columns=None, exclude=False, df=True, **kwargs) -> Iterator:
        if filename is None:
            return super(ZIPFile, self).read(columns=columns, exclude=exclude, **kwargs)
        else:
            iter_ = self._read_another_file(filename, columns, kwargs.get("delimiter", None))
            dtype = [(col, np.dtype("object")) for col in next(iter_)]
            nrows = kwargs.get("nrows", None)
            it = Iterator(iter_, dtypes=dtype, length=nrows).batchs(batch_size=kwargs.get("batch_size", 0), batch_type="df")
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


class CSVDataset(AbsDataset):
    def __init__(self, filepath, delimiter=",", engine='pandas', filename=None):
        self.filepath = filepath
        self.file_manager = get_compressed_file_manager_ext(self.filepath, engine)
        self.delimiter = delimiter
        self.filename = filename
        self._it = None

    def __enter__(self):
        return NotImplemented

    def __exit__(self, type, value, traceback):
        return NotImplemented

    def __iter__(self):
        return self

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        return NotImplemented

    def __next__(self):
        if self._it is None:
            self._it = self.data
        return next(self._it)

    @property
    def data(self):
        return self.reader(batch_size=10)

    def batchs_writer(self, keys, data, init=0):
        return NotImplemented

    def url(self):
        return self.filepath

    def exists(self):
        return os.path.exists(self.url())

    def to_df(self):
        return NotImplemented

    def to_ndarray(self, dtype=None):
        return NotImplemented

    @property
    @cache
    def labels(self):
        return self.reader(nrows=1).labels

    @property
    @cache
    def shape(self):
        return None, len(self.labels)

    def reader(self, *args, **kwargs):
        kwargs["delimiter"] = self.delimiter
        return self.file_manager.read(*args, **kwargs)
    
    def from_data(self, *args, **kwargs):
        kwargs["delimiter"] = self.delimiter
        return self.file_manager.write(*args, **kwargs)

    def destroy(self):
        rm(self.filepath)
