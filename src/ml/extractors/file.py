#https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress
import csv
import zipfile
import os
#import bz2
#import gzip
from io import StringIO, TextIOWrapper
from ml.utils.files import rm
from operator import itemgetter
from ml.data.it import Iterator, DaskIterator
from ml.data.abc import AbsDataset
from ml.utils.decorators import cache


def get_compressed_file_manager(filepath):
    with file(filepath, 'r') as f:
        start_of_file = f.read(1024)
        f.seek(0)
        for cls in (ZIPFile,):
            if cls.is_magic(start_of_file):
                return cls(filepath)
        return File(filepath)


def get_compressed_file_manager_ext(filepath, engine):
    ext = filepath.split(".").pop()
    for cls in (File, ZIPFile):
        if cls.proper_extension == ext:
            return cls(filepath, engine)


class PandasEngine:
    def read_csv(*args, **kwargs):
        import pandas as pd
        df = pd.read_csv(*args, **kwargs)
        return Iterator(df, chunks_size=kwargs.get('chunksize', 0))


class DaskEngine:
    def read_csv(*args, **kwargs):
        import dask.dataframe as dd
        df = dd.read_csv(*args, **kwargs)
        return DaskIterator(df, chunks_size=kwargs.get('chunksize', 0))


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

    def read(self, columns=None, exclude=False, df=True, **kwargs) -> Iterator:
        if exclude is True:
            cols = lambda col: col not in columns
        elif exclude is False and columns:
            cols = lambda col: col in columns
        else:
            cols = None
        return self.engine.read_csv(self.filepath, usecols=cols, **kwargs)

    def write(self, iterator, header=None, delimiter=",") -> None:
        from tqdm import tqdm
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
        import pandas as pd
        if filename is None:
            return super(ZIPFile, self).read(columns=columns, exclude=exclude, **kwargs)
        else:
            iter_ = self._read_another_file(filename, columns, kwargs.get("delimiter", None))
            dtype = [(col, object) for col in next(iter_)] 
            it = Iterator(iter_, chunks_size=kwargs.get("chunksize", 0), dtype=dtype)
            nrows = kwargs.get("nrows", None)
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
    def __init__(self, filepath, delimiter=",", engine='pandas'):
        self.filepath = filepath
        self.file_manager = get_compressed_file_manager_ext(self.filepath, engine)
        self.delimiter = delimiter

    def __enter__(self):
        return NotImplemented

    def __exit__(self, type, value, traceback):
        return NotImplemented

    def __iter__(self):
        return NotImplemented

    def __getitem__(self, key):
        return NotImplemented

    def __setitem__(self, key, value):
        return NotImplemented

    def __next__(self):
        return NotImplemented

    def chunks_writer(self, name, data, init=0):
        return NotImplemented

    def chunks_writer_split(self, data_key, labels_key, data, labels_column, init=0):
        return NotImplemented

    def url(self):
        return self.filepath

    def exists(self):
        return os.path.exists(self.url())

    def num_features(self):
        if len(self.shape) > 1:
            return self.shape[-1]
        else:
            return 1

    def to_df(self):
        return self.reader().to_memory()

    @staticmethod
    def concat(datasets, chunksize:int=0, name:str=None):
        return NotImplemented

    @property
    @cache
    def columns(self):
        return self.reader(nrows=1).columns

    @property
    @cache
    def shape(self):
        size = sum(df.shape[0] for df in self.reader(nrows=None, 
            delimiter=self.delimiter, chunksize=1000))
        return size, len(self.columns)

    def reader(self, *args, **kwargs):
        kwargs["delimiter"] = self.delimiter
        return self.file_manager.read(*args, **kwargs)
    
    def writer(self, *args, **kwargs):
        return self.file_manager.write(*args, **kwargs)

    def destroy(self):
        rm(self.filepath)
