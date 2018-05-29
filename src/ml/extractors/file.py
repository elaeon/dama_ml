#https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress
import csv
import zipfile
#import bz2
#import gzip
import StringIO
from ml.utils.files import rm
import pandas as pd


def get_compressed_file_manager(filepath):
    with file(filepath, 'rb') as f:
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

    def __init__(self, filepath):
        self.filepath = filepath

    @classmethod
    def is_magic(self, data):
        if self.magic is not None:
            return data.startswith(self.magic)
        return True

    def reader(self, header=True, limit=None, filename=None, delimiter=","):
        with open(self.filepath, 'rb') as f:
            csv_reader = csv.reader(f, delimiter=delimiter)
            if header is False:
                next(csv_reader)
            for i, row in enumerate(csv_reader):
                if limit is not None and i > limit:
                    break
                yield row

    def writer(self, iterator, filename=None, delimiter=","):
        with open(self.filepath, 'w') as f:
            csv_writer = csv.writer(f, delimiter=delimiter)
            for row in iterator:
                csv_writer.writerow(row)


class ZIPFile(File):
    magic = '\x50\x4b\x03\x04'
    file_type = 'zip'
    mime_type = 'compressed/zip'
    proper_extension = 'zip'

    def reader(self, header=True, limit=None, filename=None, delimiter=","):
        with zipfile.ZipFile(self.filepath, 'r') as zf:
            files = zf.namelist()
            if len(files) == 1:
                filename = files[0]

            with zf.open(filename, 'r') as f:
                csv_reader = csv.reader(f, delimiter=delimiter)
                if header is False:
                    next(csv_reader)
                for i, row in enumerate(csv_reader):
                    if limit is not None and i > limit:
                        break
                    yield row

    def writer(self, iterator, filename=None, delimiter=","):
        with zipfile.ZipFile(self.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            output = StringIO.StringIO()
            csv_writer = csv.writer(output, delimiter=delimiter)
            for row in iterator:
                csv_writer.writerow(row)
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
    def __init__(self, filepath):
        self.filepath = filepath

    def header(self, filename=None, delimiter=","):
        return list(self.reader(limit=0, filename=filename, delimiter=delimiter))[0]

    def shape(self, header=False, filename=None):
        size = sum(1 for _ in self.reader(limit=None, header=header, filename=filename))
        return size, len(self.header())

    def reader(self, header=True, limit=None, filename=None, delimiter=","):
        file_manager = get_compressed_file_manager(self.filepath)
        if filename is None:
            filename = self.filepath
        return file_manager.reader(header=header, limit=limit, filename=filename,
            delimiter=delimiter)
    
    def writer(self, iterator, filename=None, delimiter=","):
        file_manager = get_compressed_file_manager_ext(self.filepath)
        if filename is None:
            filename = self.filepath
        return file_manager.writer(iterator, filename=filename, 
            delimiter=delimiter)

    def destroy(self):
        rm(self.filepath)
    
    def to_iter(self, dtype=None):
        from ml.layers import IterLayer
        return IterLayer(self.reader(header=False), dtype=dtype)
