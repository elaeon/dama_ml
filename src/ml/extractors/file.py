#https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress
import csv
import zipfile
#import bz2
#import gzip
import StringIO
from ml.utils.files import rm
from operator import itemgetter
from ml.layers import IterLayer


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

    def reader(self, header=True, limit=None, filename=None, delimiter=",", columns=None):
        with open(self.filepath, 'rb') as f:
            csv_reader = csv.reader(f, delimiter=delimiter)
            if header is False:
                next(csv_reader)
            if columns is None:
                for i, row in enumerate(csv_reader):
                    if limit is not None and i > limit:
                        break
                    yield row
            else:
                for i, row in enumerate(csv_reader):
                    if limit is not None and i > limit:
                        break
                    yield itemgetter(*columns)(row)

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

    def reader(self, header=True, limit=None, filename=None, delimiter=",", columns=None):
        with zipfile.ZipFile(self.filepath, 'r') as zf:
            files = zf.namelist()
            if len(files) == 1:
                filename = files[0]

            with zf.open(filename, 'r') as f:
                csv_reader = csv.reader(f, delimiter=delimiter)
                if header is False:
                    next(csv_reader)
                if columns is None:
                    for i, row in enumerate(csv_reader):
                        if limit is not None and i > limit:
                            break
                        yield row
                else:
                    for i, row in enumerate(csv_reader):
                        if limit is not None and i > limit:
                            break
                        yield itemgetter(*columns)(row)

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
    def __init__(self, filepath, schema=None, has_header=True, delimiter=",", 
            filename=None):
        self.filepath = filepath
        self.schema = schema
        self.has_header = has_header
        self.delimiter = delimiter
        self.file_manager = get_compressed_file_manager_ext(self.filepath)#get_compressed_file_manager(self.filepath)
        if filename is None:
            self.filename = self.filepath
        else:
            self.filename = filename

    def header(self):
        return list(self.reader(header=True, limit=0, columns=None))[0]

    def columns_index(self):
        if self.schema is not None and self.has_header:
            header = self.header()
            index = []
            for c, _ in self.schema:
                index.append(header.index(c))
            return index

    def columns(self):
         if self.schema is not None:
            return [c for c, _ in self.schema]
         else:
            return self.header()

    def shape(self, filename=None):
        size = sum(1 for _ in self.reader(limit=None, header=not self.has_header))
        return size, len(self.columns())

    def reader(self, header=True, limit=None, columns=None):
        return self.file_manager.reader(header=header, limit=limit, 
            filename=self.filename, delimiter=self.delimiter, columns=columns)
    
    def writer(self, iterator):
        return self.file_manager.writer(iterator, filename=self.filename,
            delimiter=self.delimiter)

    def destroy(self):
        rm(self.filepath)
    
    def to_iter(self):
        return IterLayer(self.reader(header=not self.has_header, 
            columns=self.columns_index()), dtype=self.schema)
