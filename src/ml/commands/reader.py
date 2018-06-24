import os
import csv
import glob
from tabulate import tabulate
from ml.extractors.file import CSV


def run(args):
    filepaths = check_filepaths(args.file)
    for filepath in filepaths:
        csv = CSV(filepath)
        reader = csv.reader(limit=args.nrows)
        table = []
        headers = []
        for row in reader:
            headers = [e for e in row]
            break
        for row in reader:
            table.append([e for e in row])

        print("FILE", filepath)
        print(tabulate(table, headers))


def check_filepaths(filepaths):
    files = []
    for filepath in filepaths:
        if filepath.find("*."):
            files.extend(glob.glob(filepath))
        else:
            files.append(filepath)
    return files
