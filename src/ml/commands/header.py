import os
import csv
from tabulate import tabulate
import glob

def run(args):
    filepaths = check_filepaths(args.file)
    for filepath in filepaths:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            table = []
            headers = []
            for row in reader:
                headers = [unicode(e, 'utf-8') for e in row]
                break
            if args.nrows is not None:
                for i, row in enumerate(reader , 1):
                    if i <= args.nrows:
                        table.append([unicode(e, 'utf-8') for e in row])
                    else:
                        break
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
