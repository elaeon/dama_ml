import glob
from tabulate import tabulate


def run(args):
    from ml.data.csv import get_compressed_file_manager_ext
    filepaths = check_filepaths(args.file)
    for filepath in filepaths:
        csv = get_compressed_file_manager_ext(filepath)
        reader = csv.reader(nrows=args.nrows)
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
