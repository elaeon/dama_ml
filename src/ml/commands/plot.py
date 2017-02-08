import os

from ml.utils.config import get_settings
from ml.utils.order import order_table_print
from ml.utils.numeric_functions import humanize_bytesize
from ml.clf.wrappers import DataDrive
from ml.clf.measures import ListMeasure
from ml.ds import DataSetBuilder, DataLabel
from ml.utils.files import rm, get_models_path
from ml.utils.files import get_date_from_file, get_models_from_dataset

settings = get_settings("ml")

  
def run(args):
    if args.dataset:
        dataset = DataSetBuilder(args.dataset)

    dataset.plot(view="columns", type_g=args.type_g)
    dataset.close_reader()
