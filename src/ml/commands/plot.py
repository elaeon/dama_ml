import os

from ml.utils.config import get_settings
from ml.data.ds import Data
settings = get_settings("ml")


def run(args):
    if args.dataset:
        dataset = Data.original_ds(args.dataset)

    with dataset:
        dataset.plot(view=args.view, type_g=args.type_g, columns=args.columns)
