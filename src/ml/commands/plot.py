from ml.utils.config import get_settings

settings = get_settings("paths")


def run(args):
    from ml.data.ds import Data
    if args.dataset:
        dataset = Data.original_ds(args.dataset)

    with dataset:
        dataset.plot(view=args.view, type_g=args.type_g, columns=args.columns)
