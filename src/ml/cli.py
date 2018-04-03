import argparse

from ml import __version__
from ml.commands import dataset, models, plot, config, reader

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version=__version__)
subparsers = parser.add_subparsers()

dataset_parser = subparsers.add_parser('datasets')
dataset_parser.add_argument("--info", type=str, help="list of datasets or dateset info if a name is added")
dataset_parser.add_argument("--sparcity", type=str, help="show the dataset sparcity")
dataset_parser.add_argument("--rm", nargs="+", type=str, help="delete the dataset")
dataset_parser.add_argument("--clean", action="store_true", help="clean orphans elements")
dataset_parser.add_argument("--used-in", type=str, help="find if the models are using a specific dataset")
dataset_parser.add_argument("--remove-outlayers", type=str, help="remove the outlayers from the dataset")
dataset_parser.add_argument("--sts", type=str, help="show stadistic analysis of the dataset")
dataset_parser.set_defaults(func=dataset.run)

model_parser = subparsers.add_parser('models')
model_parser.add_argument("--info", type=str, help="list of datasets or dateset info if a name is added")
model_parser.add_argument("--rm", nargs="+", type=str, help="delete elements")
model_parser.add_argument("--measure", type=str, help="select the metric")
model_parser.add_argument("--meta", action="store_true", help="print the model's metadata")
model_parser.set_defaults(func=models.run)

plot_parser = subparsers.add_parser('plot')
plot_parser.add_argument("--dataset", type=str, help="dataset to plot")
plot_parser.add_argument("--view", type=str, help="analyze column or row")
plot_parser.add_argument("--type-g", type=str, help="graph type")
plot_parser.add_argument("--columns", type=str, help="columns to compare")
plot_parser.set_defaults(func=plot.run)

header_parser = subparsers.add_parser('reader')
header_parser.add_argument("--nrows", type=int, help="number of rows to show", default=1)
header_parser.add_argument("--file", type=str, help="filepath to file", nargs="+")
header_parser.set_defaults(func=reader.run)

config_parser = subparsers.add_parser('config')
config_parser.set_defaults(func=config.run)


def main():
    """Main CLI entrypoint."""
    args = parser.parse_args()
    args.func(args)
