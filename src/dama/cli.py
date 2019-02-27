import argparse
from pkg_resources import get_distribution
from dama.commands import dataset, models, plot, config, reader, repo


parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version=get_distribution("soft-stream").version)
subparsers = parser.add_subparsers()

dataset_parser = subparsers.add_parser('datasets')
dataset_parser_group = dataset_parser.add_mutually_exclusive_group()
dataset_parser_group.add_argument("--rm", action="store_true", help="delete the dataset")
dataset_parser_group.add_argument("--sts", action="store_true", help="basic stadistic analysis to the dataset")
dataset_parser_group.add_argument("--info", action="store_true", help="show the author and description")
dataset_parser.add_argument("hash", type=str, help="data hash", nargs="*")
dataset_parser.add_argument("--items", type=str, help="slice syntax to get elems from 0 to n (0:n)")
dataset_parser.add_argument("--driver", type=str, help="driver name")
dataset_parser.add_argument("--group-name", type=str, help="list all datasets who has this group name")
dataset_parser.set_defaults(func=dataset.run)

model_parser = subparsers.add_parser('models')
model_parser.add_argument("--info", type=str, help="list of datasets or dateset info if a name is added")
model_parser.add_argument("--rm", nargs="+", type=str, help="delete elements")
model_parser.add_argument("--measure", type=str, help="select the metric")
model_parser.add_argument("--meta", action="store_true", help="print the model's metadata")
model_parser.set_defaults(func=models.run)

# plot_parser = subparsers.add_parser('plot')
# plot_parser.add_argument("--dataset", type=str, help="dataset to plot")
# plot_parser.add_argument("--view", type=str, help="analyze column or row")
# plot_parser.add_argument("--type-g", type=str, help="graph type")
# plot_parser.add_argument("--columns", type=str, help="columns to compare")
# plot_parser.set_defaults(func=plot.run)

# header_parser = subparsers.add_parser('reader')
# header_parser.add_argument("--nrows", type=int, help="number of rows to show", default=1)
# header_parser.add_argument("--file", type=str, help="filepath to file", nargs="+")
# header_parser.set_defaults(func=reader.run)

config_parser = subparsers.add_parser('config')
config_parser_group = config_parser.add_mutually_exclusive_group()
config_parser_group.add_argument("--init-repo", type=str, help="initialize the repository with the typed name")
config_parser_group.add_argument("--edit", action="store_true", help="edit the values saved in the config file")
config_parser_group.set_defaults(func=config.run)

repo_parser = subparsers.add_parser('repo')
repo_parser.add_argument("name", type=str, help="repository name")

repo_parser_group = repo_parser.add_argument_group()
repo_parser_group.add_argument("--commit-msg", help="commit message")
repo_parser_group.add_argument("--run", type=str, help="exec the file")
repo_parser_group.add_argument("--commit", help="commit id")
repo_parser_group.add_argument("--branch", help="branch")
repo_parser_group.add_argument("--checkout", type=str, help="checkout the file")
repo_parser_group.add_argument("--head", action="store_true")
repo_parser_group.add_argument("--log", action="store_true")
repo_parser_group.set_defaults(func=repo.run)


def main():
    """Main CLI entrypoint."""
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        print(parser.format_help())
