from inspect import getmembers, isclass
import argparse
from pydoc import locate

from ml import __version__
from ml.commands import dataset, models

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version=__version__)
subparsers = parser.add_subparsers()

dataset_parser = subparsers.add_parser('dataset')
dataset_parser.add_argument("--info", type=str, help="list of datasets or dateset info if a name is added")
dataset_parser.add_argument("--rm", type=str, help="delete elements")
dataset_parser.add_argument("--clean", action="store_true", help="clean orphans elements")
dataset_parser.add_argument("--used-in", type=str, help="find if the models are using a specific dataset")
dataset_parser.set_defaults(func=dataset.run)

model_parser = subparsers.add_parser('models')
model_parser.add_argument("--info", type=str, help="list of datasets or dateset info if a name is added")
model_parser.add_argument("--rm", type=str, help="delete elements")
model_parser.set_defaults(func=models.run)

#parser.add_argument("--models", action="store_true")
#parser.add_argument("--dataset", action="store_true")
#parser.add_argument("--info", type=str, help="name")
#parser.add_argument("--rm", type=str, help="delete elements")



def main():
    """Main CLI entrypoint."""
    args = parser.parse_args()
    args.func(args)
    # Here we'll try to dynamically match the command the user is trying to run
    #print(dir(args))
    #for k, v in args._get_kwargs():
    #    print(k, v)
    #    if v is True:
    #        pass
            #command = locate('ml.commands.'+k)
        #command.run(args)
