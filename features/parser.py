import argparse

"""argument parser for model, dataset name, etc."""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str.upper,
    default="bitsr101_ilsvrc2012",
)
parser.add_argument(
    "--dataset",
    type=str.upper,
    default="textures",
    help="dataset name",
)
parser.add_argument(
    "--train",
    action="store_true",
    help="save train dataset features",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=100,
    help="batch size",
)
parser.add_argument(
    "--save-root",
    type=str,
    default="./tensors",
    help="directory to save features",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="random seed",
)
args = parser.parse_args()
