import argparse
from dataclasses import dataclass, field

parser = argparse.ArgumentParser(
    description="Reproduce results.",
    allow_abbrev=True,
)
parser.add_argument(
    "-d",
    "--detector",
    type=str.upper,
    help="OOD score method.",
)
parser.add_argument(
    "-nn",
    "--nn_name",
    type=str.upper,
    help="Neural network architecture",
)
parser.add_argument(
    "-out",
    "--ood-dataset",
    type=str.upper,
    help="OOD dataset name.",
    default="none",
)
parser.add_argument(
    "-outs",
    "--ood-datasets",
    nargs="+",
    default=None,
    help="List of OOD datasets",
)
parser.add_argument(
    "-r",
    "--reduce",
    type=str.lower,
    default="none",
    help="Reduce method.",
    choices=["pred", "mean", "max", "min", "sum", "none"],
)
parser.add_argument(
    "-agg",
    "--aggregation",
    type=str.lower,
    default="fif",
    help="Aggregation method.",
)
parser.add_argument(
    "--dont-save",
    action="store_true",
    help="If set, don't save OOD detection scores to file.",
)
parser.add_argument(
    "-bs",
    "--batch-size",
    type=int,
    default=50000,
    help="Batch size for score calculation.",
)
parser.add_argument(
    "-fns",
    "--features-names",
    nargs="+",
    default=None,
    help="List of feature names to consider.",
)
parser.add_argument(
    "-eps",
    "--eps",
    type=float,
    help="Noise magnitude epsilon value.",
    default=1e-4,
)
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    help="Softmax temperature value.",
    default=1.0,
)
parser.add_argument(
    "--train-ds",
    action="store_true",
    help="If set, consider the training dataset.",
)
parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Path to the finetuned model weights.",
)


@dataclass
class Arguments:
    """enumerate argument parser arguments"""

    nn_name: str.upper = "VIT16_ILSVRC2012"
    detector: str.upper = "mahalanobis_mono_class"
    ood_dataset: str.upper = "none"
    ood_datasets: list = None
    reduce: str = "none"
    aggregation: str = "class_fif"
    temperature: float = 1.0
    features_names: list = field(default_factory=list)
    dont_save: bool = False
    batch_size: int = 50000
    profiling: bool = False
    train_ds: bool = False
    model_path: str = None

    def __init__(self, args: argparse.Namespace):
        """override default arguments"""
        args_dict = vars(args)
        for item, value in args_dict.items():
            setattr(self, item, value)

    def update(self, **kwargs):
        """update arguments"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(vars(self))
