import argparse

import torch

import eval.results as res
from vision import nn as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""parser for model name"""
parser = argparse.ArgumentParser(
    description="Calculate model accuracy on a given dataset.",
    epilog="example: python model_accuracy.py -nn VIT16L_ILSVRC2012",
    allow_abbrev=True,
)
parser.add_argument(
    "-nn",
    "--nn_name",
    type=str.upper,
    help="Neural network architecture",
    choices=models.MODEL_NAMES,
)
parser.add_argument(
    "-t",
    "--train",
    action="store_true",
    help="If set, calculate the train accuracy",
)


def main():
    args = parser.parse_args()
    in_dataset_name = models.get_in_dataset_name(args.nn_name)
    config = res.Config(nn_name=args.nn_name, in_dataset=in_dataset_name)
    if args.train:
        targets = torch.load(res.get_train_targets_path(config), map_location=DEVICE)
        logits = torch.load(res.get_train_logits_path(config), map_location=DEVICE)
    else:
        targets = torch.load(res.get_in_targets_path(config), map_location=DEVICE)
        logits = torch.load(res.get_in_logits_path(config), map_location=DEVICE)

    print(
        "Train" if args.train else "Test",
        "accuracy:",
        (logits.argmax(dim=1) == targets).float().mean().item() * 100,
        "%",
    )


if __name__ == "__main__":
    main()
