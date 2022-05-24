import json
import os
from dataclasses import dataclass, field

import pandas as pd

TENSORS_DIR = "tensors"
RESULTS_DIR = "results"


@dataclass
class Config:
    nn_name: str.upper = None
    in_dataset: str.upper = None
    out_dataset: str.upper = None
    method: str = None
    logits_flag: bool = None
    softmax_flag: bool = None
    reduce: str = None
    aggregation: str = None
    temperature: float = 1.0
    eps: float = 0
    checkpoint_number: int = None
    features_names: list = field(default_factory=list)

    def from_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return str(vars(self))


def append_results_to_file(results, filename="results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results = {k: [v] for k, v in results.items()}
    results = pd.DataFrame.from_dict(results, orient="columns")

    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)


def get_in_tensors_path(config: Config):
    path = os.path.join(TENSORS_DIR, config.nn_name, config.in_dataset)
    os.makedirs(path, exist_ok=True)
    return path


def get_out_tensors_path(config: Config):
    path = os.path.join(TENSORS_DIR, config.nn_name, config.out_dataset)
    os.makedirs(path, exist_ok=True)
    return path


def get_metadata_path(config: Config):
    path = os.path.join(TENSORS_DIR, config.nn_name)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "metadata.json")


def get_matadata(config: Config):
    path = get_metadata_path(config)
    with open(path, "r") as f:
        metadata = json.load(f)
    return metadata


"""Features"""


def get_features_name_deprecated(config: Config):
    # DEPRECATED
    name = "all_features_nodes"
    name += (
        f"_{config.checkpoint_number}" if config.checkpoint_number is not None else ""
    )
    return name


def get_features_name(config: Config):
    name = "feature"
    name += (
        f"_{config.checkpoint_number}" if config.checkpoint_number is not None else ""
    )
    return name


def get_penultimate_feature_name(config: Config):
    name = "penultimate_feature"
    name += (
        f"_{config.checkpoint_number}" if config.checkpoint_number is not None else ""
    )
    return name


def get_train_features_path_deprecated(config: Config):
    # DEPRECATED
    name = get_features_name_deprecated(config)
    name += "_train"
    return os.path.join(get_in_tensors_path(config), name + ".pt")


def get_train_features_path(config: Config, feature_name: str):
    name = get_features_name(config)
    name += "_train"
    return os.path.join(get_in_tensors_path(config), feature_name, name + ".pt")


def get_train_penultimate_feature_path(config: Config):
    name = get_penultimate_feature_name(config)
    return os.path.join(get_in_tensors_path(config), name + ".pt")


def get_in_features_path(config: Config):
    # DEPRECATED
    name = get_features_name_deprecated(config)
    return os.path.join(get_in_tensors_path(config), name + ".pt")


# def get_in_features_path(config: Config, feature_name: str):
#     name = get_features_name(config)
#     return os.path.join(get_in_tensors_path(config), feature_name, name + ".pt")


def get_in_penultimate_feature_path(config: Config):
    name = get_penultimate_feature_name(config)
    return os.path.join(get_in_tensors_path(config), name + ".pt")


def get_out_features_path(config: Config):
    # DEPRECATED
    name = get_features_name_deprecated(config)
    return os.path.join(get_out_tensors_path(config), name + ".pt")


def get_out_penultimate_feature_path(config: Config):
    name = get_penultimate_feature_name(config)
    return os.path.join(get_out_tensors_path(config), name + ".pt")


"""logits"""


def _get_base_logits_name(config: Config):
    name = "logits"
    name += (
        f"_{config.checkpoint_number}" if config.checkpoint_number is not None else ""
    )
    return name


def get_train_logits_path(config: Config):
    name = _get_base_logits_name(config) + "_train"
    return os.path.join(get_in_tensors_path(config), name + ".pt")


def get_train_targets_path(config: Config):
    name = "targets_train"
    return os.path.join(get_in_tensors_path(config), name + ".pt")


def get_in_logits_path(config: Config):
    name = _get_base_logits_name(config)
    return os.path.join(get_in_tensors_path(config), name + ".pt")


def get_in_targets_path(config: Config):
    name = "targets"
    return os.path.join(get_in_tensors_path(config), name + ".pt")


def get_out_logits_path(config: Config):
    name = _get_base_logits_name(config)
    return os.path.join(get_out_tensors_path(config), name + ".pt")


"""scores"""


def _get_scores_path(path, config: Config, name: str):
    os.makedirs(path, exist_ok=True)
    # name += "_all" if config.all_features else "_block"
    # name += "_logits" if config.logits_flag else ""
    name += "_{}".format(config.reduce) if config.reduce else ""
    name += (
        f"_{config.checkpoint_number}" if config.checkpoint_number is not None else ""
    )
    return os.path.join(path, name + ".pt")


def get_train_scores_path(config: Config):
    path = os.path.join(get_in_tensors_path(config), config.method)
    name = "scores_train"
    return _get_scores_path(path, config, name)


def get_train_scores_path_feature(config: Config, feature_name: str):
    path = os.path.join(get_in_tensors_path(config), feature_name, config.method)
    name = "scores_train"
    return _get_scores_path(path, config, name)


def get_misclassif_train_scores_path_feature(config: Config, feature_name: str):
    path = os.path.join(get_in_tensors_path(config), feature_name, config.method)
    name = "misclassif_scores_train"
    return _get_scores_path(path, config, name)


def get_in_scores_path(config: Config):
    path = os.path.join(get_in_tensors_path(config), config.method)
    name = "scores_in"
    return _get_scores_path(path, config, name)


def get_misclassif_in_scores_path(config: Config):
    path = os.path.join(get_in_tensors_path(config), config.method)
    name = "misclassif_scores_in"
    return _get_scores_path(path, config, name)


def get_out_scores_path(config: Config):
    path = os.path.join(get_out_tensors_path(config), config.method)
    name = "scores_out"
    return _get_scores_path(path, config, name)


def get_misclassif_out_scores_path(config: Config):
    path = os.path.join(get_out_tensors_path(config), config.method)
    name = "misclassif_scores_out"
    return _get_scores_path(path, config, name)


def get_agg_in_scores_path(config: Config, suffix: str):
    path = os.path.join(get_in_tensors_path(config), config.method)
    name = f"in_agg_{suffix}"
    return _get_scores_path(path, config, name)


def get_agg_out_scores_path(config: Config, suffix: str):
    path = os.path.join(get_out_tensors_path(config), config.method)
    name = f"out_agg_{suffix}"
    return _get_scores_path(path, config, name)
