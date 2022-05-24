import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from .bit import resnetv2
from .vit.modeling import CONFIGS, VisionTransformer

load_dotenv(".env")
PRE_TRAINED_DIR = os.path.expanduser(
    os.environ.get("PRE_TRAINED_DIR", "vision/pre_trained")
)


def densenet121(num_classes: int, pre_trained: bool = True):
    return torchvision.models.densenet121(pretrained=pre_trained)


class Architectures(Enum):
    DENSENET121 = partial(densenet121)

    def __str__(self):
        return self.name

    def __call__(self, *args):
        return self.value(*args)


ARCHITECTURES = list(Architectures.__members__.keys())


def get_architecture(model_name: str, num_classes: int) -> torch.nn.Module:
    model_name = model_name.upper()
    return Architectures[model_name](num_classes)


DENSENET_FEATURES_NODES = {
    "features.conv0": "features.conv0",
    "features.transition1.conv": "features.transition1.conv",
    "features.transition2.conv": "features.transition2.conv",
    "features.transition3.conv": "features.transition3.conv",
    "relu": "relu",
    "classifier": "linear",
}

RESNET_FEATURES_NODES = [
    # node_name: user-specified key for output dict
    "conv1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "linear",
]


def load_model_from_tensor(path, map_location=torch.device("cpu")):
    model = torch.load(path, map_location=map_location)
    model.eval()
    return model


def load_model_from_state_dict(
    path: Path, model: torch.nn.Module, map_location=torch.device("cpu")
):
    parameters = torch.load(path, map_location=map_location)
    model.load_state_dict(parameters, strict=False)
    model.to(map_location)
    model.eval()
    return model


def save_model(model: torch.nn.Module, path):
    model = model.module if hasattr(model, "module") else model
    torch.save(model.state_dict(), path)


def number_of_parameters(model: torch.nn.Module):
    """Returns the number of trainable parameters of a torch module in MB."""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


@dataclass
class ModelInfo(ABC):
    name: str
    num_classes: int
    dataset: str
    get: Callable
    feature_nodes: list = None
    path: str = None
    model_path: str = None

    def get_model(self, path=None, *args, **kwargs):
        model = self.get(self.num_classes)
        if path is not None:
            if os.path.exists(path):
                print(f"Loading model from memory ({path})")
                model = load_model_from_state_dict(path, model, map_location="cpu")
            else:
                raise ValueError("File path ({path}) not found!")
        return model

    def get_penultimate_feature_node(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_all_feature_nodes(self, linear=False):
        ...


class DensenetBC121ImageNet(ModelInfo):
    def __init__(self):
        super().__init__(
            name=r"DenseNet-121\\(ImageNet)",
            num_classes=1000,
            dataset="ilsvrc2012",
            get=densenet121,
        )
        self.model_path = ""

    def get_model(self, path=None, pre_trained=True, *args, **kwargs):
        model = self.get(self.num_classes, pre_trained)
        return model

    def get_all_feature_nodes(self, linear=False):
        nodes = [
            "features.transition1.pool",
            "features.transition2.pool",
            "features.transition3.pool",
            "flatten",
        ]
        if linear:
            nodes.append("classifier")
        return nodes

    def test_transformation(self):
        img_size = 224
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform_test


class VIT(ModelInfo):
    def __init__(
        self, name, num_classes, dataset, model_type: str = None, model_path=None
    ):
        super().__init__(name=name, num_classes=num_classes, dataset=dataset, get=None)
        self.model_type = model_type
        self.model_path = model_path

    def get_all_feature_nodes(self, linear=False):
        # features: layer_0 layer_1 layer_2 layer_3 layer_4 layer_5 layer_6 layer_7 layer_8 layer_9 layer_10 layer_11 layer_12 layer_13 layer_14 layer_15 layer_16 layer_17 layer_18 layer_19 layer_20 layer_21 layer_22 layer_23 encoder_output
        raise NotImplementedError()

    def get_model(self, path=None, pre_trained=True, *args, **kwargs):
        model_type = get_model_type(self.model_type)
        config = CONFIGS[model_type]
        model = VisionTransformer(config, img_size=224, num_classes=self.num_classes)
        if pre_trained:
            if self.dataset.lower() == "ilsvrc2012":
                model.load_from(np.load(path))
            else:
                model.load_state_dict(
                    torch.load(path, map_location=torch.device("cpu"))
                )
        return model

    def test_transformation(self):
        img_size = 224
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform_test


class BITSR101(ModelInfo):
    def __init__(self):
        super().__init__(
            name=r"\multirowcell{2}{BiT-S R101\\(ImageNet)}",
            num_classes=1000,
            dataset="ilsvrc2012",
            get=None,
        )
        self.model_type = "BiT-S-R101x1"
        self.model_path = f"{PRE_TRAINED_DIR}/BITSR101/BiT-S-R101x1.npz"

    def get_all_feature_nodes(self, linear=False):
        raise NotImplementedError()

    def get_model(self, path=None, pre_trained=True, *args, **kwargs):
        model = resnetv2.KNOWN_MODELS[self.model_type](head_size=self.num_classes)
        # file_list = glob.glob(f"vision/pre_trained/**/*{self.model_type}*.npz")
        # filepath = file_list[0]
        if pre_trained:
            w = np.load(path)
            model.load_from(w)
        return model

    def test_transformation(self):
        img_size = 480
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform_test


class BITMR101(ModelInfo):
    def __init__(self):
        super().__init__(
            name=r"\multirowcell{2}{BiT-M R101\\(ImageNet)}",
            num_classes=1000,
            dataset="ilsvrc2012",
            get=None,
        )

    def get_all_feature_nodes(self, linear=False):
        raise NotImplementedError()


class Models(Enum):

    DENSENET121_ILSVRC2012 = DensenetBC121ImageNet()
    VIT32_ILSVRC2012 = VIT(
        name=r"\multirowcell{2}{ViT-B 32\\(ImageNet)}",
        num_classes=1000,
        dataset="ilsvrc2012",
        model_type="vit32_",
    )

    VIT16_ILSVRC2012 = VIT(
        name=r"\multirowcell{2}{ViT-B 16\\(ImageNet)}",
        num_classes=1000,
        dataset="ilsvrc2012",
        model_type="vit16_",
        model_path="vision/pre_trained/VITB16/imagenet/ViT-B_16-224.npz",
    )
    VIT16L_ILSVRC2012 = VIT(
        name=r"\multirowcell{2}{ViT-L 16\\(ImageNet)}",
        num_classes=1000,
        dataset="ilsvrc2012",
        model_type="vit16l_",
        model_path="vision/pre_trained/VITL16/imagenet/ViT-L_16-224.npz",
    )
    BITSR101_ILSVRC2012 = BITSR101()
    BITMR101_ILSVRC2012 = BITMR101()

    @staticmethod
    def names():
        return list(map(lambda c: c.name, Models))

    @staticmethod
    def pretty_names():
        return {
            c.name: c.value.name if hasattr(c.value, "name") else "" for c in Models
        }


MODEL_NAMES = Models.names()
MODEL_NAMES_PRETTY = Models.pretty_names()


def _get_model_enum(model_name: str):
    return Models[model_name.upper()]


def get_model_name(model_name: str) -> ModelInfo:
    return _get_model_enum(model_name).name


def get_model_info(model_name: str) -> ModelInfo:
    return _get_model_enum(model_name).value


def get_in_dataset_name(model_name: str) -> str:
    return get_model_info(model_name).dataset.upper()


def get_num_classes(model_name: str):
    return get_model_info(model_name.upper()).num_classes


def get_pre_trained_model_by_name(
    model_name: str, root: str = PRE_TRAINED_DIR, pre_trained=True, seed=1, path=None
) -> torch.nn.Module:
    model_name = get_model_name(model_name)
    if path is None:
        path = os.path.join(root, model_name.upper() + "_" + str(seed) + ".pt")
    model_info = get_model_info(model_name)
    return model_info.get_model(path, pre_trained=pre_trained)


def get_model_features_nodes_by_name(nn_name, all_conv_nodes, linear_nodes):
    model_info = get_model_info(nn_name)

    if all_conv_nodes == False:
        nodes = model_info.feature_nodes
        if linear_nodes == False:
            for ln in ["fc", "linear", "classifier"]:
                if ln in nodes:
                    nodes.remove(ln)
    else:
        nodes = model_info.get_all_feature_nodes(linear_nodes)
    return nodes


def get_feature_extractor_by_name(
    nn_name: str, all_conv_nodes=True, linear_nodes=False, pre_trained=True, seed=1
) -> torch.nn.Module:
    model_info = get_model_info(nn_name)
    model_name = get_model_name(nn_name)

    if pre_trained:
        model = get_pre_trained_model_by_name(model_name, seed=seed)
    else:
        model = model_info.get_model(None)

    nodes = get_model_features_nodes_by_name(nn_name, all_conv_nodes, linear_nodes)
    print(nodes)
    return create_feature_extractor(model, return_nodes=nodes)


def pretty_name_mapper(model_name: str):
    try:
        return get_model_info(model_name.upper()).name
    except:
        return model_name.upper()


def get_data_transform(nn_name: str, train: bool = False):
    model_info = get_model_info(nn_name)
    if train:
        return model_info.train_transformation()
    return model_info.test_transformation()


def get_model_type(model_name: str):
    model_name = model_name.lower()
    if "vit16_" in model_name:
        return "ViT-B_16"
    if "vit32_" in model_name:
        return "ViT-B_32"
    if "vit16l_" in model_name:
        return "ViT-L_16"
    if "bitsr101_" in model_name:
        return "BiT-S-R101x1"
    if "bitmr101_" in model_name:
        return "BiT-M-R101x1"


def get_model_path(model_name: str):
    model_info = get_model_info(model_name)
    return model_info.model_path
