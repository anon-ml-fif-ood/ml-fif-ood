import torch
import torch.nn.functional as F

from ._abc import LiteScore


def msp(logits):
    return torch.max(F.softmax(logits, dim=1), dim=1)[0]


def msp_calibrated(logits, temperature):
    return torch.max(F.softmax(logits / temperature, dim=1), dim=1)[0]


class MSP(LiteScore):
    """Maxiumum softmax prediction OOD detector."""

    def __init__(self):
        super().__init__("MSP")

    def fit(self, X, *args, **kwargs):
        return

    def forward(self, x, *args, **kwargs):
        return msp(x).reshape(-1, 1)


class MSPCalibrated(LiteScore):
    """Maximum softmax prediction with temperature calibration OOD detector."""

    def __init__(self):
        super().__init__("MSP Calibrated")

    def fit(self, X, *args, **kwargs):
        return

    def forward(self, x, temperature, *args, **kwargs):
        return msp_calibrated(x, temperature).reshape(-1, 1)
