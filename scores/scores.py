from copy import deepcopy
from enum import Enum

from ._abc import LiteScore
from .mahalanobis import MahalanobisLite, MahalanobisMonoClassLite
from .msp import MSP, MSPCalibrated


class ScoresOffline(Enum):
    MAHALANOBIS = MahalanobisLite()
    MAHALANOBIS_MONO_CLASS = MahalanobisMonoClassLite()
    MSP = MSP()
    MSP_CALIBRATED = MSPCalibrated()

    @staticmethod
    def names():
        return list(map(lambda c: c.name, ScoresOffline))

    @staticmethod
    def pretty_names():
        return {
            c.name: c.value.name if hasattr(c.value, "name") else c.name
            for c in ScoresOffline
        }


LITE_SCORES_NAMES = ScoresOffline.names()
SCORES_NAMES_PRETTY = ScoresOffline.pretty_names()


def get_lite_score_by_name(detector_name: str, *args, **kwargs) -> LiteScore:
    return deepcopy(ScoresOffline[detector_name.upper()].value)
