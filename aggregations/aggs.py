from enum import Enum

from ._abc import AggregationABC
from .distance_agg import (
    ClassCondEuclidesAgg,
    ClassCondMahalanobisAgg,
    EuclidesAgg,
    MahalanobisAgg,
)
from .ccfif import ClassCondFunctionalIsolationForest
from .if_agg import ClassCondIsolationForest, IsolationForest
from .svm_agg import ClassCondOneClassSVMAgg, OneClassSVMAgg


class Aggregations(Enum):

    CLASS_FIF = ClassCondFunctionalIsolationForest

    MAHALANOBIS = MahalanobisAgg
    CLASS_MAHALANOBIS = ClassCondMahalanobisAgg
    EUCLIDES = EuclidesAgg
    CLASS_EUCLIDES = ClassCondEuclidesAgg
    IF = IsolationForest
    CCIF = ClassCondIsolationForest
    ONE_CLASS_SVM = OneClassSVMAgg
    CLASS_ONE_CLASS_SVM = ClassCondOneClassSVMAgg


def get_aggregation(name: str, *args, **kwargs) -> AggregationABC:
    try:
        return Aggregations[name.upper()].value(*args, **kwargs)
    except:
        return Aggregations[name.upper()].value
