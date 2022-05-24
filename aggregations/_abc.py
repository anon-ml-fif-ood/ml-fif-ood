from abc import ABC, abstractmethod
from typing import Any


class AggregationABC(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X, *args, **kwargs):
        ...

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        ...

    def __call__(self, x, *args: Any, **kwargs: Any) -> Any:
        return self.forward(x, *args, **kwargs)
