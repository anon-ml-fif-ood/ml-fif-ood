from abc import ABC, abstractmethod

import torch


class LiteScore(ABC):
    """
    Lite abstract class for OOD detectors.
    """

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def fit(self, X, *args, **kwargs) -> None:
        return

    def fit_logits(self, X, *args, **kwargs) -> None:
        return self.fit(X, *args, **kwargs)

    @abstractmethod
    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        return

    def forward_logits(self, x, *args, **kwargs) -> torch.Tensor:
        return self.forward(x, *args, **kwargs)

    @classmethod
    def reduce(
        cls,
        stack: torch.Tensor,
        mode: str = "none",
        **kwargs,
    ) -> torch.Tensor:
        """Feature wise score reduction.

        Args:
            stack (torch.Tensor): unreduced scores stack.
            mode (str, optional): either "min", "max", "mean", "none" or "pred". Defaults to "none".
            pred (torch.Tensor, optional): tensor of predicted classes. Defaults to None.
        """
        mode = mode.lower()

        if mode == "min":
            return torch.min(stack, dim=1)[0].reshape(-1, 1)
        elif mode == "max":
            return torch.max(stack, dim=1)[0].reshape(-1, 1)
        elif mode == "mean":
            return torch.mean(stack, dim=1).reshape(-1, 1)
        elif mode == "sum":
            return torch.sum(stack, dim=1).reshape(-1, 1)
        elif mode == "none":
            return stack
        else:
            raise ValueError("reduce must be either 'pred', 'min', 'mean' or 'max'")
