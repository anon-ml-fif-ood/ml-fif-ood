from time import time
from typing import Dict

import numpy as np
import torch


def dict2numpy(features_scores: Dict[str, torch.Tensor]) -> np.ndarray:
    return torch.hstack(list(features_scores.values())).detach().cpu().numpy()


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func
