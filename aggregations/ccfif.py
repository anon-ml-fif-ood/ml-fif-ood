import fif as FIF
import numpy as np

from utils.helpers import timer

from ._abc import AggregationABC


class ClassCondFunctionalIsolationForest(AggregationABC):
    def __init__(
        self, ntrees=200, sample_size=128, dic_number=1, alpha=1, seed=0
    ) -> None:
        self.ntrees = ntrees
        self.sample_size = sample_size
        self.dic_number = dic_number
        self.alpha = alpha
        self.limit = np.log2(self.sample_size)
        self.seed = seed
        self.forest = {}

    @timer
    def fit(self, X, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(X, np.ndarray)
        self.n_classes = int(np.unique(pred).max() + 1)
        n, layers = X.shape
        tps = np.linspace(0, 1, layers)
        for c in range(self.n_classes):
            # fix seed
            self.forest[c] = FIF.FiForest(
                X[pred == c, :].astype("double"),
                time=tps,
                ntrees=self.ntrees,
                sample_size=self.sample_size,
                limit=self.limit,
                dic_number=self.dic_number,
                alpha=self.alpha,
                seed=self.seed,
            )

        assert self.forest[0] != self.forest[1]
        print("limit", self.forest[0].limit, "ntrees", self.forest[0].ntrees)

    @timer
    def forward(self, x, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(x, np.ndarray)
        scores = []
        for c in range(self.n_classes):
            if len(x[pred == c, :]) > 0:
                scores.append(
                    self.forest[c].compute_paths(x[pred == c, :].astype("double"))
                )
        scores = np.concatenate(scores)
        assert len(scores) == len(x)
        return scores

    def __repr__(self) -> str:
        return f"ClassCondFunctionalIsolationForest(ntrees={self.ntrees}, sample_size={self.sample_size}, dic_number={self.dic_number}, alpha={self.alpha}, limit={self.limit})"
