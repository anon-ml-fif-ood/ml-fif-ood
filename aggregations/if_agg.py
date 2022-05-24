import numpy as np
from sklearn.ensemble import IsolationForest as IForest

from utils.helpers import timer

from ._abc import AggregationABC


class IsolationForest(AggregationABC):
    def __init__(self, ntrees=100, sample_size=4096) -> None:
        self.ntrees = ntrees
        self.sample_size = sample_size
        self.forest = IForest(n_estimators=self.ntrees, max_samples=self.sample_size)

    @timer
    def fit(self, X, *args, **kwargs):
        assert isinstance(X, np.ndarray)
        self.forest.fit(X)

    @timer
    def forward(self, x, *args, **kwargs):
        assert isinstance(x, np.ndarray)
        return self.forest.score_samples(x)

    def __repr__(self) -> str:
        return f"IsolationForest(ntrees={self.ntrees}, sample_size={self.sample_size})"


class ClassCondIsolationForest(AggregationABC):
    def __init__(self, ntrees=100, sample_size=512) -> None:
        self.ntrees = ntrees
        self.sample_size = sample_size

    @timer
    def fit(self, X, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(X, np.ndarray)
        self.n_classes = int(np.unique(pred).max() + 1)
        self.forest = {}
        for c in range(self.n_classes):
            self.forest[c] = IForest(
                n_estimators=self.ntrees, max_samples=self.sample_size
            ).fit(X[pred == c, :])

        assert self.forest[0] != self.forest[1]

    @timer
    def forward(self, x, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(x, np.ndarray)
        scores = []
        for c in range(self.n_classes):
            if len(x[pred == c, :] > 0):
                scores.append(self.forest[c].score_samples(x[pred == c, :]))
        scores = np.concatenate(scores)
        assert len(scores) == len(x)
        return scores

    def __repr__(self) -> str:
        return f"ClassCondIsolationForest(ntrees={self.ntrees}, sample_size={self.sample_size})"
