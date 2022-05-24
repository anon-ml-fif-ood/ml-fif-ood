import numpy as np
from sklearn.svm import OneClassSVM

from utils.helpers import timer

from ._abc import AggregationABC


class OneClassSVMAgg(AggregationABC):
    def __init__(self):
        self.gamma = "auto"
        self.clf = OneClassSVM(gamma="auto")

    @timer
    def fit(self, X, pred=None, *args, **kwargs):
        assert isinstance(X, np.ndarray)
        self.clf.fit(X)

    @timer
    def forward(self, x, pred=None, *args, **kwargs):
        assert isinstance(x, np.ndarray)
        return self.clf.score_samples(x)

    def __repr__(self) -> str:
        return f"OneClassSVMAgg(gamma={self.gamma})"


class ClassCondOneClassSVMAgg(AggregationABC):
    def __init__(self) -> None:
        self.gamma = "auto"

    @timer
    def fit(self, X, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(X, np.ndarray)
        self.n_classes = int(np.unique(pred).max() + 1)
        self.clf = {}
        for c in range(self.n_classes):
            self.clf[c] = OneClassSVM(gamma=self.gamma).fit(X[pred == c, :])

        assert self.clf[0] != self.clf[1]

    @timer
    def forward(self, x, pred, *args, **kwargs):
        assert isinstance(pred, np.ndarray)
        assert isinstance(x, np.ndarray)
        scores = []
        for c in range(self.n_classes):
            if len(x[pred == c, :] > 0):
                scores.append(self.clf[c].score_samples(x[pred == c, :]))
        scores = np.concatenate(scores)
        assert len(scores) == len(x)
        return scores

    def __repr__(self) -> str:
        return f"ClassCondClassCondOneClassSVMAgg(gamma={self.gamma})"
