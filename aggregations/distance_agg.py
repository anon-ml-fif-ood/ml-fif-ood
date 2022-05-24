import numpy as np

from ._abc import AggregationABC


class MahalanobisAgg(AggregationABC):
    def fit(self, X, *args, **kwargs):
        self.mean = np.mean(X, axis=0, keepdims=True)
        if X.shape[1] == 1:
            self.cov = np.std(X, axis=0, keepdims=True) ** 2
            self.inv = np.diag(1 / self.cov)
        else:
            self.cov = np.cov(X.T)
            self.inv_cov = np.linalg.pinv(self.cov)

    def forward(self, x, *args, **kwargs):
        return -(((x - self.mean) @ self.inv_cov) * (x - self.mean)).sum(axis=1)


class ClassCondMahalanobisAgg(AggregationABC):
    def fit(self, X, pred, *args, **kwargs):
        self.n_classes = int(np.unique(pred).max() + 1)
        self.mean = {}
        self.inv_cov = {}
        for c in range(self.n_classes):
            self.cov = np.cov(X[pred == c].T)
            self.inv_cov[c] = np.linalg.pinv(self.cov)
            self.mean[c] = np.mean(X[pred == c], axis=0, keepdims=True)

    def forward(self, x, pred, *args, **kwargs):
        scores = []
        for c in range(self.n_classes):
            scores.append(
                -(
                    ((x[pred == c] - self.mean[c]) @ self.inv_cov[c])
                    * (x[pred == c] - self.mean[c])
                ).sum(axis=1)
            )
        return np.concatenate(scores)


class EuclidesAgg(AggregationABC):
    def fit(self, X, *args, **kwargs):
        self.mean = np.mean(X, axis=0, keepdims=True)

    def forward(self, x, *args, **kwargs):
        return -np.sqrt(((x - self.mean) ** 2).sum(axis=1))


class ClassCondEuclidesAgg(AggregationABC):
    def fit(self, X, pred, *args, **kwargs):
        self.n_classes = int(np.unique(pred).max() + 1)
        self.mean = {}
        for c in range(self.n_classes):
            self.mean[c] = np.mean(X[pred == c, :], axis=0, keepdims=True)

    def forward(self, x, pred, *args, **kwargs):
        scores = []
        for c in range(self.n_classes):
            scores.append(-np.sqrt(((x[pred == c, :] - self.mean[c]) ** 2).sum(axis=1)))
        return np.concatenate(scores)
