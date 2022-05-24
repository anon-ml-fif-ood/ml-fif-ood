import torch
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from ._abc import LiteScore


def torch_cov_matrix(X: torch.Tensor, mode="classic"):
    """Compute the covariance matrix of X."""
    device = X.device
    dtype = X.dtype
    if mode.lower() == "classic":
        return torch.cov(X.T)
    if mode.lower() == "robust":
        cov = MinCovDet().fit(X.detach().cpu().numpy())
        return torch.tensor(cov.covariance_, device=device, dtype=dtype)
    if mode.lower() == "empirical":
        cov = EmpiricalCovariance(assume_centered=False).fit(X.detach().cpu().numpy())
        return torch.tensor(cov.covariance_, device=device, dtype=dtype)
    raise NotImplementedError(f"{mode} not available.")


def torch_reduction_matrix(
    X: torch.Tensor, estimator="classic", method="cholesky", cov_precomputed=None
):
    if estimator == "classic":
        sigma = torch.cov(X.T)
    elif estimator == "precomputed":
        sigma = cov_precomputed

    if method == "cholesky":
        C = torch.linalg.cholesky(sigma)
        return torch.linalg.inv(C.T)
    elif method == "SVD":
        u, s, _ = torch.linalg.svd(sigma)
        return u @ torch.diag(torch.sqrt(1 / s))
    elif method == "pseudo":
        return torch.linalg.pinv(sigma)


def class_cond_mus_cov_inv_matrix(
    x: torch.Tensor,
    targets: torch.Tensor,
    cov_mode="classic",
    inv_method="pseudo",
):
    unique_classes = torch.unique(targets).detach().cpu().numpy().tolist()
    class_cond_dot = {}
    class_cond_mean = {}
    for c in unique_classes:
        filt = targets == c
        temp = x[filt]
        class_cond_dot[c] = torch_cov_matrix(temp, cov_mode)
        class_cond_mean[c] = temp.mean(0, keepdim=True)
    cov_mat = sum(list(class_cond_dot.values())) / x.shape[0]
    inv_mat = torch_reduction_matrix(cov_mat, method=inv_method)
    mus = torch.vstack(list(class_cond_mean.values()))
    return mus, cov_mat, inv_mat


def mahalanobis_distance_inv(
    x: torch.Tensor, y: torch.Tensor, inverse: torch.Tensor, axis=0
):
    return torch.nan_to_num(
        torch.sqrt(((x - y).T * (inverse @ (x - y).T)).sum(axis)), 0
    )


def mahalanobis_inv_layer_score(
    x: torch.Tensor,
    mus: torch.Tensor,
    inv: torch.Tensor,
):
    stack = torch.zeros(
        (x.shape[0], mus.shape[0]), device=x.device, dtype=torch.float32
    )
    for i, mu in enumerate(mus):
        stack[:, i] = mahalanobis_distance_inv(x, mu.reshape(1, -1), inv).reshape(-1)

    return stack


class MahalanobisLite(LiteScore):
    """Implement lite detector with mahalanobis distance."""

    def __init__(self):
        super().__init__("Mahalanobis")
        self.mus = None
        self.inv = None

    def fit(self, X: torch.Tensor, labels: torch.Tensor, *args, **kwargs):
        self.mus, _, self.inv = class_cond_mus_cov_inv_matrix(
            X, labels, cov_mode="classic", inv_method="pseudo"
        )
        self.mus = self.mus.cpu()
        self.inv = self.inv.cpu()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return mahalanobis_inv_layer_score(
            x, self.mus.to(x.device), self.inv.to(x.device)
        )


class MahalanobisMonoClassLite(LiteScore):
    def __init__(self):
        super().__init__("Mahalanobis (mono class)")
        self.mus = None
        self.inv = None

    def fit(self, X: torch.Tensor, *args, **kwargs):
        self.mus = X.mean(0, keepdim=True).cpu()
        self.inv = torch.linalg.pinv(torch.cov(X.T)).cpu()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return mahalanobis_inv_layer_score(
            x, self.mus.to(x.device), self.inv.to(x.device)
        )
