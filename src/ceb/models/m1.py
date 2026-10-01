"""M1: L2-penalised proportional-odds (cumulative logit) model with analytic gradients.

P(Y < j) = sigmoid(theta_j - eta), eta = X beta, theta_1 < ... < theta_{K-1} (unpenalised).
Objective = weighted mean NLL + (alpha / 2) * ||beta||^2. Ranking by eta is identical to
ranking by expected votes because eta is the only player-specific quantity.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def bands_to_codes(votes: np.ndarray, bands: list[list[int]]) -> np.ndarray:
    codes = np.full(len(votes), -1, dtype=int)
    for k, (lo, hi) in enumerate(bands):
        codes[(votes >= lo) & (votes <= hi)] = k
    if (codes < 0).any():
        raise ValueError("votes outside the protocol bands")
    return codes


class OrdinalLogit:
    def __init__(self, n_classes: int, alpha: float):
        self.K, self.alpha = n_classes, alpha

    # -- parameter handling -------------------------------------------------------------
    def _theta(self, a: np.ndarray) -> np.ndarray:
        return np.cumsum(np.r_[a[0], np.exp(a[1:])])

    def _cdf(self, eta: np.ndarray, theta: np.ndarray) -> np.ndarray:
        F = np.empty((len(eta), self.K + 1))
        F[:, 0], F[:, self.K] = 0.0, 1.0
        F[:, 1 : self.K] = expit(theta[None, :] - eta[:, None])
        return F

    def _nll_grad(self, params, X, y, w):
        p = X.shape[1]
        beta, a = params[:p], params[p:]
        theta = self._theta(a)
        eta = X @ beta
        F = self._cdf(eta, theta)
        f = F * (1 - F)
        n = np.arange(len(y))
        prob = np.clip(F[n, y + 1] - F[n, y], 1e-12, None)
        W = w.sum()
        nll = -(w * np.log(prob)).sum() / W
        g_eta = -(w / W) * (f[n, y] - f[n, y + 1]) / prob
        g_beta = X.T @ g_eta + self.alpha * beta
        # theta_j (j = 1..K-1) enters F_j only: +f_{y+1} when y+1 == j, -f_y when y == j
        g_theta = np.zeros(self.K - 1)
        up = y + 1 <= self.K - 1
        np.add.at(g_theta, y[up], -(w[up] / W) * f[n[up], y[up] + 1] / prob[up])
        dn = y >= 1
        np.add.at(g_theta, y[dn] - 1, (w[dn] / W) * f[n[dn], y[dn]] / prob[dn])
        g_a = np.empty_like(a)
        tail = np.cumsum(g_theta[::-1])[::-1]
        g_a[0] = tail[0]
        g_a[1:] = np.exp(a[1:]) * tail[1:]
        obj = nll + 0.5 * self.alpha * float(beta @ beta)
        return obj, np.r_[g_beta, g_a]

    # -- public API -----------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None, x0=None) -> OrdinalLogit:
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if x0 is None:
            cum = np.clip(np.cumsum(np.bincount(y, weights=w, minlength=self.K))[:-1] / w.sum(), 1e-6, 1 - 1e-6)
            th = np.log(cum / (1 - cum))
            th = np.maximum.accumulate(th + np.arange(len(th)) * 1e-3)
            a0 = np.r_[th[0], np.log(np.diff(th))]
            x0 = np.r_[np.zeros(X.shape[1]), a0]
        res = minimize(self._nll_grad, x0, args=(X, y, w), jac=True, method="L-BFGS-B",
                       options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
        p = X.shape[1]
        self.params_ = res.x
        self.coef_ = res.x[:p]
        self.theta_ = self._theta(res.x[p:])
        self.converged_ = bool(res.success)
        return self

    def eta(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def proba(self, X: np.ndarray) -> np.ndarray:
        F = self._cdf(self.eta(X), self.theta_)
        return np.diff(F, axis=1)

    def proba_any(self, X: np.ndarray) -> np.ndarray:
        return 1.0 - self.proba(X)[:, 0]
