"""Match-level bootstrap, paired comparisons, sign-flip permutation test, Holm correction."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


def rng_for(seed: int, label: str) -> np.random.Generator:
    """Deterministic generator per (seed, label), independent of call order."""
    return np.random.default_rng([seed, zlib.crc32(label.encode())])


def _ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.nanquantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "n": self.n}


def mean_ci(x: np.ndarray, B: int, seed: int, label: str, alpha: float = 0.05) -> Interval:
    """Bootstrap CI of the mean over matches; NaN matches (undefined metric) are dropped."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    idx = rng_for(seed, label).integers(0, len(x), size=(B, len(x)))
    lo, hi = _ci(x[idx].mean(1), alpha)
    return Interval(float(x.mean()), lo, hi, len(x))


@dataclass
class Comparison:
    diff: Interval
    relative: Interval
    p_raw: float

    def as_dict(self) -> dict:
        return {"diff": self.diff.as_dict(), "relative": self.relative.as_dict(), "p_raw": self.p_raw}


def sign_flip_p(d: np.ndarray, n_perm: int, rng: np.random.Generator, chunk: int = 2000) -> float:
    """Two-sided sign-flip permutation p-value for mean(d) = 0."""
    t = abs(d.mean())
    ge, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(m, len(d))) * 2 - 1
        ge += int((np.abs((signs * d).mean(1)) >= t - 1e-15).sum())
        done += m
    return (1 + ge) / (1 + n_perm)


def paired_comparison(
    a: np.ndarray, b: np.ndarray, B: int, n_perm: int, seed: int, label: str, alpha: float = 0.05
) -> Comparison:
    """A minus B on per-match metric vectors; pairs with an undefined value are dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    d = a - b
    rng = rng_for(seed, label)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    diff = d[idx].mean(1)
    mb = b[idx].mean(1)
    rel = a[idx].mean(1) / np.where(mb == 0, np.nan, mb) - 1
    dlo, dhi = _ci(diff, alpha)
    rlo, rhi = _ci(rel, alpha)
    rel_point = float(a.mean() / b.mean() - 1) if b.mean() != 0 else float("nan")
    return Comparison(
        Interval(float(d.mean()), dlo, dhi, len(d)),
        Interval(rel_point, rlo, rhi, len(d)),
        sign_flip_p(d, n_perm, rng_for(seed, label + "|perm")),
    )


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def ece_ci(
    cnt: np.ndarray, sp: np.ndarray, sy: np.ndarray, B: int, seed: int, label: str, alpha: float = 0.05
) -> Interval:
    n = cnt.shape[0]
    idx = rng_for(seed, label).integers(0, n, size=(B, n))
    boots = np.empty(B)
    for i in range(B):
        j = idx[i]
        boots[i] = np.abs(sy[j].sum(0) - sp[j].sum(0)).sum() / cnt[j].sum()
    point = float(np.abs(sy.sum(0) - sp.sum(0)).sum() / cnt.sum())
    lo, hi = _ci(boots, alpha)
    return Interval(point, lo, hi, n)


def boot_se_p(point: float, draws: np.ndarray) -> float:
    """Two-sided p for H0: effect = 0 from the bootstrap standard error (z = point / SE).

    Used instead of the percentile tail share because the latter cannot go below 1/(B+1), which
    after Holm over ~20 features would make even an overwhelming effect look non-significant.
    """
    from scipy.stats import norm

    d = np.asarray(draws, float)
    d = d[~np.isnan(d)]
    se = d.std(ddof=1)
    if se == 0:
        return 0.0 if point != 0 else 1.0
    return float(2 * norm.sf(abs(point) / se))
