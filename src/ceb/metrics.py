"""Per-match ranking metrics. Gains are the coaches' votes (linear).

Ties in model scores are handled tie-aware (a tie block shares the mean gain / the expected
share of the positions it spans) so that coarse scores such as disposals are not rewarded or
penalised by an arbitrary row order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METRICS = ["ndcg5", "top1", "recall5", "spearman"]


def _blocks(sorted_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(sorted_scores)
    change = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    return np.r_[0, change], np.r_[change, n]


def _discount(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(n) + 2.0)


def dcg_at_k(gains: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Tie-aware DCG@k (McSherry & Najork): tied items share the mean gain of their positions."""
    order = np.argsort(-scores, kind="stable")
    g, s = gains[order].astype(float), scores[order]
    disc = _discount(len(g))
    total = 0.0
    for a, b in zip(*_blocks(s), strict=True):
        if a >= k:
            break
        total += g[a:b].mean() * disc[a : min(b, k)].sum()
    return float(total)


def ndcg_at_k(votes: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    ideal = np.sort(votes.astype(float))[::-1]
    idcg = float((ideal[:k] * _discount(len(ideal))[:k]).sum())
    if idcg == 0:
        return float("nan")
    return dcg_at_k(votes, scores, k) / idcg


def top1_hit(votes: np.ndarray, scores: np.ndarray) -> float:
    """Share of the tied top-scored players who hold the match's maximum vote count."""
    top = scores == scores.max()
    return float(np.mean(votes[top] == votes.max()))


def recall_at_k(votes: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    """Expected number of vote-receivers in the top k, over min(k, number of receivers)."""
    rec = (votes > 0).astype(float)
    n_rec = rec.sum()
    if n_rec == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    r, s = rec[order], scores[order]
    expected = 0.0
    for a, b in zip(*_blocks(s), strict=True):
        if a >= k:
            break
        expected += r[a:b].sum() * (min(b, k) - a) / (b - a)
    return float(expected / min(k, n_rec))


def spearman_receivers(votes: np.ndarray, scores: np.ndarray, min_n: int = 3) -> float:
    m = votes > 0
    if m.sum() < min_n:
        return float("nan")
    v, s = votes[m].astype(float), scores[m].astype(float)
    if np.ptp(v) == 0 or np.ptp(s) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(v), rankdata(s))[0, 1])


def per_match_metrics(
    df: pd.DataFrame,
    score_col: str,
    vote_col: str = "votes",
    group_col: str = "match_id",
    k: int = 5,
    min_receivers: int = 3,
) -> pd.DataFrame:
    """One row per match: ndcg5, top1, recall5, spearman."""
    d = df[[group_col, vote_col, score_col]].sort_values(group_col, kind="stable")
    g = d[group_col].to_numpy()
    v = d[vote_col].to_numpy(dtype=float)
    s = d[score_col].to_numpy(dtype=float)
    bounds = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(g)]
    rows = []
    for a, b in zip(starts, ends, strict=True):
        rows.append(
            (
                g[a],
                ndcg_at_k(v[a:b], s[a:b], k),
                top1_hit(v[a:b], s[a:b]),
                recall_at_k(v[a:b], s[a:b], k),
                spearman_receivers(v[a:b], s[a:b], min_receivers),
            )
        )
    return pd.DataFrame(rows, columns=[group_col, *METRICS]).set_index(group_col)


def ece_components(
    df: pd.DataFrame, p_col: str, vote_col: str = "votes", group_col: str = "match_id", bins: int = 10
) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    """Per-match bin counts, sum of predicted p and sum of outcomes for P(votes > 0).

    Returned per match so the match-level bootstrap can resample them.
    """
    d = df[[group_col, vote_col, p_col]].copy()
    d["bin"] = np.minimum((d[p_col].clip(0, 1) * bins).astype(int), bins - 1)
    d["y"] = (d[vote_col] > 0).astype(float)
    ids = pd.Index(sorted(d[group_col].unique()), name=group_col)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    mi = pos.loc[d[group_col]].to_numpy()
    cnt = np.zeros((len(ids), bins))
    sp = np.zeros((len(ids), bins))
    sy = np.zeros((len(ids), bins))
    np.add.at(cnt, (mi, d["bin"].to_numpy()), 1.0)
    np.add.at(sp, (mi, d["bin"].to_numpy()), d[p_col].to_numpy())
    np.add.at(sy, (mi, d["bin"].to_numpy()), d["y"].to_numpy())
    return ids, cnt, sp, sy


def ece_from_components(cnt: np.ndarray, sp: np.ndarray, sy: np.ndarray) -> float:
    """Expected calibration error from (matches x bins) component arrays."""
    return float(np.abs(sy.sum(0) - sp.sum(0)).sum() / cnt.sum())
