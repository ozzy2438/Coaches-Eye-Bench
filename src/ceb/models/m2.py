"""M2: LightGBM LambdaRank grouped by match, plus a binary calibration companion."""

from __future__ import annotations

from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

N_JOBS = 1  # single thread: bit-for-bit reproducible across machines


def group_sizes(match_ids: np.ndarray) -> np.ndarray:
    """Rows must already be contiguous by match."""
    _, idx, counts = np.unique(match_ids, return_index=True, return_counts=True)
    return counts[np.argsort(idx)]


def sort_by_match(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("match_id", kind="stable").reset_index(drop=True)


def _common(P: SimpleNamespace, leaves: int, min_leaf: int, n_estimators: int) -> dict:
    return dict(
        n_estimators=n_estimators, learning_rate=P.m2.learning_rate, num_leaves=leaves,
        min_child_samples=min_leaf, colsample_bytree=P.m2.feature_fraction, subsample=1.0,
        random_state=P.seed, n_jobs=N_JOBS, deterministic=True, force_row_wise=True, verbosity=-1,
    )


def fit_ranker(X, votes, match_ids, P, leaves, min_leaf, n_estimators) -> lgb.LGBMRanker:
    m = lgb.LGBMRanker(
        objective="lambdarank", label_gain=list(range(11)),
        **_common(P, leaves, min_leaf, n_estimators),
    )
    m.fit(X, votes.astype(int), group=group_sizes(match_ids))
    return m


def fit_classifier(X, votes, P, leaves, min_leaf, n_estimators) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(objective="binary", **_common(P, leaves, min_leaf, n_estimators))
    m.fit(X, (votes > 0).astype(int))
    return m


def fit_regressor(X, votes, P, leaves, min_leaf, n_estimators) -> lgb.LGBMRegressor:
    """Expected votes (post-hoc Q2 lens only; not part of the Q1 comparison)."""
    m = lgb.LGBMRegressor(objective="regression", **_common(P, leaves, min_leaf, n_estimators))
    m.fit(X, votes.astype(float))
    return m
