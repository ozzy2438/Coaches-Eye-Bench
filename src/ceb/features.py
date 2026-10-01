"""Feature views and the leakage guard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

# a column with any of these underscore-separated tokens can never be a model input
FORBIDDEN_TOKENS = frozenset({
    "vote", "votes", "brownlow", "coach", "coaches", "player", "name", "team", "age", "career",
    "round", "season", "venue", "match", "date", "id", "tier", "role", "club",
})


def assert_no_leakage(columns: list[str]) -> None:
    """Feature columns must never carry votes, identity, or reputation proxies."""
    bad = [c for c in columns if FORBIDDEN_TOKENS & set(c.removesuffix("_mz").split("_"))]
    if bad:
        raise AssertionError(f"forbidden feature columns: {bad}")


class Prep:
    """Fits medians/means/SDs on the fitting data only, then builds views."""

    def __init__(self, P: SimpleNamespace):
        self.stats = list(P.features.stats)
        self.context = list(P.features.context)
        assert_no_leakage(self.stats + self.context)

    def fit(self, df: pd.DataFrame) -> Prep:
        x = df[self.stats]
        self.median_ = x.median()
        filled = x.fillna(self.median_)
        self.mean_ = filled.mean()
        self.std_ = filled.std(ddof=0).replace(0, 1.0)
        return self

    def _filled(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.stats].astype(float).fillna(self.median_)

    def _match_z(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._filled(df)
        g = x.groupby(df["match_id"].to_numpy())
        mu = g.transform("mean")
        sd = g.transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
        return ((x - mu) / sd).fillna(0.0)

    def view(self, df: pd.DataFrame, view: str) -> tuple[np.ndarray, list[str]]:
        ctx = df[self.context].astype(float)
        if view == "raw":
            parts, names = [self._filled(df), ctx], self.stats + self.context
        elif view == "global":
            parts = [(self._filled(df) - self.mean_) / self.std_, ctx]
            names = self.stats + self.context
        elif view == "match_z":
            parts, names = [self._match_z(df), ctx], self.stats + self.context
        elif view == "both":
            z = self._match_z(df)
            parts = [self._filled(df), z.add_suffix("_mz"), ctx]
            names = self.stats + [f"{c}_mz" for c in self.stats] + self.context
        else:
            raise ValueError(f"unknown view {view!r}")
        assert_no_leakage(names)
        return np.hstack([p.to_numpy(float) for p in parts]), names
