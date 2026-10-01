"""B0 (disposals) and B1 (fantasy points): zero training, no features beyond the named columns."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


def b0_disposals(df: pd.DataFrame) -> pd.Series:
    return (df["kicks"] + df["handballs"]).astype(float)


def b1_fantasy(df: pd.DataFrame, P: SimpleNamespace) -> pd.Series:
    w = vars(P.fantasy)
    return sum(df[k].astype(float) * v for k, v in w.items())
