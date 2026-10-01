"""Time-based splits, the mechanical start-year rule, and the test-season lock."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


class TestSeasonLocked(RuntimeError):
    __test__ = False


def feature_coverage(pm: pd.DataFrame, P: SimpleNamespace) -> pd.DataFrame:
    """Share of non-missing values per season and feature (rows = seasons)."""
    cols = list(P.features.stats)
    return pm.groupby("season")[cols].agg(lambda s: s.notna().mean())


def valid_vote_share(join_report: dict) -> dict[int, float]:
    """Per season: share of box-score matches with votes present and totalling 30."""
    out = {}
    for s, d in join_report["per_season"].items():
        bad = d["excluded_by_reason"].get("no_votes_for_match", 0) + d["excluded_by_reason"].get(
            "vote_total_not_30", 0)
        out[int(s)] = 1 - bad / d["stats_matches"] if d["stats_matches"] else 0.0
    return out


def effective_start(
    join_report: dict, coverage: pd.DataFrame, P: SimpleNamespace
) -> tuple[int, list[str]]:
    """Smallest start season s <= latest_start_allowed meeting both coverage conditions to the end."""
    share = valid_vote_share(join_report)
    last = max(P.splits.val_seasons)
    notes = []
    for s in range(P.splits.train_start, P.splits.latest_start_allowed + 1):
        ok = True
        for t in range(s, last + 1):
            if share.get(t, 0.0) < P.coverage.min_valid_match_share:
                ok = False
                notes.append(f"{t}: valid-vote match share {share.get(t, 0.0):.3f} below floor")
            if t not in coverage.index or (coverage.loc[t] < P.coverage.min_feature_coverage).any():
                ok = False
                low = [] if t not in coverage.index else list(coverage.columns[coverage.loc[t] < P.coverage.min_feature_coverage])
                notes.append(f"{t}: feature coverage below floor {low}")
        if ok:
            return s, ([f"start moved to {s}: " + "; ".join(sorted(set(notes)))] if s > P.splits.train_start else [])
    raise RuntimeError(
        f"no start season <= {P.splits.latest_start_allowed} satisfies the coverage rule; "
        f"stop and report. {'; '.join(sorted(set(notes)))}"
    )


def make_splits(
    pm: pd.DataFrame, P: SimpleNamespace, start: int | None = None, allow_test: bool = False
) -> dict[str, pd.DataFrame]:
    start = P.splits.train_start if start is None else start
    test = P.splits.test_season
    if (pm["season"] == test).any() and not allow_test:
        raise TestSeasonLocked(f"rows from the locked test season {test} are present; refusing to split")
    train = pm[(pm["season"] >= start) & (pm["season"] <= P.splits.train_end)]
    val = pm[pm["season"].isin(P.splits.val_seasons)]
    out = {"train": train.copy(), "val": val.copy()}
    if allow_test:
        out["test"] = pm[pm["season"] == test].copy()
    assert not set(train["season"]) & set(val["season"])
    return out
