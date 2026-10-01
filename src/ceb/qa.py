"""Data-quality checks. Each returns (ok, detail); ``run_qa`` aggregates and can raise."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from .features import assert_no_leakage
from .ingest import STAT_INT_COLS

_MAX = {"goals": 20, "disposals": 90, "tackles": 40, "hit_outs": 90, "marks": 40, "kicks": 60,
        "handballs": 60, "clearances": 40, "inside_50s": 30, "rebounds": 30}


def run_qa(pm: pd.DataFrame, P: SimpleNamespace, stats: pd.DataFrame | None = None,
           include_test: bool = False) -> dict:
    total = P.votes.total_per_match
    checks: dict[str, tuple[bool, str]] = {}

    sums = pm.groupby("match_id")["votes"].sum()
    bad = sums[sums != total]
    checks["votes_per_match_total"] = (bad.empty, f"{len(bad)} matches != {total}")

    dup = pm.duplicated(["match_id", "player_key"]).sum()
    checks["no_duplicate_player_matches"] = (dup == 0, f"{dup} duplicates")

    neg = int((pm[STAT_INT_COLS] < 0).any(axis=1).sum())
    checks["stats_non_negative"] = (neg == 0, f"{neg} rows with negative counts")

    over = {c: int((pm[c] > lim).sum()) for c, lim in _MAX.items()}
    over = {k: v for k, v in over.items() if v}
    checks["stats_within_sane_maxima"] = (not over, f"rows above sanity maxima: {over}")

    tog = pm["time_on_ground"].dropna()
    checks["time_on_ground_0_100"] = (bool(((tog >= 0) & (tog <= 100)).all()), "TOG outside [0,100]")

    disp_ok = bool((pm["disposals"] == pm["kicks"] + pm["handballs"]).all())
    if "disposals_raw" in pm and pm["disposals_raw"].notna().any():
        mism = int((pm["disposals_raw"].fillna(pm["disposals"]) != pm["disposals"]).sum())
        checks["disposals_equal_kicks_plus_handballs"] = (disp_ok and mism == 0,
                                                          f"{mism} rows where source Disposals differ")
    else:
        checks["disposals_equal_kicks_plus_handballs"] = (disp_ok, "derived column inconsistent")

    per_team = pm.groupby(["match_id", "team"]).size()
    off = per_team[(per_team < 18) | (per_team > 26)]
    checks["players_per_team_match_18_26"] = (off.empty, f"{len(off)} team-matches outside 18-26")

    rec = pm[pm["votes"] > 0].groupby("match_id").size()
    bad_rec = rec[(rec < 5) | (rec > 10)]
    checks["vote_receivers_per_match_5_10"] = (bad_rec.empty, f"{len(bad_rec)} matches outside 5-10")

    g = pm.groupby(["match_id", "team"])["margin"]
    mm = g.first().groupby("match_id").agg(["sum", "count"])
    ctx_bad = int(((mm["count"] != 2) | (mm["sum"] != 0)).sum()) + int((g.nunique() != 1).sum())
    checks["context_margins_antisymmetric"] = (
        ctx_bad == 0, f"{ctx_bad} team-matches/matches whose margins are inconsistent or do not cancel")

    leak_ok = True
    try:
        assert_no_leakage(list(P.features.stats) + list(P.features.context))
    except AssertionError as e:
        leak_ok = False
        checks["feature_columns_leak_free"] = (False, str(e))
    if leak_ok:
        checks["feature_columns_leak_free"] = (True, "")

    if not include_test:
        n_test = int((pm["season"] == P.splits.test_season).sum())
        checks["no_test_season_rows_in_dev_data"] = (n_test == 0, f"{n_test} rows from {P.splits.test_season}")

    brown = pm.groupby("match_id")["brownlow_votes"].sum()
    share = float((brown == 6).mean())
    checks["brownlow_total_6_share_ge_95pct"] = (share >= 0.95, f"{share:.3f} of matches total 6")

    return {"ok": all(v[0] for v in checks.values()),
            "checks": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in checks.items()}}


class QAFailure(RuntimeError):
    pass


def assert_qa(report: dict) -> None:
    failed = {k: v["detail"] for k, v in report["checks"].items() if not v["ok"]}
    if failed:
        raise QAFailure(f"data-quality checks failed: {failed}")
