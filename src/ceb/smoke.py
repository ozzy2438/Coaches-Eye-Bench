"""`make smoke`: the whole pipeline on a tiny synthetic fixture. Offline, deterministic, < 2 min."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import analysis
from .config import smoke_params
from .io_utils import write_json
from .pipeline import join_and_check
from .qa import assert_qa
from .synth import generate


def run_smoke(outdir: Path, seed: int = 7, matches_per_season: int = 54, log=print) -> dict:
    t0 = time.time()
    P = smoke_params()
    outdir = analysis.reset_dir(Path(outdir))
    seasons = [P.splits.train_start, P.splits.train_end, *P.splits.val_seasons, P.splits.test_season]
    sy = generate(sorted(set(seasons)), matches_per_season, seed)
    res, qa, _ = join_and_check(sy.stats_raw, sy.votes_raw, P, include_test=True)
    assert_qa(qa)
    log(f"[smoke] joined: {res.report['overall']['match_rate_rows']:.3f} row match rate, "
        f"{res.player_match['match_id'].nunique()} matches")
    pm = res.player_match
    test_season = P.splits.test_season
    dev, test = pm[pm["season"] != test_season], pm[pm["season"] == test_season]
    dev_report = join_and_check(sy.stats_raw[sy.stats_raw["Season"] != test_season],
                                sy.votes_raw[sy.votes_raw["Season"] != test_season], P)[0].report
    test_report = join_and_check(sy.stats_raw[sy.stats_raw["Season"] == test_season],
                                 sy.votes_raw[sy.votes_raw["Season"] == test_season], P,
                                 include_test=True, enforce_rate=False)[0].report
    analysis.run_selection(dev, dev_report, P, outdir / "val", synthetic=True)
    val = analysis.run_validation(dev, dev_report, P, outdir / "val", synthetic=True, label="smoke-val")
    log(f"[smoke] validation done in {time.time() - t0:.0f}s")
    tst = analysis.run_test(pd.concat([dev, test], ignore_index=True), test_report, P, outdir,
                            outdir / "test", synthetic=True, enforce_guard=False,
                            derived_dir=outdir / "derived")
    write_json(outdir / "SYNTHETIC.json", {"synthetic": True, "note": "fixture output; never a result"})
    log(f"[smoke] test run done; total {time.time() - t0:.0f}s")
    return {"val": val, "test": tst, "seconds": time.time() - t0}
