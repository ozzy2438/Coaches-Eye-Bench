"""Validation analysis (Q1-Q3 on Train/Validation) and the single guarded test run."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import blindspots, figures, guard, trends
from . import evaluate as E
from .config import protocol_sha256
from .io_utils import git_state, read_json, utc_now, write_json
from .splits import effective_start, feature_coverage, make_splits, valid_vote_share

SYNTH_STAMP = "SYNTHETIC DATA - NOT A RESULT"


class StopAndReport(RuntimeError):
    pass


def _stamp(P: SimpleNamespace, synthetic: bool, kind: str) -> dict:
    return {"kind": kind, "synthetic": synthetic, "protocol_sha256": protocol_sha256(),
            "git": git_state(), "created_utc": utc_now()}


def run_selection(pm: pd.DataFrame, join_report: dict, P: SimpleNamespace, outdir: Path,
                  synthetic: bool = False) -> dict:
    """`make train`: effective start, grid selection on validation, frozen choices."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cov = feature_coverage(pm, P)
    start, notes = effective_start(join_report, cov, P)
    sp = make_splits(pm, P, start)
    choices, tabs = E.select_all(sp["train"], sp["val"], P)
    tabs["m1_grid"].to_csv(outdir / "selection_m1.csv", index=False)
    tabs["m2_grid"].to_csv(outdir / "selection_m2.csv", index=False)
    split = {"effective_start": start, "notes": notes, "train_seasons": sorted(sp["train"]["season"].unique()),
             "val_seasons": sorted(sp["val"]["season"].unique()),
             "n_train_matches": int(sp["train"]["match_id"].nunique()),
             "n_val_matches": int(sp["val"]["match_id"].nunique()),
             "valid_vote_share": valid_vote_share(join_report), "feature_coverage_min": float(cov.min().min())}
    write_json(outdir / "split.json", split)
    frozen = {"choices": choices, "effective_start": start, **_stamp(P, synthetic, "frozen_choices")}
    write_json(outdir / "frozen_choices.json", frozen)
    return {"choices": choices, "split": split, "tables": tabs}


def run_validation(pm: pd.DataFrame, join_report: dict, P: SimpleNamespace, outdir: Path,
                   synthetic: bool = False, label: str = "val") -> dict:
    """`make eval-val`: Q1 table, Q2 blind spots, coaches vs umpires, Q3 trends (all pre-test)."""
    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    frozen_path = outdir / "frozen_choices.json"
    if not frozen_path.exists():
        run_selection(pm, join_report, P, outdir, synthetic)
    frozen = read_json(frozen_path)
    choices, start = frozen["choices"], frozen["effective_start"]
    sp = make_splits(pm, P, start)
    stamp = SYNTH_STAMP if synthetic else None

    fm = E.fit_models(sp["train"], choices, P)
    scored = fm.score(sp["val"])
    res, per_match = E.evaluate_scores(scored, P, label)
    res["stamp"] = _stamp(P, synthetic, "validation")
    res["note"] = ("Validation figures for M1/M2 are mildly optimistic: the specification was selected on "
                   "these seasons. The test run is the unbiased figure.")
    write_json(outdir / "metrics.json", res)
    per_match.to_csv(outdir / "per_match.csv", index=False)
    figures.ndcg_chart(res, outdir / "figures" / "ndcg5.png", "Agreement with coaches' votes (validation)", stamp)
    figures.calibration_chart(scored, outdir / "figures" / "calibration.png",
                              "Calibration of P(any votes) (validation)", stamp, P.metrics.ece_bins)

    q2 = blindspots.blind_spots(scored, P, label)
    for k in ("role", "closeness", "team"):
        q2[k].to_csv(outdir / f"q2_{k}.csv", index=False)
    q2m2 = blindspots.blind_spots(scored, P, label, ev_col="ev_m2")  # post-hoc lens
    for k in ("role", "closeness", "team"):
        q2m2[k].to_csv(outdir / f"q2_posthoc_m2_{k}.csv", index=False)
    ua = blindspots.umpire_agreement(scored, P, label)
    pm_dev = pm[pm["season"] <= max(P.splits.val_seasons)]
    cvu = trends.coaches_vs_umpires(pm_dev, choices["m1"], P, start)
    cvu["table"].to_csv(outdir / "q2_coaches_vs_umpires.csv", index=False)
    write_json(outdir / "q2_summary.json", {
        "h2a_defender": q2["h2a_defender"],
        "posthoc_m2_h2a_defender": q2m2["h2a_defender"],
        "note_posthoc": "POST-HOC: residuals relative to M2 (flexible) expected votes; not in protocol.md",
        "role_mix": q2["role_mix"], "n_matches": q2["n_matches"],
        "residual_sum_per_match_max_abs": q2["residual_sum_per_match_max_abs"],
        "umpire_agreement": ua, "coaches_vs_umpires_n_matches": cvu["n_matches"],
        "stamp": _stamp(P, synthetic, "q2")})
    figures.blindspot_chart(q2["role"], q2["closeness"], outdir / "figures" / "blind_spots.png",
                            "Where box scores miss what coaches reward (validation)", stamp)
    figures.coef_chart(cvu["table"], outdir / "figures" / "coaches_vs_umpires.png",
                       "Coaches vs umpires: M1 coefficients", stamp)

    q3 = trends.block_trends(pm_dev, choices["m1"], start, P)
    q3["table"].to_csv(outdir / "q3_trends.csv", index=False)
    q3["separation"].to_csv(outdir / "q3_separation.csv", index=False)
    figures.trend_chart(q3["table"], list(P.trends.highlight), outdir / "figures" / "trends.png",
                        "What coaches reward, by season block (M1 coefficients, 95% CI)", stamp)
    return {"metrics": res, "q2": q2, "q3": q3, "umpires": ua, "cvu": cvu}


def run_test(pm_all: pd.DataFrame, join_report_test: dict, P: SimpleNamespace, results_dir: Path,
             outdir: Path, synthetic: bool = False, enforce_guard: bool = True,
             derived_dir: Path | None = None, data_manifest: Path | None = None) -> dict:
    """`make eval-test`: ONE run with the frozen choices, refitted on Train + Validation."""
    results_dir, outdir = Path(results_dir), Path(outdir)
    if enforce_guard:
        guard.check_test_allowed(results_dir)
    frozen_path = results_dir / "val" / "frozen_choices.json" if enforce_guard else outdir.parent / "val" / "frozen_choices.json"
    frozen = read_json(frozen_path)
    share = valid_vote_share(join_report_test).get(P.splits.test_season, 0.0)
    if share < P.coverage.min_test_valid_match_share:
        raise StopAndReport(f"only {share:.3f} of {P.splits.test_season} matches have valid votes "
                            f"(floor {P.coverage.min_test_valid_match_share}); stop and report")
    sp = make_splits(pm_all, P, frozen["effective_start"], allow_test=True)
    fit_df = pd.concat([sp["train"], sp["val"]], ignore_index=True)
    fm = E.fit_models(fit_df, frozen["choices"], P)
    scored = fm.score(sp["test"])
    res, per_match = E.evaluate_scores(scored, P, "test")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    stamp = SYNTH_STAMP if synthetic else None
    res["stamp"] = _stamp(P, synthetic, "test")
    res["fit_seasons"] = sorted(fit_df["season"].unique())
    res["test_season"] = P.splits.test_season
    write_json(outdir / "metrics.json", res)
    per_match.to_csv(outdir / "per_match.csv", index=False)
    figures.ndcg_chart(res, outdir / "figures" / "ndcg5.png",
                       f"Agreement with coaches' votes ({P.splits.test_season}, held-out)", stamp)
    figures.calibration_chart(scored, outdir / "figures" / "calibration.png",
                              f"Calibration of P(any votes) ({P.splits.test_season})", stamp, P.metrics.ece_bins)
    if derived_dir is not None:
        Path(derived_dir).mkdir(parents=True, exist_ok=True)
        scored.to_parquet(Path(derived_dir) / "test_scores.parquet", index=False)
    receipt = {**_stamp(P, synthetic, "test_receipt"),
               "frozen_choices_sha256": __import__("hashlib").sha256(frozen_path.read_bytes()).hexdigest(),
               "n_matches": res["n_matches"], "test_season": P.splits.test_season,
               "data_manifest_sha256": (__import__("hashlib").sha256(Path(data_manifest).read_bytes()).hexdigest()
                                        if data_manifest and Path(data_manifest).exists() else None)}
    write_json(outdir / "test_run.json", receipt)
    return {"metrics": res, "scored": scored}


def reset_dir(p: Path) -> Path:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p
