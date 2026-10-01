"""Generate RESULTS.md and the README results block from results/*.json. Never hand-edited."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import load_params
from .evaluate import COMPARISONS, MODELS
from .io_utils import read_json

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
METRIC_NAMES = {"ndcg5": "NDCG@5", "top1": "Top-1 hit", "recall5": "Normalized Recall@5", "spearman": "Spearman"}
MODEL_NAMES = {"B0": "B0 Disposals", "B1": "B1 Fantasy points", "M1": "M1 Interpretable",
               "M2": "M2 LightGBM LambdaRank"}


def _f(d: dict, nd: int = 3) -> str:
    return f"{d['point']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _analysis_table(table: pd.DataFrame, nd: int = 3) -> str:
    table = table.copy()
    for col in table.columns:
        if col.startswith("p_"):
            table[col] = table[col].map(_p)
    return table.round(nd).to_markdown(index=False)


def metrics_table(res: dict) -> str:
    rows = ["| Model | " + " | ".join(METRIC_NAMES.values()) + " | ECE P(any votes) |",
            "|---|" + "---|" * (len(METRIC_NAMES) + 1)]
    for m in MODELS:
        ece = _f(res["calibration"][m]) if m in res["calibration"] else "n/a"
        rows.append(f"| {MODEL_NAMES[m]} | " + " | ".join(_f(res["metrics"][m][k]) for k in METRIC_NAMES)
                    + f" | {ece} |")
    return "\n".join(rows)


def comparison_table(res: dict, metric: str = "ndcg5") -> str:
    rows = [f"| Comparison ({METRIC_NAMES[metric]}) | Difference [95% CI] | Relative gain [95% CI] | Holm-adj. p |",
            "|---|---|---|---|"]
    for name in COMPARISONS:
        c = res["comparisons"][name][metric]
        rows.append(f"| {name.replace('_vs_', ' vs ')} | {_f(c['diff'])} | "
                    f"{c['relative']['point']:+.1%} [{c['relative']['lo']:+.1%}, {c['relative']['hi']:+.1%}] "
                    f"| {_p(c['p_holm'])} |")
    return "\n".join(rows)


def findings(test: dict, q2: dict | None, split: str) -> list[str]:
    c, m, dec = test["comparisons"], test["metrics"], test["decisions"]
    out = []
    r = c["M1_vs_B0"]["ndcg5"]
    if dec["m1_beats_b0"]:
        out.append(
            f"**A transparent model agrees with coaches' votes {r['relative']['point']:.0%} more than disposals "
            f"alone** (NDCG@5 {m['M1']['ndcg5']['point']:.3f} vs {m['B0']['ndcg5']['point']:.3f}; {split}; "
            f"relative gain 95% CI {r['relative']['lo']:+.0%} to {r['relative']['hi']:+.0%}; Holm-adjusted p {_p(r['p_holm'])}).")
    else:
        out.append(
            f"**The transparent model is not reliably better than disposals alone** (NDCG@5 "
            f"{m['M1']['ndcg5']['point']:.3f} vs {m['B0']['ndcg5']['point']:.3f}; {split}; difference 95% CI "
            f"[{r['diff']['lo']:+.3f}, {r['diff']['hi']:+.3f}]).")
    r1 = c["M1_vs_B1"]["ndcg5"]
    if dec["m1_equivalent_to_b1"]:
        out[-1] += (f" It is practically equivalent to the fantasy-points formula (difference 95% CI "
                    f"[{r1['diff']['lo']:+.3f}, {r1['diff']['hi']:+.3f}] inside ±{dec['delta']}).")
    elif dec["m1_beats_b1"]:
        out[-1] += (f" It also beats the fantasy-points formula by {r1['diff']['point']:+.3f} NDCG@5 "
                    f"(95% CI [{r1['diff']['lo']:+.3f}, {r1['diff']['hi']:+.3f}]).")
    r2 = c["M2_vs_M1"]["ndcg5"]
    if dec["m2_justified"]:
        out.append(
            f"**The heavier model adds {r2['diff']['point']:+.3f} NDCG@5 over the interpretable one** "
            f"(95% CI [{r2['diff']['lo']:+.3f}, {r2['diff']['hi']:+.3f}], above the {dec['delta']} threshold) - "
            "meeting the pre-specified statistical decision rule; practical value still needs coach review.")
    elif dec["m2_significantly_better"]:
        out.append(
            f"**The heavier model is statistically better, but the gain is not established above the threshold:** "
            f"{r2['diff']['point']:+.3f} NDCG@5 (95% CI [{r2['diff']['lo']:+.3f}, {r2['diff']['hi']:+.3f}]) "
            f"vs the pre-specified {dec['delta']} threshold; the lower confidence bound does not clear it.")
    else:
        out.append(
            f"**The estimated difference for the heavier model is {r2['diff']['point']:+.3f} NDCG@5** "
            f"(95% CI [{r2['diff']['lo']:+.3f}, {r2['diff']['hi']:+.3f}]); evidence is insufficient to meet the decision rule.")
    if q2 is None:
        out.append("**Blind spots:** pending Q2 outputs.")
    else:
        h = q2.get("h2a_defender")
        role = pd.DataFrame(q2["_role"]) if "_role" in q2 else None
        txt = ""
        if role is not None and len(role):
            pos = role[(role["lo"] > 0)]
            neg = role[(role["hi"] < 0)]
            top = pos.sort_values("mean_resid", ascending=False)
            txt = (f"the largest positive M1 residual is for **{top.iloc[0]['role']}** (mean +{top.iloc[0]['mean_resid']:.2f} votes "
                   f"per player-match, 95% CI [{top.iloc[0]['lo']:+.2f}, {top.iloc[0]['hi']:+.2f}]); "
                   if len(top) else "no role has a clearly positive M1 residual; ")
            if len(neg):
                n0 = neg.sort_values("mean_resid").iloc[0]
                txt += f"largest negative residual: {n0['role']} ({n0['mean_resid']:+.2f}, CI [{n0['lo']:+.2f}, {n0['hi']:+.2f}]). "
        if h is not None:
            txt += (f"Pre-registered H2a (defenders under-rated): {'supported' if h['supported'] else 'not supported'} "
                    f"(mean residual {h['mean_resid']:+.3f}, 95% CI [{h['lo']:+.3f}, {h['hi']:+.3f}]). ")
        out.append("**Role residuals (validation seasons):** " + txt +
                   "Role comparisons beyond H2a are exploratory. This proxy cannot isolate unrecorded work, "
                   "and non-support does not establish that a blind spot is absent (see Limitations).")
    return out


def build_results(results: Path, synthetic_ok: bool = False) -> tuple[str, str]:
    """Returns (RESULTS.md text, README block text)."""
    results = Path(results)
    val_p, test_p = results / "val" / "metrics.json", results / "test" / "metrics.json"
    q2_p = results / "val" / "q2_summary.json"
    lines = ["# Results (generated by `make report`; do not edit by hand)\n"]
    block = []
    val = read_json(val_p) if val_p.exists() else None
    test = read_json(test_p) if test_p.exists() else None
    synthetic = any(d and d.get("stamp", {}).get("synthetic") for d in (val, test))
    if synthetic:
        lines.append("> **SYNTHETIC FIXTURE OUTPUT - NOT A RESULT.** Produced by `make smoke` to test the machinery.\n")
        block.append("**SYNTHETIC FIXTURE OUTPUT - NOT A RESULT.** No findings about AFL players.")
    q2 = read_json(q2_p) if q2_p.exists() else None
    if q2 is not None and (results / "val" / "q2_role.csv").exists():
        q2["_role"] = pd.read_csv(results / "val" / "q2_role.csv").to_dict("records")
    if test is None and not synthetic:
        status = ("**Status: real-data run pending.** No held-out result has been produced yet: the pipeline is "
                  "implemented and verified on synthetic data, but `make data` has not been run against the live sources. "
                  "No number in this repository is a finding about the AFL until `results/test/metrics.json` exists "
                  "(generated by `make eval-test`).")
        block.append(status)
        lines.append(status + "\n")
        if val is not None:
            status = ("**Status: held-out evaluation pending.** Validation results are available below; "
                      "model selection used these seasons, so they are not held-out evidence. "
                      "The 2026 test has not been evaluated.")
            block[-1] = status
            lines[-1] = status + "\n"
    if val is not None:
        lines += ["## Validation seasons (also used for M1/M2 specification selection)\n",
                  f"Evaluated on {val['n_matches']} matches and {val['n_player_matches']} player-matches. "
                  "Intervals are 95% match-bootstrap intervals conditional on the selected specification. "
                  "The three pre-specified paired model comparisons use sign-flip tests with Holm correction "
                  "within each metric; NDCG@5 is primary. Normalized Recall@5 divides hits by "
                  "`min(5, vote receivers)` and is not conventional recall.\n",
                  metrics_table(val), "", comparison_table(val), ""]
        lines.append("### Secondary metrics: the same pre-specified contrasts\n")
        for metric in ("top1", "recall5", "spearman"):
            lines += [comparison_table(val, metric), ""]
        descriptive = val.get("descriptive", {}).get("B1_vs_B0", {}).get("ndcg5")
        if descriptive is not None:
            lines += [f"B1 minus B0 NDCG@5: {_f(descriptive['diff'])}. This baseline comparison "
                      "is descriptive; it is outside the three pre-specified inferential contrasts.\n"]
        for f in findings(val, q2, "validation seasons; used for model selection"):
            lines.append(f"- {f}")
        lines.append("")
        block += ["", f"**Validation (2024–2025): {val['n_matches']} matches; used for model selection.**",
                  "", metrics_table(val), "", "These conditional intervals omit model-selection "
                  "uncertainty. Full paired tests and Q2/Q3 limitations are in `results/RESULTS.md`."]
        split_p = results / "val" / "split.json"
        if split_p.exists():
            split_d = read_json(split_p)
            lines += [f"Training seasons: {split_d['train_seasons']}; validation seasons: "
                      f"{split_d['val_seasons']}. Effective start: {split_d['effective_start']}. "
                      f"Training matches: {split_d['n_train_matches']}; validation matches: "
                      f"{split_d['n_val_matches']}.\n"]
        frozen_p = results / "val" / "frozen_choices.json"
        if frozen_p.exists():
            lines += [f"Study choices selected on validation: `{read_json(frozen_p)['choices']}`. "
                      "Pilot choices are stored separately and are not imported.\n"]
    if test is not None:
        split = f"{test.get('test_season', 'test')} held-out season"
        params = load_params()
        lines += [f"## Test: {split} (single run, frozen protocol)\n",
                  f"Evaluated on {test['n_matches']} matches and {test['n_player_matches']} player-matches. "
                  f"Frozen specifications were refitted on seasons {test.get('fit_seasons', [])}; "
                  "no test-based model or parameter selection followed. "
                  f"95% match-bootstrap intervals use {params.inference.bootstrap_B} resamples and "
                  f"paired sign-flip tests use {params.inference.permutation_n} permutations, with "
                  "Holm correction across the three pre-specified contrasts within each metric. "
                  "Intervals condition on the fitted models, treat matches as resampling units "
                  "and do not measure uncertainty across future seasons or training runs.\n",
                  metrics_table(test), "",
                  comparison_table(test), ""]
        lines.append("### Secondary metrics: the same pre-specified contrasts\n")
        for metric in ("top1", "recall5", "spearman"):
            lines += [comparison_table(test, metric), ""]
        descriptive = test.get("descriptive", {}).get("B1_vs_B0", {}).get("ndcg5")
        if descriptive is not None:
            lines += [f"B1 minus B0 NDCG@5: {_f(descriptive['diff'])}. This baseline comparison "
                      "is descriptive; it is outside the three pre-specified inferential contrasts.\n"]
        block += ["", f"**Held-out test ({test.get('test_season', 'test')}): "
                  f"{test['n_matches']} matches; one evaluation with frozen choices.**",
                  "", metrics_table(test), "", comparison_table(test), ""]
        test_findings = findings(test, None, split)[:2]
        secondary = []
        for metric in ("top1", "recall5", "spearman"):
            c = test["comparisons"]["M2_vs_M1"][metric]
            supported = c["p_holm"] < params.inference.alpha and c["diff"]["lo"] > 0
            secondary.append(f"{METRIC_NAMES[metric]} {_f(c['diff'])}, Holm p {_p(c['p_holm'])} "
                             f"({'positive difference supported' if supported else 'positive difference not established'})")
        test_findings.append("**M2 minus M1, secondary findings:** " + "; ".join(secondary) + ".")
        for f in test_findings:
            lines.append(f"- {f}")
            block.append(f"- {f}")
        lines += ["", "A failed decision rule does not establish that the gain is below delta, "
                  "nor that the models are equivalent. ECE intervals are descriptive; no paired "
                  "calibration test was pre-specified. This is one held-out season and measures "
                  "agreement with coaches, not causal value or demonstrated usefulness to a club. "
                  "Q2/Q3 below use development data only.\n"]
        qa_p = results / "test" / "data_quality.json"
        if qa_p.exists():
            audit = read_json(qa_p)
            rates = audit["join"]["per_season"][str(test["test_season"])]
            lines += ["### Test data quality and exclusions\n",
                      f"Included {rates['included_matches']}/{rates['stats_matches']} source matches "
                      f"({rates['included_matches'] / rates['stats_matches']:.3%}); vote-row match rate "
                      f"{rates['match_rate_rows']:.3%}, vote-mass match rate {rates['match_rate_mass']:.3%}. "
                      "All frozen QA gates passed. " + audit["exclusion_note"] + "\n"]
    if q2 is not None:
        lines.append("## Q2 blind spots (validation seasons)\n")
        lines.append("Residuals are actual minus M1 expected votes after expected votes are rescaled "
                     "to sum to 30 in each match. Positive residuals describe disagreement with M1; "
                     "they do not establish missing defensive work or causal player value. Roles are "
                     "box-score proxies. Only H2a was pre-specified; other role contrasts are exploratory.\n")
        rc = results / "val" / "q2_role.csv"
        if rc.exists():
            lines += [_analysis_table(pd.read_csv(rc)), ""]
        ph = results / "val" / "q2_posthoc_m2_role.csv"
        if ph.exists():
            lines += ["### Post-hoc: residuals relative to M2 (not in protocol)\n",
                      _analysis_table(pd.read_csv(ph)), ""]
        team_p = results / "val" / "q2_team.csv"
        if team_p.exists():
            lines += ["### Team residuals (match-bootstrap intervals; Holm-adjusted within teams)\n",
                      _analysis_table(pd.read_csv(team_p), 4), ""]
        close_p = results / "val" / "q2_closeness.csv"
        if close_p.exists():
            lines += ["### Match closeness: signed residuals are structurally zero\n",
                      "All players in a match share its closeness group, and residuals sum to zero "
                      "per match. These signed group means cannot measure differences in accuracy "
                      "across match contexts; numerical p-values must not be interpreted.\n",
                      _analysis_table(pd.read_csv(close_p), 4), ""]
        if q2.get("umpire_agreement"):
            ua = q2["umpire_agreement"]
            lines += ["### Agreement with coaches: Brownlow ranking versus M1\n",
                      f"Same {ua['n_matches']} validation matches: Brownlow NDCG@5 "
                      f"{_f(ua['umpires_ndcg5'])}; M1 {_f(ua['m1_ndcg5'])}; paired M1 minus "
                      f"Brownlow difference {_f(ua['m1_minus_umpires'])}. This is descriptive "
                      "agreement with coaches, not a test of whose judgement is correct.\n"]
        cvu = results / "val" / "q2_coaches_vs_umpires.csv"
        if cvu.exists():
            t = pd.read_csv(cvu)
            lines += ["### Coaches vs umpires: features where the 95% CIs do not overlap\n",
                      "Shared scaling and the selected M1 specification are used on all included "
                      "development seasons. Non-overlap is a descriptive criterion, not a "
                      "Holm-corrected test. Logit coefficients for different ordinal targets "
                      "are relative to each target's residual scale.\n",
                      _analysis_table(t[t["differs"]]) if t["differs"].any() else "None.", ""]
    q3 = results / "val" / "q3_separation.csv"
    if q3.exists():
        t = pd.read_csv(q3)
        alpha = load_params().inference.alpha
        first_last = t["p_holm_first_last"] < alpha
        named = t["highlight"]
        lines += ["## Q3 trends: features whose block CIs clearly separate\n",
                  f"Across {len(t)} features, {int(t['any_pair_ci_nonoverlap'].sum())} meet the "
                  f"any-pair interval criterion; {int(first_last.sum())} first-versus-last Holm "
                  f"tests have p < {alpha}. Among the {int(named.sum())} pre-named features, "
                  f"{int((first_last & named).sum())} first-versus-last Holm tests meet that threshold.\n",
                  "The selected M1 specification is fitted separately in each season block, with "
                  "one shared development-data scaler. The protocol flags any pair of "
                  "non-overlapping block intervals; the Holm p-values below test first versus last "
                  "blocks across features. These are different criteria. Coefficient shifts are "
                  "associations on a logit scale and may reflect changing residual noise, correlated "
                  "statistics, rules or model fit, rather than changed coaching preferences.\n",
                  _analysis_table(t[t["any_pair_ci_nonoverlap"]])
                  if t["any_pair_ci_nonoverlap"].any() else "No feature clearly separates.", ""]
        highlighted = t[t["highlight"]].copy()
        if len(highlighted):
            highlighted["p_holm_first_last"] = highlighted["p_holm_first_last"].map(_p)
            cols = ["feature", "first_block", "last_block", "first_coef", "last_coef",
                    "diff", "diff_lo", "diff_hi", "p_holm_first_last"]
            lines += ["### Pre-named features of interest: first and last block estimates\n",
                      "Estimates are shown for all five pre-named features. A change is claimed "
                      "only under the non-overlap criterion above. Holm correction covers all "
                      "model features, not only these rows.\n",
                      highlighted[cols].round(3).to_markdown(index=False), ""]
    if val is not None:
        test_status = ("The 2026 test remains unopened until authorized. " if test is None
                       else "Held-out results, when present, are reported separately above. ")
        lines += ["## Limits of this validation evidence\n",
                  "M1/M2 were selected on these same validation seasons; their reported performance "
                  "and conditional confidence intervals omit model-selection uncertainty. They "
                  "are not held-out estimates. " + test_status +
                  "Coaches' votes are subjective, public box scores omit tracking and off-ball work, "
                  "and the role proxy can let the model absorb an omitted-work premium. Failure to "
                  "support H2a therefore does not show that no blind spot exists. The delta = 0.01 "
                  "decision threshold is a research convention, not demonstrated usefulness to a coach. "
                  "2020 had shortened quarters; statistical scaling cannot remove all season "
                  "differences. No causal or adoption claim follows from these results.\n"]
    return "\n".join(lines), "\n".join(block)


def write_report(results: Path, readme: Path | None) -> Path:
    results = Path(results)
    md, block = build_results(results)
    out = results / "RESULTS.md"
    out.write_text(md + "\n")
    if readme is not None and Path(readme).exists():
        t = Path(readme).read_text()
        if START not in t or END not in t:
            raise ValueError(f"{readme} lacks the {START} / {END} markers")
        t = re.sub(re.escape(START) + r".*?" + re.escape(END), START + "\n" + block + "\n" + END, t, flags=re.S)
        Path(readme).write_text(t)
    return out
