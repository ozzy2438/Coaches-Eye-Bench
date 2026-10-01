"""Model selection (validation only), final fitting, scoring and the Q1 inference table."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import bootstrap as bs
from . import metrics as M
from .features import Prep
from .models import baselines
from .models.m1 import OrdinalLogit, bands_to_codes
from .models.m2 import fit_classifier, fit_ranker, fit_regressor, sort_by_match

MODELS = ["B0", "B1", "M1", "M2"]
SCORE_COL = {"B0": "b0", "B1": "b1", "M1": "m1", "M2": "m2"}
COMPARISONS = {"M1_vs_B0": ("M1", "B0"), "M1_vs_B1": ("M1", "B1"), "M2_vs_M1": ("M2", "M1")}


def mean_ndcg(df: pd.DataFrame, score: np.ndarray, k: int = 5) -> float:
    tmp = df[["match_id", "votes"]].assign(_s=score)
    return float(M.per_match_metrics(tmp, "_s", k=k)["ndcg5"].mean())


# --------------------------------------------------------------------------------------- selection
def select_m1(train: pd.DataFrame, val: pd.DataFrame, P: SimpleNamespace) -> tuple[dict, pd.DataFrame]:
    prep = Prep(P).fit(train)
    codes = bands_to_codes(train["votes"].to_numpy(), P.votes.bands)
    rows = []
    for view in P.m1.view_grid:
        Xtr, _ = prep.view(train, view)
        Xva, _ = prep.view(val, view)
        for alpha in P.m1.alpha_grid:
            m = OrdinalLogit(len(P.votes.bands), alpha).fit(Xtr, codes)
            rows.append({"view": view, "alpha": alpha, "val_ndcg5": mean_ndcg(val, m.eta(Xva)),
                         "converged": m.converged_})
    tab = pd.DataFrame(rows)
    best = tab.assign(_n=tab["val_ndcg5"].round(12)).sort_values(["_n", "alpha"], ascending=[False, False]).iloc[0]
    return {"view": best["view"], "alpha": float(best["alpha"])}, tab


def select_m2(train: pd.DataFrame, val: pd.DataFrame, P: SimpleNamespace) -> tuple[dict, pd.DataFrame]:
    prep = Prep(P).fit(train)
    tr = sort_by_match(train)
    Xtr, _ = prep.view(tr, "both")
    Xva, _ = prep.view(val, "both")
    rmax = max(P.m2.rounds_grid)
    rows = []
    for leaves in P.m2.num_leaves_grid:
        for min_leaf in P.m2.min_data_in_leaf_grid:
            rk = fit_ranker(Xtr, tr["votes"].to_numpy(), tr["match_id"].to_numpy(), P, leaves, min_leaf, rmax)
            for r in P.m2.rounds_grid:
                rows.append({"num_leaves": leaves, "min_data_in_leaf": min_leaf, "rounds": r,
                             "val_ndcg5": mean_ndcg(val, rk.predict(Xva, num_iteration=r))})
    tab = pd.DataFrame(rows)
    best = tab.assign(_n=tab["val_ndcg5"].round(12)).sort_values(
        ["_n", "rounds", "num_leaves"], ascending=[False, True, True]).iloc[0]
    leaves, min_leaf = int(best["num_leaves"]), int(best["min_data_in_leaf"])
    clf = fit_classifier(Xtr, tr["votes"].to_numpy(), P, leaves, min_leaf, rmax)
    yv = (val["votes"] > 0).astype(int)
    ll = {r: log_loss(yv, clf.predict_proba(Xva, num_iteration=r)[:, 1]) for r in P.m2.rounds_grid}
    clf_rounds = min(ll, key=lambda r: (round(ll[r], 12), r))
    return {"num_leaves": leaves, "min_data_in_leaf": min_leaf, "rounds": int(best["rounds"]),
            "clf_rounds": int(clf_rounds)}, tab


def select_all(train: pd.DataFrame, val: pd.DataFrame, P: SimpleNamespace) -> tuple[dict, dict]:
    c1, t1 = select_m1(train, val, P)
    c2, t2 = select_m2(train, val, P)
    return {"m1": c1, "m2": c2}, {"m1_grid": t1, "m2_grid": t2}


# --------------------------------------------------------------------------------------- fitting
@dataclass
class FittedModels:
    P: SimpleNamespace
    choices: dict
    prep: Prep
    m1: OrdinalLogit
    band_means: np.ndarray
    ranker: object
    clf: object
    reg: object

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["b0"] = baselines.b0_disposals(df)
        out["b1"] = baselines.b1_fantasy(df, self.P)
        X1, _ = self.prep.view(df, self.choices["m1"]["view"])
        out["m1"] = self.m1.eta(X1)
        out["p_m1"] = self.m1.proba_any(X1)
        out["ev_m1"] = self.m1.proba(X1) @ self.band_means
        X2, _ = self.prep.view(df, "both")
        out["m2"] = self.ranker.predict(X2)
        out["p_m2"] = self.clf.predict_proba(X2)[:, 1]
        out["ev_m2"] = np.clip(self.reg.predict(X2), 1e-6, None)  # post-hoc Q2 lens
        return out


def fit_models(df: pd.DataFrame, choices: dict, P: SimpleNamespace) -> FittedModels:
    prep = Prep(P).fit(df)
    codes = bands_to_codes(df["votes"].to_numpy(), P.votes.bands)
    X1, _ = prep.view(df, choices["m1"]["view"])
    m1 = OrdinalLogit(len(P.votes.bands), choices["m1"]["alpha"]).fit(X1, codes)
    bm = np.array([df.loc[codes == k, "votes"].mean() for k in range(len(P.votes.bands))])
    srt = sort_by_match(df)
    X2, _ = prep.view(srt, "both")
    c2 = choices["m2"]
    ranker = fit_ranker(X2, srt["votes"].to_numpy(), srt["match_id"].to_numpy(), P,
                        c2["num_leaves"], c2["min_data_in_leaf"], c2["rounds"])
    clf = fit_classifier(X2, srt["votes"].to_numpy(), P, c2["num_leaves"], c2["min_data_in_leaf"],
                         c2["clf_rounds"])
    reg = fit_regressor(X2, srt["votes"].to_numpy(), P, c2["num_leaves"], c2["min_data_in_leaf"],
                        c2["clf_rounds"])
    return FittedModels(P, choices, prep, m1, bm, ranker, clf, reg)


# --------------------------------------------------------------------------------------- inference
def evaluate_scores(scored: pd.DataFrame, P: SimpleNamespace, label: str) -> tuple[dict, pd.DataFrame]:
    """The Q1 table: metrics with bootstrap CIs, Holm-adjusted paired comparisons, decision rules."""
    B, perm, seed, alpha = P.inference.bootstrap_B, P.inference.permutation_n, P.seed, P.inference.alpha
    delta = P.inference.sesoi_ndcg5
    per = {m: M.per_match_metrics(scored, SCORE_COL[m], k=P.metrics.k,
                                  min_receivers=P.metrics.min_receivers_spearman) for m in MODELS}
    ids = per["B0"].index
    out: dict = {"label": label, "n_matches": int(len(ids)), "n_player_matches": int(len(scored)),
                 "metrics": {}, "calibration": {}, "comparisons": {}, "descriptive": {}}
    for m in MODELS:
        out["metrics"][m] = {
            k: bs.mean_ci(per[m][k].to_numpy(), B, seed, f"{label}|{m}|{k}", alpha).as_dict()
            for k in M.METRICS
        }
    out["spearman_undefined_matches"] = {m: int(per[m]["spearman"].isna().sum()) for m in MODELS}
    for m, col in (("M1", "p_m1"), ("M2", "p_m2")):
        _, cnt, sp, sy = M.ece_components(scored, col, bins=P.metrics.ece_bins)
        out["calibration"][m] = bs.ece_ci(cnt, sp, sy, B, seed, f"{label}|{m}|ece", alpha).as_dict()
    for k in M.METRICS:
        comps = {name: bs.paired_comparison(per[a][k].to_numpy(), per[b][k].to_numpy(), B, perm,
                                            seed, f"{label}|{name}|{k}", alpha)
                 for name, (a, b) in COMPARISONS.items()}
        adj = bs.holm([c.p_raw for c in comps.values()])
        for (name, c), p_h in zip(comps.items(), adj, strict=True):
            out["comparisons"].setdefault(name, {})[k] = {**c.as_dict(), "p_holm": p_h}
        d = bs.paired_comparison(per["B1"][k].to_numpy(), per["B0"][k].to_numpy(), B, 0, seed,
                                 f"{label}|B1_vs_B0|{k}", alpha)
        out["descriptive"].setdefault("B1_vs_B0", {})[k] = {"diff": d.diff.as_dict(),
                                                          "relative": d.relative.as_dict()}
    c = out["comparisons"]
    nd = "ndcg5"
    out["decisions"] = {
        "delta": delta,
        "m1_beats_b0": c["M1_vs_B0"][nd]["p_holm"] < alpha and c["M1_vs_B0"][nd]["diff"]["lo"] > 0,
        "m1_beats_b1": c["M1_vs_B1"][nd]["p_holm"] < alpha and c["M1_vs_B1"][nd]["diff"]["lo"] > 0,
        "m1_equivalent_to_b1": c["M1_vs_B1"][nd]["diff"]["lo"] > -delta
        and c["M1_vs_B1"][nd]["diff"]["hi"] < delta,
        "m2_justified": c["M2_vs_M1"][nd]["p_holm"] < alpha and c["M2_vs_M1"][nd]["diff"]["lo"] > delta,
        "m2_significantly_better": c["M2_vs_M1"][nd]["p_holm"] < alpha and c["M2_vs_M1"][nd]["diff"]["lo"] > 0,
    }
    long = pd.concat([per[m].assign(model=m) for m in MODELS]).reset_index()
    return out, long
