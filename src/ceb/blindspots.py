"""Q2: where do box scores systematically miss what coaches reward? (validation seasons only)

residual = actual votes - expected votes, with expected votes (M1) rescaled inside every match so
they sum to the fixed 30-vote budget. Positive = coaches gave more of the budget than the box score
predicts. Groups are summarised with a match-level (cluster) bootstrap.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import bootstrap as bs
from .roles import ROLES, assign_roles


def add_residuals(scored: pd.DataFrame, P: SimpleNamespace, ev_col: str = "ev_m1") -> pd.DataFrame:
    d = scored.copy()
    tot = d.groupby("match_id")[ev_col].transform("sum")
    d["ev_scaled"] = d[ev_col] * P.votes.total_per_match / tot
    d["resid"] = d["votes"] - d["ev_scaled"]
    d["role"] = assign_roles(d, P)
    gap = d["margin"].abs()
    d["closeness"] = np.select(
        [gap <= P.q2.close_margin_max, gap >= P.q2.blowout_margin_min], ["close", "blowout"], "medium")
    return d


def group_summary(d: pd.DataFrame, by: str, P: SimpleNamespace, label: str,
                  levels: list[str] | None = None) -> pd.DataFrame:
    """Mean residual per group with match-cluster bootstrap CI and two-sided bootstrap p."""
    levels = levels or sorted(d[by].unique())
    mids = pd.Index(sorted(d["match_id"].unique()))
    pos = pd.Series(np.arange(len(mids)), index=mids).loc[d["match_id"]].to_numpy()
    S = np.zeros((len(mids), len(levels)))
    N = np.zeros_like(S)
    for j, lv in enumerate(levels):
        m = (d[by] == lv).to_numpy()
        np.add.at(S[:, j], pos[m], d["resid"].to_numpy()[m])
        np.add.at(N[:, j], pos[m], 1.0)
    B = P.q2.bootstrap_B
    idx = bs.rng_for(P.seed, f"q2|{label}|{by}").integers(0, len(mids), size=(B, len(mids)))
    num, den = S[idx].sum(1), N[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num / den
    rows = []
    for j, lv in enumerate(levels):
        b = boot[:, j][~np.isnan(boot[:, j])]
        pt = S[:, j].sum() / N[:, j].sum() if N[:, j].sum() else np.nan
        if len(b):
            lo, hi = np.quantile(b, [0.025, 0.975])
            p = bs.boot_se_p(pt, b)
        else:
            lo = hi = p = np.nan
        rows.append({by: lv, "n_player_matches": int(N[:, j].sum()), "mean_resid": pt, "lo": lo,
                     "hi": hi, "p_boot": p})
    out = pd.DataFrame(rows)
    ok = out["p_boot"].notna()
    out["p_holm"] = np.nan
    out.loc[ok, "p_holm"] = bs.holm(out.loc[ok, "p_boot"].tolist())
    return out


def blind_spots(scored_val: pd.DataFrame, P: SimpleNamespace, label: str = "val",
                ev_col: str = "ev_m1") -> dict:
    """``ev_col='ev_m1'`` is the protocol analysis; ``'ev_m2'`` is the labelled post-hoc lens."""
    label = label if ev_col == "ev_m1" else f"{label}|{ev_col}"
    d = add_residuals(scored_val, P, ev_col)
    role = group_summary(d, "role", P, label, [r for r in ROLES if (d["role"] == r).any()])
    close = group_summary(d, "closeness", P, label, ["close", "medium", "blowout"])
    team = group_summary(d, "team", P, label)
    de = role[role["role"] == "Defender"]
    h2a = None
    if len(de):
        r = de.iloc[0]
        h2a = {"mean_resid": float(r["mean_resid"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
               "supported": bool(r["lo"] > 0)}
    return {"role": role, "closeness": close, "team": team, "h2a_defender": h2a,
            "residual_sum_per_match_max_abs": float(d.groupby("match_id")["resid"].sum().abs().max()),
            "n_matches": int(d["match_id"].nunique()), "role_mix": d["role"].value_counts(normalize=True).to_dict(),
            "frame": d}


def umpire_agreement(scored: pd.DataFrame, P: SimpleNamespace, label: str = "val") -> dict:
    """NDCG@5 against coaches' votes of (a) ranking by Brownlow votes and (b) M1, same matches."""
    from . import metrics as M

    s = scored[scored.groupby("match_id")["brownlow_votes"].transform("sum") == 6]
    u = M.per_match_metrics(s, "brownlow_votes", k=P.metrics.k)["ndcg5"].to_numpy()
    m1 = M.per_match_metrics(s, "m1", k=P.metrics.k)["ndcg5"].to_numpy()
    B = P.inference.bootstrap_B
    cmp = bs.paired_comparison(m1, u, B, 0, P.seed, f"{label}|M1_vs_umpires")
    return {
        "n_matches": int(len(u)),
        "umpires_ndcg5": bs.mean_ci(u, B, P.seed, f"{label}|umpires").as_dict(),
        "m1_ndcg5": bs.mean_ci(m1, B, P.seed, f"{label}|m1_same_matches").as_dict(),
        "m1_minus_umpires": cmp.diff.as_dict(),
    }
