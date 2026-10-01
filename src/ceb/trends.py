"""Q3 (trends over seasons) and the coaches-vs-umpires coefficient comparison share this engine."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import bootstrap as bs
from .features import Prep
from .models.m1 import OrdinalLogit, bands_to_codes


def season_blocks(start: int, last: int, width: int) -> list[list[int]]:
    """Consecutive blocks of ``width`` seasons; a remainder of one season merges into the previous."""
    seasons = list(range(start, last + 1))
    blocks = [seasons[i : i + width] for i in range(0, len(seasons), width)]
    if len(blocks) > 1 and len(blocks[-1]) == 1 and width > 1:
        blocks[-2] += blocks.pop()
    return blocks


def bootstrap_coefs(
    X: np.ndarray, codes: np.ndarray, match_ids: np.ndarray, n_classes: int, alpha: float,
    B: int, seed: int, label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate and match-level bootstrap draws of the coefficients (warm-started)."""
    point = OrdinalLogit(n_classes, alpha).fit(X, codes)
    uniq, inv = np.unique(match_ids, return_inverse=True)
    rng = bs.rng_for(seed, label)
    draws = np.empty((B, X.shape[1]))
    for b in range(B):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float)
        draws[b] = OrdinalLogit(n_classes, alpha).fit(X, codes, sample_weight=w, x0=point.params_).coef_
    return point.coef_, draws


def _ci(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return lo, hi


def block_trends(pm: pd.DataFrame, spec: dict, start: int, P: SimpleNamespace) -> dict:
    """Selected M1 spec per season block; scaler frozen on all included data so blocks are comparable."""
    last = P.trends.last_season_included
    data = pm[(pm["season"] >= start) & (pm["season"] <= last)]
    prep = Prep(P).fit(data)
    X_all, names = prep.view(data, spec["view"])
    codes_all = bands_to_codes(data["votes"].to_numpy(), P.votes.bands)
    blocks = season_blocks(start, last, P.trends.block_width)
    rows, store = [], {}
    for blk in blocks:
        m = data["season"].isin(blk).to_numpy()
        lab = f"{blk[0]}-{blk[-1]}" if len(blk) > 1 else str(blk[0])
        coef, draws = bootstrap_coefs(X_all[m], codes_all[m], data.loc[m, "match_id"].to_numpy(),
                                      len(P.votes.bands), spec["alpha"], P.trends.bootstrap_B, P.seed,
                                      f"q3|{lab}")
        lo, hi = _ci(draws)
        store[lab] = (coef, draws, lo, hi)
        for j, f in enumerate(names):
            rows.append({"block": lab, "feature": f, "coef": coef[j], "lo": lo[j], "hi": hi[j],
                         "n_matches": int(data.loc[m, "match_id"].nunique())})
    table = pd.DataFrame(rows)
    labs = list(store)
    sep = []
    if len(labs) >= 2:
        f_lab, l_lab = labs[0], labs[-1]
        d = store[l_lab][1] - store[f_lab][1]
        point_d = store[l_lab][0] - store[f_lab][0]
        p = np.array([bs.boot_se_p(point_d[j], d[:, j]) for j in range(d.shape[1])])
        p_holm = bs.holm(p.tolist())
        dlo, dhi = _ci(d)
        for j, f in enumerate(names):
            nonoverlap = any(
                store[a][2][j] > store[b][3][j] or store[b][2][j] > store[a][3][j]
                for ia, a in enumerate(labs) for b in labs[ia + 1 :]
            )
            sep.append({"feature": f, "first_block": f_lab, "last_block": l_lab,
                        "first_coef": store[f_lab][0][j], "last_coef": store[l_lab][0][j],
                        "diff": store[l_lab][0][j] - store[f_lab][0][j], "diff_lo": dlo[j],
                        "diff_hi": dhi[j], "p_holm_first_last": p_holm[j],
                        "any_pair_ci_nonoverlap": bool(nonoverlap),
                        "highlight": f in P.trends.highlight})
    return {"blocks": labs, "table": table, "separation": pd.DataFrame(sep), "spec": spec}


def coaches_vs_umpires(pm: pd.DataFrame, spec: dict, P: SimpleNamespace, start: int) -> dict:
    """Same M1 spec fitted on coaches' bands and Brownlow bands (Train + Validation, shared scaler)."""
    last = max(P.splits.val_seasons)
    data = pm[(pm["season"] >= start) & (pm["season"] <= last)]
    tot = data.groupby("match_id")["brownlow_votes"].transform("sum")
    data = data[tot == 6]  # 3+2+1; matches without a valid Brownlow tally cannot be used
    prep = Prep(P).fit(data)
    X, names = prep.view(data, spec["view"])
    B = P.q2.brownlow_bootstrap_B
    cc, cd = bootstrap_coefs(X, bands_to_codes(data["votes"].to_numpy(), P.votes.bands),
                             data["match_id"].to_numpy(), len(P.votes.bands), spec["alpha"], B,
                             P.seed, "q2|coaches")
    bc, bd = bootstrap_coefs(X, bands_to_codes(data["brownlow_votes"].to_numpy(), P.votes.brownlow_bands),
                             data["match_id"].to_numpy(), len(P.votes.brownlow_bands), spec["alpha"],
                             B, P.seed, "q2|umpires")
    clo, chi = _ci(cd)
    blo, bhi = _ci(bd)
    rows = [{"feature": f, "coaches": cc[j], "coaches_lo": clo[j], "coaches_hi": chi[j],
             "umpires": bc[j], "umpires_lo": blo[j], "umpires_hi": bhi[j],
             "differs": bool(clo[j] > bhi[j] or blo[j] > chi[j])} for j, f in enumerate(names)]
    return {"table": pd.DataFrame(rows), "n_matches": int(data["match_id"].nunique())}
