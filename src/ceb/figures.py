"""Matplotlib figures (light surface, direct labels, thin marks, one axis per chart)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COLOR = {"B0": "#8a8985", "B1": "#52514e", "M1": "#2a78d6", "M2": "#eb6834"}
NAME = {"B0": "B0 Disposals", "B1": "B1 Fantasy points", "M1": "M1 Interpretable", "M2": "M2 LightGBM"}


def _style(ax, title: str, subtitle: str | None = None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED)


def _save(fig, path: Path, stamp: str | None):
    fig.patch.set_facecolor(SURFACE)
    if stamp:
        fig.text(0.99, 0.01, stamp, ha="right", va="bottom", fontsize=7, color="#b03030")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def ndcg_chart(res: dict, path: Path, title: str, stamp: str | None = None):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    models = ["M2", "M1", "B1", "B0"]
    for i, m in enumerate(models):
        d = res["metrics"][m]["ndcg5"]
        ax.plot([d["lo"], d["hi"]], [i, i], color=COLOR[m], lw=2, solid_capstyle="round")
        ax.plot(d["point"], i, "o", color=COLOR[m], ms=8, mec=SURFACE, mew=2)
        ax.text(d["hi"] + 0.004, i, f"{d['point']:.3f}  [{d['lo']:.3f}, {d['hi']:.3f}]",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(models)), [NAME[m] for m in models])
    lo = min(res["metrics"][m]["ndcg5"]["lo"] for m in models)
    hi = max(res["metrics"][m]["ndcg5"]["hi"] for m in models)
    ax.set_xlim(lo - 0.02, hi + 0.12)
    ax.set_xlabel("NDCG@5 against coaches' votes (mean over matches, 95% bootstrap CI)", color=MUTED, fontsize=9)
    _style(ax, title, f"{res['n_matches']} matches")
    _save(fig, path, stamp)


def calibration_chart(scored: pd.DataFrame, path: Path, title: str, stamp: str | None = None, bins: int = 10):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2)
    for m, col in (("M1", "p_m1"), ("M2", "p_m2")):
        b = np.minimum((scored[col].clip(0, 1) * bins).astype(int), bins - 1)
        g = scored.assign(b=b, y=(scored["votes"] > 0).astype(float)).groupby("b").agg(
            p=(col, "mean"), y=("y", "mean"), n=("y", "size"))
        ax.plot(g["p"], g["y"], "-o", color=COLOR[m], lw=1.6, ms=5, mec=SURFACE, mew=1.5, label=NAME[m])
    ax.set_xlabel("predicted P(receives any votes)", color=MUTED, fontsize=9)
    ax.set_ylabel("observed share", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style(ax, title)
    _save(fig, path, stamp)


def coef_chart(cvu: pd.DataFrame, path: Path, title: str, stamp: str | None = None):
    t = cvu.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(t) + 1.6))
    y = np.arange(len(t))
    for key, col, dy in (("coaches", "#2a78d6", 0.15), ("umpires", "#1baf7a", -0.15)):
        ax.hlines(y + dy, t[f"{key}_lo"], t[f"{key}_hi"], color=col, lw=1.6)
        ax.plot(t[key], y + dy, "o", color=col, ms=5, mec=SURFACE, mew=1.2, label=key.capitalize())
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, t["feature"], fontsize=8)
    ax.set_xlabel("effect on log-odds of a higher vote band (per feature unit, 95% CI)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    _style(ax, title)
    _save(fig, path, stamp)


def trend_chart(table: pd.DataFrame, features: list[str], path: Path, title: str, stamp: str | None = None):
    feats = [f for f in features if f in set(table["feature"])]
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.9), sharey=False, squeeze=False)
    blocks = list(dict.fromkeys(table["block"]))
    for ax, f in zip(axes[0], feats, strict=True):
        t = table[table["feature"] == f].set_index("block").loc[blocks]
        x = np.arange(len(blocks))
        ax.fill_between(x, t["lo"], t["hi"], color="#2a78d6", alpha=0.18, lw=0)
        ax.plot(x, t["coef"], "-o", color="#2a78d6", lw=1.8, ms=5, mec=SURFACE, mew=1.2)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xticks(x, blocks, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(f, fontsize=9, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    _save(fig, path, stamp)


def blindspot_chart(role: pd.DataFrame, closeness: pd.DataFrame, path: Path, title: str,
                    stamp: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, t, col in ((axes[0], role, "role"), (axes[1], closeness, "closeness")):
        t = t.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(t))
        ax.hlines(y, t["lo"], t["hi"], color="#2a78d6", lw=2)
        ax.plot(t["mean_resid"], y, "o", color="#2a78d6", ms=7, mec=SURFACE, mew=1.5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, t[col])
        ax.set_xlabel("mean (actual - expected) votes per player-match", color=MUTED, fontsize=8.5)
        _style(ax, "By role proxy" if col == "role" else "By match closeness")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    _save(fig, path, stamp)
