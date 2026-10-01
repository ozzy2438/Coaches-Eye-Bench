"""Deterministic, audited join of AFLCA votes onto box scores.

Order of resolution inside a match (never across matches):
  alias  curated ``name_aliases.csv`` rewrites a votes-side key, then exact
  exact  normalised full-name key equal
  initial  first initial + last token, accepted only if unique on both sides among leftovers
Every vote row ends matched (with its tier) or in the unmatched log with a reason.
A match enters the modelling table only if its votes total 30 and none of its vote rows is unmatched.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd

from .teams import canon_team


class JoinRateError(RuntimeError):
    pass


@dataclass
class JoinResult:
    player_match: pd.DataFrame
    unmatched: pd.DataFrame
    excluded: pd.DataFrame
    report: dict


def load_aliases(path) -> dict[tuple[str, int | None], str]:
    df = pd.read_csv(path, dtype={"votes_name_key": str, "stats_name_key": str})
    out = {}
    for r in df.itertuples():
        season = None if pd.isna(r.season) else int(r.season)
        out[(r.votes_name_key, season)] = r.stats_name_key
    return out


def _crosswalk(stats: pd.DataFrame, votes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach ``match_id`` to vote rows.

    1. exact on (season, round, home, away);
    2. fallback on (season, home, away) only when that ordered pair occurs once in the season's box
       scores (AFLCA/AFL Tables round labels disagree) -- counted as ``round_mismatch``;
    3. a match whose vote rows carry several round labels (the site served the same page under a
       wrong round) keeps the rows whose label equals the box-score round; the rest are dropped
       and counted as duplicate pages.
    """
    sm = stats.drop_duplicates("match_id")[["season", "round", "home_team", "away_team", "match_id"]]
    pair = ["season", "home_team", "away_team"]
    v = votes.reset_index(drop=True)
    exact = v.merge(sm, on=[*pair, "round"], how="left")
    exact["stats_round"] = exact["round"].where(exact["match_id"].notna())
    todo = exact["match_id"].isna().to_numpy()
    if todo.any():
        once = sm.groupby(pair).filter(lambda g: len(g) == 1).rename(
            columns={"round": "stats_round_fb", "match_id": "match_id_fb"})
        fb = exact.loc[todo, list(v.columns)].merge(once, on=pair, how="left")
        exact.loc[todo, "match_id"] = fb["match_id_fb"].to_numpy()
        exact.loc[todo, "stats_round"] = fb["stats_round_fb"].to_numpy()
    dup_rows = 0
    multi = exact.dropna(subset=["match_id"]).groupby("match_id")["round"].nunique()
    for mid in multi[multi > 1].index:
        g = exact[exact["match_id"] == mid]
        keep_round = g["stats_round"].iloc[0] if (g["round"] == g["stats_round"]).any() else g["round"].min()
        drop = g.index[g["round"] != keep_round]
        dup_rows += len(drop)
        exact = exact.drop(index=drop)
    return exact.reset_index(drop=True), dup_rows


def join_votes(
    stats: pd.DataFrame,
    votes: pd.DataFrame,
    P: SimpleNamespace,
    aliases: dict | None = None,
) -> JoinResult:
    aliases = aliases or {}
    total = P.votes.total_per_match
    vm, dup_rows = _crosswalk(stats, votes)
    vm["vkey"] = [aliases.get((k, s), aliases.get((k, None), k))
                  for k, s in zip(vm["name_key"], vm["season"], strict=True)]
    vm["aliased"] = vm["vkey"] != vm["name_key"]
    vm["row"] = range(len(vm))
    st = stats.reset_index(drop=True).copy()
    st["srow"] = range(len(st))

    assigned: dict[int, tuple[int, str]] = {}  # vote row -> (stats row, tier)
    reason: dict[int, str] = {}
    for r in vm.index[vm["match_id"].isna()]:
        reason[r] = "match_not_in_stats"

    by_match = {m: g for m, g in st.groupby("match_id", sort=False)}
    for mid, vg in vm[vm["match_id"].notna()].groupby("match_id", sort=False):
        sg = by_match[mid]
        used: set[int] = set()
        pending = []
        # tier alias/exact
        for v in vg.itertuples():
            cand = sg[sg["name_key"] == v.vkey]
            if v.club_hint is not None and len(cand) > 1:
                try:
                    hint = canon_team(v.club_hint)
                    cand = cand[cand["team"] == hint]
                except ValueError:
                    pass
            cand = cand[~cand["srow"].isin(used)]
            if len(cand) == 1:
                s = int(cand["srow"].iloc[0])
                used.add(s)
                assigned[v.row] = (s, "alias" if v.aliased else "exact")
            elif len(cand) > 1:
                reason[v.row] = "ambiguous_name"
            else:
                pending.append(v)
        # tier initial (unique on both sides among leftovers)
        left = sg[~sg["srow"].isin(used)]
        for v in pending:
            if not v.init_last:
                reason[v.row] = "no_match"
                continue
            cand = left[left["init_last"] == v.init_last]
            rivals = [p for p in pending if p.init_last == v.init_last]
            if len(cand) == 1 and len(rivals) == 1:
                s = int(cand["srow"].iloc[0])
                assigned[v.row] = (s, "initial")
                left = left[left["srow"] != s]
            else:
                reason[v.row] = "no_match" if len(cand) == 0 else "ambiguous_initial"
        for v in vg.itertuples():
            if v.row not in assigned and v.row not in reason:
                reason[v.row] = "duplicate_vote_row"

    vm["matched"] = vm["row"].isin(assigned)
    vm["tier"] = vm["row"].map(lambda r: assigned[r][1] if r in assigned else None)
    vm["reason"] = vm["row"].map(reason)
    unmatched = vm[~vm["matched"]][
        ["season", "round", "home_team", "away_team", "name_raw", "votes", "reason"]
    ].reset_index(drop=True)

    # match inclusion
    mt = vm.groupby("match_id").agg(
        vote_total=("votes", "sum"), n_unmatched=("matched", lambda s: int((~s).sum()))
    )
    all_matches = stats.drop_duplicates("match_id").set_index("match_id")[["season", "round"]]
    ex = []
    for mid, row in all_matches.iterrows():
        if mid not in mt.index:
            ex.append((mid, row.season, row["round"], "no_votes_for_match", None))
        elif mt.loc[mid, "vote_total"] != total:
            ex.append((mid, row.season, row["round"], "vote_total_not_30", int(mt.loc[mid, "vote_total"])))
        elif mt.loc[mid, "n_unmatched"] > 0:
            ex.append((mid, row.season, row["round"], "unmatched_vote_rows", int(mt.loc[mid, "n_unmatched"])))
    excluded = pd.DataFrame(ex, columns=["match_id", "season", "round", "reason", "detail"])
    included = set(all_matches.index) - set(excluded["match_id"])

    pm = st[st["match_id"].isin(included)].copy()
    votes_arr = vm["votes"].to_numpy()
    vote_by_srow = {s: (int(votes_arr[r]), t) for r, (s, t) in assigned.items()}
    pm["votes"] = pm["srow"].map(lambda s: vote_by_srow[s][0] if s in vote_by_srow else 0).astype(int)
    pm["vote_tier"] = pm["srow"].map(lambda s: vote_by_srow[s][1] if s in vote_by_srow else None)
    pm = pm.drop(columns=["srow"]).reset_index(drop=True)

    report = _report(vm, excluded, all_matches, dup_rows)
    return JoinResult(pm, unmatched, excluded, report)


def _report(vm: pd.DataFrame, excluded: pd.DataFrame, all_matches: pd.DataFrame, dup_rows: int) -> dict:
    def block(g: pd.DataFrame) -> dict:
        rows, mass = len(g), int(g["votes"].sum())
        return {
            "vote_rows": rows,
            "matched_rows": int(g["matched"].sum()),
            "match_rate_rows": float(g["matched"].mean()) if rows else float("nan"),
            "vote_mass": mass,
            "matched_mass": int(g.loc[g["matched"], "votes"].sum()),
            "match_rate_mass": float(g.loc[g["matched"], "votes"].sum() / mass) if mass else float("nan"),
            "tiers": {k: int(v) for k, v in g["tier"].value_counts().items()},
        }

    per_season = {}
    for s, g in vm.groupby("season"):
        d = block(g)
        n_stats = int((all_matches["season"] == s).sum())
        ex = excluded[excluded["season"] == s]
        d.update(
            stats_matches=n_stats,
            included_matches=n_stats - len(ex),
            excluded_by_reason={k: int(v) for k, v in ex["reason"].value_counts().items()},
        )
        per_season[int(s)] = d
    return {
        "overall": block(vm),
        "per_season": per_season,
        "duplicate_vote_page_rows_dropped": dup_rows,
        "round_mismatch_rows": int((vm["stats_round"].notna() & (vm["round"] != vm["stats_round"])).sum()),
        "excluded_matches": int(len(excluded)),
    }


def check_join_rate(report: dict, P: SimpleNamespace) -> None:
    """Stop-and-report rule: every season and the total must reach the protocol join rate."""
    floor = P.coverage.min_join_rate
    bad = []
    for name, d in [("overall", report["overall"]), *[(str(s), d) for s, d in report["per_season"].items()]]:
        for k in ("match_rate_rows", "match_rate_mass"):
            if d[k] < floor:
                bad.append(f"{name} {k}={d[k]:.4f}")
    if bad:
        raise JoinRateError(
            f"join rate below protocol floor {floor}: " + "; ".join(bad)
            + ". Inspect data/processed/unmatched_votes.csv; add curated aliases or stop and report."
        )
