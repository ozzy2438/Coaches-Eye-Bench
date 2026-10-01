"""Raw fitzRoy tables -> canonical internal schema (home-and-away only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .names import init_last, name_key, parse_vote_name
from .teams import canon_team_series, team_slug

RAW_STATS_COLUMNS = {
    "Season": "season", "Round": "round", "Date": "date", "Home.team": "home_team",
    "Away.team": "away_team", "Home.score": "home_score", "Away.score": "away_score",
    "Playing.for": "team", "First.name": "first_name", "Surname": "surname", "ID": "player_id",
    "Kicks": "kicks", "Handballs": "handballs", "Disposals": "disposals_raw", "Marks": "marks",
    "Contested.Marks": "contested_marks", "Contested.Possessions": "contested_possessions",
    "Uncontested.Possessions": "uncontested_possessions", "Goals": "goals", "Behinds": "behinds",
    "Goal.Assists": "goal_assists", "Marks.Inside.50": "marks_inside_50",
    "Inside.50s": "inside_50s", "Tackles": "tackles", "Hit.Outs": "hit_outs",
    "Clearances": "clearances", "Rebounds": "rebounds", "Clangers": "clangers",
    "Frees.For": "frees_for", "Frees.Against": "frees_against",
    "One.Percenters": "one_percenters", "Bounces": "bounces",
    "Time.on.Ground": "time_on_ground", "Brownlow.Votes": "brownlow_votes",
}
_REQUIRED = [c for c in RAW_STATS_COLUMNS if c not in ("Disposals", "ID")]
STAT_INT_COLS = [
    "kicks", "handballs", "marks", "contested_marks", "contested_possessions",
    "uncontested_possessions", "goals", "behinds", "goal_assists", "marks_inside_50",
    "inside_50s", "tackles", "hit_outs", "clearances", "rebounds", "clangers", "frees_for",
    "frees_against", "one_percenters", "bounces",
]


def ingest_stats(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """AFL Tables player-match stats (as returned by fitzRoy) -> canonical home-and-away rows."""
    missing = [c for c in _REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"stats table is missing columns: {missing}")
    df = raw[[c for c in RAW_STATS_COLUMNS if c in raw.columns]].rename(columns=RAW_STATS_COLUMNS)
    n_in = len(df)
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    n_finals = int(df["round"].isna().sum())
    df = df[df["round"].notna()].copy()
    df["round"] = df["round"].astype(int)
    df["season"] = df["season"].astype(int)
    for c in ("home_team", "away_team", "team"):
        df[c] = canon_team_series(df[c])
    if not ((df["team"] == df["home_team"]) | (df["team"] == df["away_team"])).all():
        raise ValueError("a player's team is neither the home nor the away team of the match")
    if "player_id" not in df:
        df["player_id"] = np.nan
    for c in (*STAT_INT_COLS, "brownlow_votes"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["time_on_ground"] = pd.to_numeric(df["time_on_ground"], errors="coerce")
    df["disposals"] = df["kicks"] + df["handballs"]
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    is_home = df["team"] == df["home_team"]
    df["team_score"] = np.where(is_home, df["home_score"], df["away_score"])
    df["opp_score"] = np.where(is_home, df["away_score"], df["home_score"])
    df["margin"] = df["team_score"] - df["opp_score"]
    if df["margin"].isna().any():
        raise ValueError("missing final scores: cannot derive match context (won, margin)")
    df["won"] = np.select([df["margin"] > 0, df["margin"] == 0], [1.0, 0.5], 0.0)
    df["margin_10"] = df["margin"] / 10.0
    df["opp_team"] = np.where(is_home, df["away_team"], df["home_team"])
    df["match_id"] = (
        df["season"].astype(str) + "-R" + df["round"].astype(str).str.zfill(2) + "-"
        + df["home_team"].map(team_slug) + "-" + df["away_team"].map(team_slug)
    )
    df["name_key"] = [name_key(f, s) for f, s in zip(df["first_name"], df["surname"], strict=True)]
    df["init_last"] = [init_last(f, s) for f, s in zip(df["first_name"], df["surname"], strict=True)]
    pid = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64").astype("string")
    df["player_key"] = pid.where(pid.notna(), df["name_key"] + "|" + df["team"]).astype(str)
    return df.reset_index(drop=True), {"rows_in": n_in, "non_home_away_rows_dropped": n_finals}


def ingest_votes(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """AFLCA table (fitzRoy ``Season, Round, Home.Team, Away.Team, Player.Name, Coaches.Votes``)."""
    need = ["Season", "Round", "Home.Team", "Away.Team", "Player.Name", "Coaches.Votes"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"votes table is missing columns: {missing}")
    df = raw[need].rename(
        columns={"Season": "season", "Round": "round", "Home.Team": "home_team",
                 "Away.Team": "away_team", "Player.Name": "name_raw", "Coaches.Votes": "votes"}
    )
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    n_bad = int(df["votes"].isna().sum())
    df = df[df["votes"].notna()].copy()
    df["votes"] = df["votes"].astype(int)
    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    for c in ("home_team", "away_team"):
        df[c] = canon_team_series(df[c])
    parsed = [parse_vote_name(n) for n in df["name_raw"]]
    df["name_key"] = [p.key for p in parsed]
    df["init_last"] = [p.init_last for p in parsed]
    df["club_hint"] = [p.club for p in parsed]
    return df.reset_index(drop=True), {"rows_in": len(raw), "non_numeric_vote_rows_dropped": n_bad}
