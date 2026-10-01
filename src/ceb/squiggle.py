"""Polite Squiggle client (cross-check of final scores only; never a feature source).

Requires CEB_CONTACT_EMAIL (Squiggle asks for a User-Agent with a contact). Responses are cached
on disk and an identical request is never repeated.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .teams import canon_team

API = "https://api.squiggle.com.au/"


def user_agent() -> str:
    email = os.environ.get("CEB_CONTACT_EMAIL", "").strip()
    if not email:
        raise RuntimeError("set CEB_CONTACT_EMAIL (Squiggle requires a contact in the User-Agent)")
    return f"coaches-eye-bench/0.1 ({email})"


def fetch_year(year: int, cache_dir: Path, sleep: float = 1.0) -> list[dict]:
    cache = Path(cache_dir) / f"squiggle_games_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())["games"]
    req = urllib.request.Request(f"{API}?q=games;year={year};complete=100",
                                 headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        payload = json.loads(r.read())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    time.sleep(sleep)
    return payload["games"]


def games_frame(games: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(games)
    df = df[(df["is_final"] == 0) & df["hscore"].notna()].copy()
    df["home_team"] = df["hteam"].map(canon_team)
    df["away_team"] = df["ateam"].map(canon_team)
    df["margin_home"] = df["hscore"] - df["ascore"]
    return df[["year", "round", "home_team", "away_team", "hscore", "ascore", "margin_home"]]


def cross_check(stats: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Compare AFL Tables final scores with Squiggle per (season, round, home, away)."""
    m = stats.drop_duplicates("match_id")[["season", "round", "home_team", "away_team", "home_score",
                                           "away_score"]]
    j = m.merge(games.rename(columns={"year": "season"}), on=["season", "round", "home_team", "away_team"],
                how="left", suffixes=("", "_sq"))
    missing = int(j["hscore"].isna().sum())
    mism = int(((j["home_score"] != j["hscore"]) | (j["away_score"] != j["ascore"]))[j["hscore"].notna()].sum())
    return {"matches": len(m), "not_in_squiggle": missing, "score_mismatches": mism}
