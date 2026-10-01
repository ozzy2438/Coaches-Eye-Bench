"""Canonical team names. Unknown names raise: a silent pass-through would corrupt the join."""

from __future__ import annotations

import re

import pandas as pd

_ALIASES = {
    "Adelaide": ["adelaide", "adelaide crows", "crows", "kuwarna"],
    "Brisbane Lions": ["brisbane lions", "brisbane", "lions"],
    "Carlton": ["carlton", "carlton blues", "blues"],
    "Collingwood": ["collingwood", "collingwood magpies", "magpies", "pies"],
    "Essendon": ["essendon", "essendon bombers", "bombers"],
    "Fremantle": ["fremantle", "fremantle dockers", "dockers", "walyalup"],
    "Geelong": ["geelong", "geelong cats", "cats"],
    "Gold Coast": ["gold coast", "gold coast suns", "suns"],
    "GWS": ["gws", "gws giants", "greater western sydney", "giants", "gw sydney"],
    "Hawthorn": ["hawthorn", "hawthorn hawks", "hawks"],
    "Melbourne": ["melbourne", "melbourne demons", "demons", "narrm"],
    "North Melbourne": ["north melbourne", "north melbourne kangaroos", "kangaroos"],
    "Port Adelaide": ["port adelaide", "port adelaide power", "power", "yartapuulti"],
    "Richmond": ["richmond", "richmond tigers", "tigers"],
    "St Kilda": ["st kilda", "st kilda saints", "saints", "euro-yroke", "euro yroke"],
    "Sydney": ["sydney", "sydney swans", "swans"],
    "West Coast": ["west coast", "west coast eagles", "eagles", "waalitj marawar", "wallitj marawar"],
    "Western Bulldogs": ["western bulldogs", "footscray", "bulldogs"],
}
TEAMS = sorted(_ALIASES)
_LOOKUP = {a: canon for canon, al in _ALIASES.items() for a in al}
# Club hints printed beside AFLCA player names (verified in the 2012 source).
_AFLCA_CLUB_HINTS = {
    "ADEL": "Adelaide", "BL": "Brisbane Lions", "CARL": "Carlton",
    "COLL": "Collingwood", "ESS": "Essendon", "FRE": "Fremantle",
    "GCFC": "Gold Coast", "GEEL": "Geelong", "GWS": "GWS", "HAW": "Hawthorn",
    "MELB": "Melbourne", "NMFC": "North Melbourne", "PORT": "Port Adelaide",
    "RICH": "Richmond", "STK": "St Kilda", "SYD": "Sydney", "WB": "Western Bulldogs",
    "WCE": "West Coast",
}
_LOOKUP.update({code.lower(): team for code, team in _AFLCA_CLUB_HINTS.items()})


def canon_team(name: str) -> str:
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    try:
        return _LOOKUP[key]
    except KeyError:
        raise ValueError(f"unknown team name {name!r}; add it to ceb.teams") from None


def canon_team_series(s: pd.Series) -> pd.Series:
    uniq = {u: canon_team(u) for u in s.dropna().unique()}
    return s.map(uniq)


def team_slug(canon: str) -> str:
    return re.sub(r"[^a-z]+", "_", canon.lower()).strip("_")
