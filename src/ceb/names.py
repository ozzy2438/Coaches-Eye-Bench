"""Deterministic player-name normalisation for joining AFLCA votes to box scores."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_CLUB_SUFFIX = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def norm_name(s: str) -> str:
    """Lower-case, accent-free, punctuation-free, single-spaced."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[’'`´.]", "", s)  # apostrophes and full stops vanish
    s = re.sub(r"[-_,/]", " ", s)
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def name_key(first: str, surname: str) -> str:
    """Space-insensitive full-name key."""
    return norm_name(f"{first} {surname}").replace(" ", "")


def init_last(first: str, surname: str) -> str:
    f, s = norm_name(first), norm_name(surname)
    return (f[:1] + "|" + (s.split(" ")[-1] if s else "")) if f and s else ""


@dataclass(frozen=True)
class ParsedName:
    key: str
    init_last: str
    club: str | None
    clean: str


def parse_vote_name(raw: str) -> ParsedName:
    """AFLCA ``Player.Name`` -> keys. Tolerates ``First Last``, ``Last, First`` and ``Name (Club)``."""
    s = str(raw).strip()
    club = None
    m = _CLUB_SUFFIX.match(s)
    if m:
        s, club = m.group(1).strip(), m.group(2).strip() or None
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    toks = norm_name(s).split(" ")
    toks = [t for t in toks if t]
    if not toks:
        return ParsedName("", "", club, s)
    il = (toks[0][:1] + "|" + toks[-1]) if len(toks) > 1 else ""
    return ParsedName("".join(toks), il, club, s)
