"""Synthetic fixture in the *raw* fitzRoy schemas, with known planted effects.

Purpose: exercise every code path (ingest, name matching, join, QA, models, inference, Q2/Q3)
offline, and prove the machinery recovers effects whose truth we control. Nothing produced here is
ever a result: outputs are stamped ``synthetic`` and live under ``results/smoke``.

Planted effects (all switchable):
  * tackles matter more to coaches each season (trend, for Q3)
  * defenders get a coaches' bonus that no box-score stat records (blind spot, for Q2)
  * a non-linear bonus for 4+ goals and 35+ disposals (so a flexible model can beat a linear one)
  * umpires (Brownlow) weigh stats differently from coaches (for the coaches-vs-umpires comparison)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .teams import TEAMS

AFLTABLES_NAME = {"GWS": "Greater Western Sydney"}
AFLCA_NAME = {
    "Adelaide": "Adelaide Crows", "Geelong": "Geelong Cats", "Gold Coast": "Gold Coast Suns",
    "GWS": "GWS Giants", "Sydney": "Sydney Swans", "West Coast": "West Coast Eagles",
}
STATS = [
    "kicks", "handballs", "marks", "contested_marks", "contested_possessions",
    "uncontested_possessions", "goals", "behinds", "goal_assists", "marks_inside_50",
    "inside_50s", "tackles", "hit_outs", "clearances", "rebounds", "clangers", "frees_for",
    "frees_against", "one_percenters", "bounces",
]
# role -> mean per game at 100% time on ground, order as STATS
_BASE = {
    "ruck": [6, 5, 3.5, 1.2, 8, 5, .5, .4, .3, .5, 1, 3, 22, 4, 1, 2, 1.5, 1.5, 2.5, 0],
    "mid": [11, 10, 3, .8, 12, 12, .5, .4, .5, .3, 3, 4.5, .3, 5.5, 1.5, 3, 1.5, 1, 1.5, .8],
    "forward": [9, 4, 5, 1.5, 5, 8, 1.6, 1.1, .8, 2, 1, 2.5, .2, 1, .3, 2, 1, 1, 1.2, 0],
    "defender": [11, 7, 5, 1, 5, 13, .1, .1, .2, .1, .3, 2, .1, .8, 3.5, 2.5, .6, 1, 4.5, .5],
    "other": [10, 8, 3.5, .6, 7, 12, .5, .4, .4, .3, 2.5, 3, .1, 2, 1.5, 2.5, 1, 1, 1.5, 1.5],
}
_QUOTA = {"ruck": 1, "mid": 6, "forward": 5, "defender": 6, "other": 4}
_SQUAD = {"ruck": 2, "mid": 8, "forward": 7, "defender": 8, "other": 6}
_W_COACH = dict(
    kicks=.25, handballs=.15, marks=.15, contested_marks=.15, contested_possessions=.45,
    uncontested_possessions=.10, goals=.55, behinds=.05, goal_assists=.20, marks_inside_50=.20,
    inside_50s=.30, tackles=.10, hit_outs=.15, clearances=.35, rebounds=.25, clangers=-.20,
    frees_for=.05, frees_against=-.15, one_percenters=.10, bounces=.05, time_on_ground=.20,
)
_W_UMP = dict(
    kicks=.35, handballs=.25, marks=.2, contested_marks=.1, contested_possessions=.2, goals=.4,
    goal_assists=.1, inside_50s=.15, tackles=.0, clearances=.2, frees_for=.3, frees_against=-.1,
    clangers=-.1, time_on_ground=.1,
)
_FIRST = [
    ("Matthew", "Matt"), ("Zachary", "Zac"), ("Cameron", "Cam"), ("Jacob", "Jake"),
    ("Mitchell", "Mitch"), ("Nicholas", "Nick"), ("Christopher", "Chris"), ("Benjamin", "Ben"),
    ("Jordan", "Jordan"), ("Lachlan", "Lachie"), ("Harrison", "Harry"), ("Samuel", "Sam"),
    ("Thomas", "Tom"), ("Joshua", "Josh"), ("Daniel", "Dan"), ("Patrick", "Paddy"),
    ("Callum", "Callum"), ("Hugh", "Hugh"), ("Brodie", "Brodie"), ("Oscar", "Oscar"),
    ("Luke", "Luke"), ("Max", "Max"), ("Noah", "Noah"), ("Finn", "Finn"), ("Zane", "Zane"),
    ("Riley", "Riley"), ("Connor", "Connor"), ("Bailey", "Bailey"), ("Archie", "Archie"),
    ("Kane", "Kane"), ("Nathan", "Nate"), ("Alexander", "Alex"), ("Andrew", "Andy"),
]
_SURNAME = [
    "Smith", "Jones", "Brown", "Taylor", "Wilson", "O'Brien", "O'Meara", "De Goey", "Van Rooyen",
    "McKay", "Pearce-Paul", "Müller", "Nguyen", "Kennedy", "Murphy", "Walker", "Hall", "Young",
    "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Hill", "Campbell", "Mitchell",
    "Roberts", "Carter", "Phillips", "Evans", "Turner", "Parker", "Collins", "Edwards", "Stewart",
    "Morris", "Rogers", "Reed", "Cook", "Bell", "Cooper", "Richardson", "Cox", "Howard", "Ward",
    "Peterson", "Gray", "James", "Watson", "Brooks", "Kelly", "Sanders", "Price", "Bennett",
    "Wood", "Barnes", "Ross", "Henderson", "Coleman", "Jenkins", "Perry", "Powell", "Long",
    "Patterson", "Hughes", "Flores", "Washington", "Butler", "Simmons", "Foster", "Gonzales",
    "Bryant", "Alexander", "Russell", "Griffin", "Hayes", "Myers", "Ford", "Hamilton", "Graham",
    "Sullivan", "Wallace", "Woods", "Cole", "West", "Jordan", "Owens", "Reynolds", "Fisher",
    "Ellis", "Harrison", "Gibson", "McDonald", "Cruz", "Marshall", "Ortiz", "Gomez", "Murray",
    "Freeman", "Wells", "Webb", "Simpson", "Stevens", "Tucker", "Porter", "Hunter", "Hicks",
    "Crawford", "Henry", "Boyd", "Mason", "Morales", "Kennedy-Lowe", "Åberg", "Núñez", "Béla",
]


@dataclass
class Planted:
    tackle_trend: float = 0.03  # added weight on tackles per season
    defender_bonus: float = 0.45
    nonlinear: float = 1.0  # scale of the 4+ goals / 35+ disposals bonus (0 = purely linear)
    coach_noise: float = 1.7
    ump_noise: float = 1.4


@dataclass
class Synth:
    stats_raw: pd.DataFrame
    votes_raw: pd.DataFrame
    truth: dict = field(default_factory=dict)


_SYL = ["bar", "ben", "cal", "dor", "el", "fen", "gar", "hol", "iv", "jar", "kel", "lan", "mor",
        "nor", "ol", "per", "quin", "ros", "sal", "tay", "ul", "ven", "wil", "yar", "zel", "ing",
        "son", "ford", "ton", "man", "ley", "well", "berg", "stone", "field", "wick", "ham"]


def _surname(rng: np.random.Generator) -> str:
    if rng.random() < 0.18:  # tricky spellings: accents, apostrophes, hyphens, two words
        return _SURNAME[rng.integers(len(_SURNAME))]
    return "".join(rng.choice(_SYL, size=rng.integers(2, 4))).capitalize()


def _name(rng: np.random.Generator, taken: set) -> tuple[str, str, str]:
    while True:
        full, short = _FIRST[rng.integers(len(_FIRST))]
        sur = _surname(rng)
        if (full, sur) not in taken:
            taken.add((full, sur))
            return full, short, sur


def _pairings(rng: np.random.Generator, n_matches: int) -> list[list[tuple[str, str]]]:
    rounds, left = [], n_matches
    while left > 0:
        t = list(rng.permutation(TEAMS))
        pairs = [(t[i], t[i + 1]) for i in range(0, 18, 2)][: min(9, left)]
        rounds.append(pairs)
        left -= len(pairs)
    return rounds


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std(0)
    return (x - x.mean(0)) / np.where(sd == 0, 1, sd)


def generate(
    seasons: list[int],
    matches_per_season: int = 90,
    seed: int = 0,
    planted: Planted | None = None,
    include_finals: bool = True,
    flaw_rate: float = 0.0,
    perturb_names: bool = True,
) -> Synth:
    pl = planted or Planted()
    rng = np.random.default_rng(seed)
    next_id = [10_000]
    taken: set = set()
    squads: dict[str, list[dict]] = {}

    def new_player(role: str) -> dict:
        full, short, sur = _name(rng, taken)
        next_id[0] += 1
        return dict(id=next_id[0], first=full, short=short, sur=sur, role=role,
                    skill=float(rng.normal()), used_names=set())

    for t in TEAMS:
        squads[t] = [new_player(r) for r, n in _SQUAD.items() for _ in range(n)]

    stat_rows, vote_rows = [], []
    for si, season in enumerate(seasons):
        for t in TEAMS:  # turnover
            for k in rng.choice(len(squads[t]), 5, replace=False):
                squads[t][k] = new_player(squads[t][k]["role"])
        strength = dict(zip(TEAMS, rng.normal(0, 0.6, 18), strict=True))
        for ri, pairs in enumerate(_pairings(rng, matches_per_season), start=1):
            for home, away in pairs:
                s, v = _simulate_match(rng, pl, season, si, ri, home, away, squads, strength,
                                       perturb_names)
                stat_rows.append(s)
                vote_rows.append(v)
        if include_finals:
            h, a = TEAMS[0], TEAMS[1]
            s, _ = _simulate_match(rng, pl, season, si, 25, h, a, squads, strength, False)
            s["Round"] = "QF"
            stat_rows.append(s)

    stats = pd.concat(stat_rows, ignore_index=True)
    votes = pd.concat(vote_rows, ignore_index=True)
    if flaw_rate > 0:  # break the 30-vote total in a few matches
        keys = votes.drop_duplicates(["Season", "Round", "Home.Team"])
        bad = keys.sample(frac=flaw_rate, random_state=seed)
        for season, rnd, home in bad[["Season", "Round", "Home.Team"]].itertuples(index=False):
            m = (votes["Season"] == season) & (votes["Round"] == rnd) & (votes["Home.Team"] == home)
            votes = votes.drop(votes[m].index[:1])
    return Synth(stats, votes.reset_index(drop=True), {"planted": pl, "seed": seed})


def _simulate_match(rng, pl, season, si, rnd, home, away, squads, strength, perturb):
    players = []
    for team in (home, away):
        for role, q in _QUOTA.items():
            pool = [p for p in squads[team] if p["role"] == role]
            for k in rng.choice(len(pool), q, replace=False):
                players.append((team, pool[k]))
    n = len(players)
    roles = [p["role"] for _, p in players]
    skill = np.array([p["skill"] for _, p in players])
    is_home = np.array([t == home for t, _ in players])
    base = np.array([_BASE[r] for r in roles])
    tog = np.clip(rng.normal(82, 10, n), 25, 100)
    tog[[10, 32]] = rng.uniform(30, 60, 2)  # one sub per side
    perf = rng.normal(0, 1, n)
    tempo = np.exp(rng.normal(0, 0.08))
    edge = strength[home] - strength[away]
    team_f = np.where(is_home, np.exp(0.10 * edge), np.exp(-0.10 * edge))
    mult = (np.exp(0.18 * skill + 0.22 * perf) * tog / 82 * tempo)[:, None] * np.ones((1, 20))
    atk = np.zeros(20, bool)
    atk[[6, 7, 8, 9, 10]] = True  # goals, behinds, goal_assists, marks_inside_50, inside_50s
    mult = mult * np.where(atk[None, :], team_f[:, None], 1.0)
    counts = rng.poisson(base * mult).astype(float)
    c = dict(zip(STATS, counts.T, strict=True))
    goals_h = c["goals"][is_home].sum()
    goals_a = c["goals"][~is_home].sum()
    hs = int(6 * goals_h + c["behinds"][is_home].sum())
    as_ = int(6 * goals_a + c["behinds"][~is_home].sum())
    margin_h = hs - as_
    margin = np.where(is_home, margin_h, -margin_h)
    won = np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0))

    feats = {**c, "time_on_ground": tog}
    z = {k: _z(v[:, None])[:, 0] for k, v in feats.items()}
    w = dict(_W_COACH)
    w["tackles"] = _W_COACH["tackles"] + pl.tackle_trend * si
    latent = sum(w[k] * z[k] for k in w)
    disposals = c["kicks"] + c["handballs"]
    clip = lambda a: np.clip(a, -4, 4)  # noqa: E731
    latent = latent + pl.nonlinear * (
        0.45 * (c["goals"] >= 4) + 0.25 * (disposals >= 35)
        + 0.35 * clip(z["clearances"]) * clip(z["tackles"])  # frequent interactions a linear model misses
        + 0.35 * clip(z["goals"]) * clip(z["marks_inside_50"])
    )
    latent = latent + pl.defender_bonus * np.array([r == "defender" for r in roles])
    latent = latent + 0.5 * won + 0.01 * margin

    votes = np.zeros(n, int)
    for _coach in range(2):
        zc = latent + rng.normal(0, pl.coach_noise, n)
        top = np.argsort(-zc)[:5]
        votes[top] += np.array([5, 4, 3, 2, 1])
    lu = sum(_W_UMP[k] * z[k] for k in _W_UMP) + 0.4 * won + rng.normal(0, pl.ump_noise, n)
    brown = np.zeros(n, int)
    brown[np.argsort(-lu)[:3]] = [3, 2, 1]

    date = pd.Timestamp(f"{season}-03-14") + pd.Timedelta(days=7 * (rnd - 1))
    tn = lambda t: AFLTABLES_NAME.get(t, t)  # noqa: E731
    cn = lambda t: AFLCA_NAME.get(t, t)  # noqa: E731
    stats = pd.DataFrame({
        "Season": season, "Round": str(rnd), "Date": date, "Home.team": tn(home),
        "Away.team": tn(away), "Home.score": hs, "Away.score": as_,
        "Playing.for": [tn(t) for t, _ in players], "First.name": [p["first"] for _, p in players],
        "Surname": [p["sur"] for _, p in players], "ID": [p["id"] for _, p in players],
        "Kicks": c["kicks"], "Handballs": c["handballs"], "Disposals": disposals,
        "Marks": c["marks"], "Contested.Marks": c["contested_marks"],
        "Contested.Possessions": c["contested_possessions"],
        "Uncontested.Possessions": c["uncontested_possessions"], "Goals": c["goals"],
        "Behinds": c["behinds"], "Goal.Assists": c["goal_assists"],
        "Marks.Inside.50": c["marks_inside_50"], "Inside.50s": c["inside_50s"],
        "Tackles": c["tackles"], "Hit.Outs": c["hit_outs"], "Clearances": c["clearances"],
        "Rebounds": c["rebounds"], "Clangers": c["clangers"], "Frees.For": c["frees_for"],
        "Frees.Against": c["frees_against"], "One.Percenters": c["one_percenters"],
        "Bounces": c["bounces"], "Time.on.Ground": np.round(tog),
        "Brownlow.Votes": brown,
    })
    ints = [col for col in stats.columns if col not in
            ("Season", "Round", "Date", "Home.team", "Away.team", "Playing.for", "First.name",
             "Surname", "Time.on.Ground")]
    stats[ints] = stats[ints].astype(int)

    rec = np.flatnonzero(votes > 0)
    names = []
    for i in rec:
        _, p = players[i]
        first, sur = p["first"], p["sur"]
        if perturb:
            if rng.random() < 0.5:
                first = p["short"]
            if rng.random() < 0.9:
                sur = sur.replace("'", "’") if rng.random() < 0.5 else sur
        nm = f"{first} {sur}"
        if perturb and rng.random() < 0.03:
            nm = f"{sur}, {first}"
        elif perturb and rng.random() < 0.10:
            nm = f"{nm} ({cn(players[i][0])})"
        names.append(nm)
    vdf = pd.DataFrame({
        "Season": season, "Round": rnd, "Home.Team": cn(home), "Away.Team": cn(away),
        "Player.Name": names, "Coaches.Votes": votes[rec].astype(str),
    })
    return stats, vdf
