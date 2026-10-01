"""Hand-built raw frames in the fitzRoy schemas for edge-case tests."""

import pandas as pd

STAT_COLS = ["Kicks", "Handballs", "Disposals", "Marks", "Contested.Marks", "Contested.Possessions",
             "Uncontested.Possessions", "Goals", "Behinds", "Goal.Assists", "Marks.Inside.50",
             "Inside.50s", "Tackles", "Hit.Outs", "Clearances", "Rebounds", "Clangers", "Frees.For",
             "Frees.Against", "One.Percenters", "Bounces", "Brownlow.Votes"]


def stats_rows(players, season=2024, rnd="1", home="Collingwood", away="Carlton", hs=100, as_=80):
    """players: list of (team, first, surname, id)."""
    rows = []
    for team, first, sur, pid in players:
        r = {"Season": season, "Round": rnd, "Date": "2024-03-20", "Home.team": home, "Away.team": away,
             "Home.score": hs, "Away.score": as_, "Playing.for": team, "First.name": first,
             "Surname": sur, "ID": pid, "Time.on.Ground": 80.0}
        r.update({c: 5 for c in STAT_COLS if c != "Disposals"})
        r["Kicks"], r["Handballs"] = 10, 8
        r["Disposals"] = 18
        r["Brownlow.Votes"] = 0
        rows.append(r)
    return pd.DataFrame(rows)


def votes_rows(entries, season=2024, rnd=1, home="Collingwood", away="Carlton"):
    """entries: list of (player name string, votes)."""
    return pd.DataFrame([{"Season": season, "Round": rnd, "Home.Team": home, "Away.Team": away,
                          "Player.Name": n, "Coaches.Votes": str(v)} for n, v in entries])
