"""Static Match Review Card: model ranking vs coaches' votes for one round, biggest disagreements first.

Not affiliated with any club or the AFL. Player-level votes are not shipped in this repo; generate
the card locally from your own `make eval-test` output, or see docs/demo_card_synthetic.html.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e1;--hot:#b4402a;--blue:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--mut:#c3c2b7;--line:#383835;--hot:#e66767;--blue:#3987e5}}
body{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:900px;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 6px}.mut{color:var(--mut);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.hot{color:var(--hot);font-weight:600}.banner{border:1px solid var(--hot);color:var(--hot);padding:6px 10px;margin:8px 0}
@media (max-width:560px){table{font-size:12.5px}th,td{padding:4px 5px}}
"""


def _match_table(g: pd.DataFrame, top: int = 8) -> str:
    g = g.copy()
    g["model_rank"] = g["m1"].rank(ascending=False, method="first").astype(int)
    g["coach_rank"] = g["votes"].rank(ascending=False, method="min").astype(int)
    g["gap"] = g["model_rank"] - g["coach_rank"]
    show = g[(g["model_rank"] <= 5) | (g["votes"] > 0)].copy()
    show["disagree"] = (show["model_rank"].clip(upper=12) - show["coach_rank"].clip(upper=12)).abs()
    show = show.sort_values(["model_rank"]).head(max(top, 10))
    big = set(show.sort_values("disagree", ascending=False).head(3).index)
    rows = []
    for i, r in show.iterrows():
        cls = ' class="hot"' if i in big and r["disagree"] >= 3 else ""
        rows.append(
            f"<tr><td class='n'>{r['model_rank']}</td><td{cls}>{html.escape(r['first_name'] + ' ' + r['surname'])}"
            f"</td><td>{html.escape(r['team'])}</td><td class='n'>{int(r['votes'])}</td>"
            f"<td class='n'>{'' if r['votes'] == 0 else int(r['coach_rank'])}</td></tr>")
    return ("<table><thead><tr><th>Model rank</th><th>Player</th><th>Team</th><th>Coaches' votes</th>"
            "<th>Coaches' rank</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def build_card(scored: pd.DataFrame, season: int, rnd: int, out: Path, synthetic: bool = False) -> Path:
    d = scored[(scored["season"] == season) & (scored["round"] == rnd)]
    if d.empty:
        raise ValueError(f"no matches for season {season} round {rnd}")
    parts = [f"<h1>Match Review Card - {season} round {rnd}</h1>",
             "<p class='mut'>Interpretable model (M1) ranking from public box scores vs AFL Coaches Association votes. "
             "Red names are the biggest disagreements. Post-match evaluation, not a prediction. "
             "Not affiliated with any club or the AFL.</p>"]
    if synthetic:
        parts.append("<div class='banner'>SYNTHETIC DATA - fictional players, for demonstration only.</div>")
    for _mid, g in d.groupby("match_id", sort=True):
        h, a = g["home_team"].iloc[0], g["away_team"].iloc[0]
        hs, as_ = int(g["team_score"][g["team"] == h].iloc[0]), int(g["team_score"][g["team"] == a].iloc[0])
        parts.append(f"<h2>{html.escape(h)} {hs} - {as_} {html.escape(a)}</h2>")
        parts.append(_match_table(g))
    doc = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
           "content='width=device-width,initial-scale=1'><title>Match Review Card</title>"
           f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out
