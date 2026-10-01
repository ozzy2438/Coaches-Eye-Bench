import numpy as np
import pandas as pd

from ceb import blindspots as BS
from ceb import trends as T
from ceb.models.m1 import OrdinalLogit
from ceb.roles import assign_roles


def test_role_rules_priority_and_min_games(SP, P):
    def player(key, n, **kw):
        base = dict(hit_outs=0, clearances=0, goals=0, marks_inside_50=0, rebounds=0, one_percenters=0)
        base.update(kw)
        return pd.DataFrame([{"season": 2024, "player_key": key, **base} for _ in range(n)])

    df = pd.concat([
        player("ruck", 10, hit_outs=20, clearances=6), player("mid", 10, clearances=5, goals=2),
        player("fwd", 10, goals=2), player("def", 10, rebounds=3), player("def2", 10, one_percenters=5),
        player("other", 10), player("rookie", 3, hit_outs=30)], ignore_index=True)
    r = assign_roles(df, P).groupby(df["player_key"]).first()
    assert r.to_dict() == {"ruck": "Ruck", "mid": "Midfielder", "fwd": "Forward", "def": "Defender",
                           "def2": "Defender", "other": "Other", "rookie": "Unclassified"}


def _frame(n_matches=250, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        votes = np.zeros(44)
        votes[rng.choice(44, 7, replace=False)] = rng.permutation([10, 7, 5, 4, 2, 1, 1])
        ev = 0.3 + 0.7 * votes / 3 + rng.uniform(0, 0.4, 44)  # a decent stand-in for model expectations
        role = np.array(["Defender"] * 11 + ["Midfielder"] * 11 + ["Forward"] * 11 + ["Other"] * 11)
        rng.shuffle(role)
        ev = np.where(role == "Defender", ev * (1 - shift), ev)  # the model under-predicts defenders
        rows.append(pd.DataFrame({"match_id": f"m{m}", "votes": votes, "ev_m1": ev, "role": role,
                                  "margin": rng.integers(-60, 60), "team": rng.choice(["A", "B", "C"], 44)}))
    return pd.concat(rows, ignore_index=True)


def test_residuals_respect_the_vote_budget_and_group_means_match_pandas(SP):
    f = _frame()
    zeros = {c: 0 for c in ("hit_outs", "clearances", "goals", "marks_inside_50", "rebounds", "one_percenters")}
    d = BS.add_residuals(f.assign(season=2024, player_key="x", **zeros), SP)
    assert np.allclose(d.groupby("match_id")["resid"].sum(), 0, atol=1e-9)
    d["role"] = f["role"]  # use the frame's labels; role assignment itself is tested separately
    s = BS.group_summary(d, "role", SP, "u")
    for lv, m in d.groupby("role")["resid"].mean().items():
        assert abs(s.set_index("role").loc[lv, "mean_resid"] - m) < 1e-12


def test_group_summary_detects_planted_under_prediction_and_not_a_null(SP):
    def defender_row(shift):
        d = _frame(shift=shift)
        tot = d.groupby("match_id")["ev_m1"].transform("sum")
        d["resid"] = d["votes"] - d["ev_m1"] * 30 / tot
        return BS.group_summary(d, "role", SP, f"x{shift}").set_index("role").loc["Defender"]
    planted, null = defender_row(0.4), defender_row(0.0)
    assert planted["lo"] > 0 and planted["p_holm"] < 0.05
    assert null["lo"] < 0 < null["hi"]


def test_season_blocks():
    assert T.season_blocks(2012, 2025, 3) == [[2012, 2013, 2014], [2015, 2016, 2017], [2018, 2019, 2020],
                                              [2021, 2022, 2023], [2024, 2025]]
    assert T.season_blocks(2012, 2024, 3)[-1] == [2021, 2022, 2023, 2024]  # lone remainder merges
    assert T.season_blocks(2022, 2024, 1) == [[2022], [2023], [2024]]


def test_bootstrap_coefs_cover_truth():
    rng = np.random.default_rng(1)
    n, beta = 6000, np.array([0.9, -0.5, 0.0])
    X = rng.normal(size=(n, 3))
    y = np.searchsorted([0.5, 1.8, 3.0], X @ beta + rng.logistic(size=n))
    mid = np.repeat(np.arange(n // 40), 40)
    pt, draws = T.bootstrap_coefs(X, y, mid, 4, 1e-3, 80, 3, "t")
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    assert ((lo <= beta) & (beta <= hi)).sum() >= 2 and draws.shape == (80, 3)
    assert np.allclose(pt, OrdinalLogit(4, 1e-3).fit(X, y).coef_)
