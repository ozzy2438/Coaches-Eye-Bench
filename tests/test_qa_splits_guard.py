import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from ceb import guard
from ceb.config import PROTOCOL_PATH
from ceb.features import Prep, assert_no_leakage
from ceb.pipeline import join_and_check
from ceb.qa import QAFailure, assert_qa, run_qa
from ceb.splits import TestSeasonLocked, effective_start, feature_coverage, make_splits
from ceb.synth import generate


@pytest.fixture(scope="module")
def pm(SP):
    sy = generate([2022, 2023, 2024, 2025], 40, seed=3)
    res, qa, _ = join_and_check(sy.stats_raw, sy.votes_raw, SP, include_test=True)
    return res, qa


def test_qa_passes_on_clean_data(pm):
    assert pm[1]["ok"], pm[1]


@pytest.mark.parametrize("mutate,check", [
    (lambda d: d.assign(votes=d["votes"].where(d.index != d.index[d["votes"] > 0][0], 0)), "votes_per_match_total"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "no_duplicate_player_matches"),
    (lambda d: d.assign(tackles=d["tackles"].where(d.index != 5, -1)), "stats_non_negative"),
    (lambda d: d.assign(time_on_ground=d["time_on_ground"].where(d.index != 5, 140)), "time_on_ground_0_100"),
    (lambda d: d.assign(goals=d["goals"].where(d.index != 5, 50)), "stats_within_sane_maxima"),
    (lambda d: d.assign(margin=d["margin"].where(d.index != 5, 999)), "context_margins_antisymmetric"),
])
def test_qa_catches_corruption(pm, SP, mutate, check):
    bad = mutate(pm[0].player_match.copy())
    rep = run_qa(bad, SP, include_test=True)
    assert not rep["checks"][check]["ok"], rep["checks"][check]
    with pytest.raises(QAFailure):
        assert_qa(rep)


def test_qa_flags_test_season_rows_in_dev_data(pm, SP):
    rep = run_qa(pm[0].player_match, SP, include_test=False)  # contains 2025 = smoke test season
    assert not rep["checks"]["no_test_season_rows_in_dev_data"]["ok"]


def test_leakage_guard_rejects_forbidden_columns():
    for bad in ("brownlow_votes", "votes", "player_id", "coaches_votes", "team", "age", "career_games",
                "round", "season", "match_id", "role"):
        with pytest.raises(AssertionError):
            assert_no_leakage(["kicks", bad])
    assert_no_leakage(["time_on_ground", "marks_inside_50_mz", "won", "margin_10"])


def test_prep_views_contain_no_forbidden_columns_and_match_z_is_standardised(pm, SP):
    df = pm[0].player_match
    prep = Prep(SP).fit(df)
    for view in ("raw", "global", "match_z", "both"):
        X, names = prep.view(df, view)
        assert X.shape == (len(df), len(names)) and not np.isnan(X).any()
        assert_no_leakage(names)
    X, names = prep.view(df, "match_z")
    z = pd.DataFrame(X[:, : len(prep.stats)], columns=prep.stats).groupby(df["match_id"].to_numpy())
    assert np.allclose(z.mean().to_numpy(), 0, atol=1e-9)
    X, _ = prep.view(df, "both")
    assert X.shape[1] == 2 * len(prep.stats) + len(prep.context)


def test_make_splits_by_time_and_test_lock(pm, SP):
    df = pm[0].player_match
    with pytest.raises(TestSeasonLocked):
        make_splits(df, SP)
    dev = df[df["season"] != SP.splits.test_season]
    sp = make_splits(dev, SP)
    assert set(sp["train"]["season"]) == {2022, 2023} and set(sp["val"]["season"]) == {2024}
    full = make_splits(df, SP, allow_test=True)
    assert set(full["test"]["season"]) == {SP.splits.test_season}
    assert max(sp["train"]["season"]) < min(sp["val"]["season"]) < min(full["test"]["season"])


def _report(shares, stats_matches=100):
    return {"per_season": {s: {"stats_matches": stats_matches,
                               "excluded_by_reason": {"vote_total_not_30": round((1 - sh) * stats_matches)}}
                           for s, sh in shares.items()}}


def test_effective_start_rule(P):
    seasons = list(range(2012, 2026))
    cov = pd.DataFrame(1.0, index=seasons, columns=list(P.features.stats))
    assert effective_start(_report({s: 1.0 for s in seasons}), cov, P)[0] == 2012
    # votes unavailable before 2017 -> start moves, with the reason recorded
    shares = {s: (0.0 if s < 2017 else 1.0) for s in seasons}
    start, notes = effective_start(_report(shares), cov, P)
    assert start == 2017 and notes and "start moved to 2017" in notes[0]
    # low feature coverage in some seasons moves it too
    cov2 = cov.copy()
    cov2.loc[:2015, "time_on_ground"] = 0.5
    assert effective_start(_report({s: 1.0 for s in seasons}), cov2, P)[0] == 2016
    # nothing qualifies by the latest allowed start -> stop and report, never silently continue
    with pytest.raises(RuntimeError, match="stop and report"):
        effective_start(_report({s: (0.0 if s < 2022 else 1.0) for s in seasons}), cov, P)


def test_feature_coverage_shape(pm, SP):
    cov = feature_coverage(pm[0].player_match, SP)
    assert set(cov.columns) == set(SP.features.stats) and (cov <= 1).all().all()


# ---- git guard on a throw-away repo -----------------------------------------------------------------
def _repo(tmp_path):
    def run(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True,
                       capture_output=True)
    run("init", "-q")
    shutil.copy(PROTOCOL_PATH, tmp_path / "protocol.md")
    run("add", "protocol.md")
    run("commit", "-qm", "protocol")
    return run, tmp_path / "protocol.md"


def test_guard_requires_tag_unchanged_protocol_choices_and_single_run(tmp_path):
    run, proto = _repo(tmp_path)
    results = tmp_path / "results"
    with pytest.raises(guard.GuardError, match="does not exist"):
        guard.check_test_allowed(results, tmp_path, proto)
    run("tag", "-a", guard.TAG, "-m", "frozen")
    with pytest.raises(guard.GuardError, match="frozen_choices"):
        guard.check_test_allowed(results, tmp_path, proto)
    (results / "val").mkdir(parents=True)
    (results / "val" / "frozen_choices.json").write_text("{}")
    guard.check_test_allowed(results, tmp_path, proto)  # all conditions met
    proto.write_text(proto.read_text() + "\nsilent edit\n")
    with pytest.raises(guard.GuardError, match="differs"):
        guard.check_test_allowed(results, tmp_path, proto)
    run("checkout", "protocol.md")
    (results / "test").mkdir()
    (results / "test" / "test_run.json").write_text("{}")
    with pytest.raises(guard.GuardError, match="already been evaluated"):
        guard.check_test_allowed(results, tmp_path, proto)
