"""Does the machinery recover effects whose truth we planted - and stay quiet when there are none?

Synthetic data only. Marked slow (~45 s per scenario). These tests prove the pipeline is able to find
what it is supposed to find; they say nothing about the AFL.
"""

import copy

import pytest

from ceb import evaluate as E
from ceb import trends as T
from ceb.config import _ns, load_params_dict
from ceb.pipeline import join_and_check
from ceb.splits import make_splits
from ceb.synth import Planted, generate

pytestmark = pytest.mark.slow


def _params():
    d = copy.deepcopy(load_params_dict())
    d["splits"] = dict(train_start=2014, train_end=2019, val_seasons=[2020, 2021], test_season=2022,
                       latest_start_allowed=2016)
    d["inference"].update(bootstrap_B=300, permutation_n=3000)
    d["m1"] = dict(alpha_grid=[0.001, 0.01], view_grid=["global", "match_z"])
    d["m2"].update(num_leaves_grid=[7, 15], min_data_in_leaf_grid=[50], rounds_grid=[50, 100, 200, 300])
    d["q2"].update(bootstrap_B=300, brownlow_bootstrap_B=60)
    d["trends"].update(bootstrap_B=60, block_width=3, last_season_included=2021)
    return _ns(d)


def _run(planted):
    P = _params()
    sy = generate(list(range(2014, 2022)), 150, seed=11, planted=planted, include_finals=False)
    pm = join_and_check(sy.stats_raw, sy.votes_raw, P)[0].player_match
    sp = make_splits(pm, P)
    choices, _ = E.select_all(sp["train"], sp["val"], P)
    scored = E.fit_models(sp["train"], choices, P).score(sp["val"])
    res, _ = E.evaluate_scores(scored, P, "recovery")
    q3 = T.block_trends(pm, choices["m1"], 2014, P)["separation"].set_index("feature")
    return res, q3


@pytest.fixture(scope="module")
def planted():
    return _run(Planted(tackle_trend=0.10, defender_bonus=2.0, nonlinear=1.5))


@pytest.fixture(scope="module")
def null():
    return _run(Planted(tackle_trend=0.0, defender_bonus=0.0, nonlinear=0.0))


def test_interpretable_model_beats_both_baselines_when_signal_exists(planted, null):
    for res, _ in (planted, null):
        d = res["decisions"]
        assert d["m1_beats_b0"] and d["m1_beats_b1"] and not d["m1_equivalent_to_b1"]
        m = res["metrics"]
        assert m["M1"]["ndcg5"]["lo"] > m["B1"]["ndcg5"]["hi"] > m["B0"]["ndcg5"]["point"]


def test_complexity_is_justified_only_when_nonlinearity_is_planted(planted, null):
    assert planted[0]["decisions"]["m2_justified"]
    assert planted[0]["comparisons"]["M2_vs_M1"]["ndcg5"]["diff"]["lo"] > 0.01
    assert not null[0]["decisions"]["m2_justified"]
    c = null[0]["comparisons"]["M2_vs_M1"]["ndcg5"]
    assert c["diff"]["lo"] < 0.01  # "not enough to justify losing explainability"


def test_planted_trend_in_tackles_is_found_and_no_phantom_trends_without_it(planted, null):
    t = planted[1].loc["tackles"]
    assert t["any_pair_ci_nonoverlap"] and t["diff_lo"] > 0 and t["p_holm_first_last"] < 0.05
    assert t["last_coef"] > 1.4 * t["first_coef"]
    quiet = null[1]
    assert (quiet["p_holm_first_last"] >= 0.05).all()  # Holm keeps the family-wise error under control
    assert quiet.loc["tackles", "diff_lo"] < 0 < quiet.loc["tackles", "diff_hi"]
