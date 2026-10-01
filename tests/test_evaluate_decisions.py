import numpy as np
import pandas as pd

from ceb import evaluate as E


def _scored(m1_noise, m2_noise, n_matches=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        votes = np.zeros(44)
        votes[rng.choice(44, 7, replace=False)] = rng.permutation([10, 7, 5, 4, 2, 1, 1])
        rows.append(pd.DataFrame({
            "match_id": f"m{m}", "votes": votes.astype(int),
            "b0": rng.integers(0, 8, 44).astype(float),
            "b1": votes * 0.3 + rng.normal(0, 3, 44),
            "m1": votes + rng.normal(0, m1_noise, 44),
            "m2": votes + rng.normal(0, m2_noise, 44),
        }))
    d = pd.concat(rows, ignore_index=True)
    d["p_m1"] = np.where(d["votes"] > 0, 0.6, 0.1)
    d["p_m2"] = np.where(d["votes"] > 0, 0.7, 0.05)
    return d


def test_decision_rules_both_directions(SP):
    # M2 much better than M1 -> justified
    res, long = E.evaluate_scores(_scored(m1_noise=6.0, m2_noise=0.5), SP, "t1")
    d = res["decisions"]
    assert d["m2_justified"] and d["m2_significantly_better"] and d["m1_beats_b0"]
    assert res["comparisons"]["M2_vs_M1"]["ndcg5"]["diff"]["lo"] > SP.inference.sesoi_ndcg5
    # identical quality -> not justified, not even significant
    same = _scored(m1_noise=2.0, m2_noise=2.0)
    same["m2"] = same["m1"]
    res2, _ = E.evaluate_scores(same, SP, "t2")
    assert not res2["decisions"]["m2_justified"] and not res2["decisions"]["m2_significantly_better"]
    assert res2["comparisons"]["M2_vs_M1"]["ndcg5"]["diff"]["point"] == 0.0
    assert set(long["model"]) == set(E.MODELS) and long["match_id"].nunique() == 120


def test_better_but_below_threshold_is_not_justified(SP):
    # tiny real gain: significant maybe, but the lower CI bound cannot clear delta
    d = _scored(m1_noise=2.0, m2_noise=1.9, n_matches=300, seed=4)
    res, _ = E.evaluate_scores(d, SP, "t3")
    c = res["comparisons"]["M2_vs_M1"]["ndcg5"]
    assert c["diff"]["lo"] < SP.inference.sesoi_ndcg5 or not res["decisions"]["m2_justified"]
    assert res["decisions"]["m2_justified"] == (c["p_holm"] < 0.05 and c["diff"]["lo"] > SP.inference.sesoi_ndcg5)


def test_m1_equivalence_flag(SP):
    d = _scored(m1_noise=2.0, m2_noise=2.0, n_matches=200, seed=2)
    d["b1"] = d["m1"] + np.random.default_rng(1).normal(0, 0.01, len(d))  # B1 ~ M1
    res, _ = E.evaluate_scores(d, SP, "t4")
    assert res["decisions"]["m1_equivalent_to_b1"] and not res["decisions"]["m1_beats_b1"]


def test_holm_is_applied_within_each_metric_family(SP):
    res, _ = E.evaluate_scores(_scored(3.0, 2.0), SP, "t5")
    for k in ("ndcg5", "top1", "recall5", "spearman"):
        ps = [res["comparisons"][c][k] for c in E.COMPARISONS]
        assert all(p["p_holm"] >= p["p_raw"] - 1e-12 for p in ps)
        assert max(p["p_holm"] for p in ps) <= 1.0
