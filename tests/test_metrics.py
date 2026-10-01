import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score

from ceb import metrics as M


def _rand_case(rng, n=44, ties=True):
    votes = np.zeros(n)
    top = rng.choice(n, size=7, replace=False)
    votes[top] = rng.permutation([10, 7, 5, 4, 2, 1, 1])
    scores = rng.integers(0, 6, n).astype(float) if ties else rng.normal(size=n)
    return votes, scores


@pytest.mark.parametrize("ties", [True, False])
def test_ndcg_matches_sklearn_tie_aware(ties):
    rng = np.random.default_rng(0)
    for _ in range(50):
        v, s = _rand_case(rng, ties=ties)
        mine = M.ndcg_at_k(v, s, 5)
        ref = ndcg_score(v[None], s[None], k=5, ignore_ties=False)
        assert mine == pytest.approx(ref, abs=1e-12)


def test_dcg_tie_aware_equals_mean_over_random_tie_breaks():
    gains = np.array([5, 0, 3, 0, 1, 0, 2], float)
    scores = np.array([2, 2, 2, 1, 1, 0, 0], float)
    k = 4
    disc = 1 / np.log2(np.arange(len(gains)) + 2)
    vals = []
    # all orderings consistent with the tie blocks
    for p1 in itertools.permutations([0, 1, 2]):
        for p2 in itertools.permutations([3, 4]):
            for p3 in itertools.permutations([5, 6]):
                order = list(p1) + list(p2) + list(p3)
                vals.append((gains[order][:k] * disc[:k]).sum())
    assert M.dcg_at_k(gains, scores, k) == pytest.approx(np.mean(vals))


def test_perfect_and_worst_rankings():
    v = np.array([10, 7, 5, 0, 0, 0, 0, 0], float)
    assert M.ndcg_at_k(v, v, 5) == pytest.approx(1.0)
    assert M.top1_hit(v, v) == 1.0
    assert M.recall_at_k(v, v, 5) == 1.0
    worst = -v
    assert M.ndcg_at_k(v, worst, 3) == 0.0
    assert M.top1_hit(v, worst) == 0.0


def test_constant_scores_give_chance_level_not_arbitrary_order():
    v = np.array([10, 0, 0, 0], float)
    s = np.zeros(4)
    # all four tied: expected gain at each position is 2.5; top-1 hit chance is 1/4
    assert M.top1_hit(v, s) == 0.25
    assert M.recall_at_k(v, s, 2) == pytest.approx(0.5)
    assert M.ndcg_at_k(v, s, 2) == pytest.approx(
        (2.5 * (1 + 1 / np.log2(3))) / (10 * 1.0)
    )


def test_recall_normalisation_and_spearman_undefined_cases():
    v = np.array([10, 9, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0], float)  # 4 receivers
    s = np.array([4, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0], float)
    assert M.recall_at_k(v, s, 5) == 1.0  # all 4 receivers inside top 5, denominator min(5, 4)
    assert np.isnan(M.spearman_receivers(np.array([10, 0, 0]), np.array([3, 2, 1]), 3))
    assert np.isnan(M.spearman_receivers(np.array([5, 5, 5, 0.0]), np.array([3, 2, 1, 0]), 3))
    assert M.spearman_receivers(v, s) == pytest.approx(1.0)


def test_per_match_metrics_and_group_order_invariance():
    rng = np.random.default_rng(1)
    parts = []
    for m in range(6):
        v, s = _rand_case(rng)
        parts.append(pd.DataFrame({"match_id": f"m{m}", "votes": v, "s": s}))
    df = pd.concat(parts, ignore_index=True)
    a = M.per_match_metrics(df, "s")
    b = M.per_match_metrics(df.sample(frac=1, random_state=3), "s")
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == M.METRICS and len(a) == 6


def test_ece_components_roundtrip_and_perfect_calibration():
    rng = np.random.default_rng(2)
    n = 4000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    df = pd.DataFrame({"match_id": np.repeat(np.arange(n // 40), 40), "votes": y, "p": p})
    ids, cnt, sp, sy = M.ece_components(df, "p")
    assert cnt.sum() == n and len(ids) == n // 40
    assert M.ece_from_components(cnt, sp, sy) < 0.05
    df["p"] = np.where(y > 0, 0.05, 0.95) * 0 + 0.05  # badly calibrated: always 5%
    _, cnt, sp, sy = M.ece_components(df, "p")
    assert M.ece_from_components(cnt, sp, sy) > 0.3
