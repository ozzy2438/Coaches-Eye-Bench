import numpy as np
import pytest

from ceb import bootstrap as B


def test_holm_known_values_and_monotone():
    assert B.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert B.holm([0.5]) == [0.5]
    adj = B.holm([0.2, 0.001, 0.9, 0.04])
    assert all(a >= p for a, p in zip(adj, [0.2, 0.001, 0.9, 0.04], strict=True))
    assert max(adj) <= 1.0


def test_seeding_is_deterministic_and_label_dependent():
    x = np.random.default_rng(0).normal(size=100)
    a = B.mean_ci(x, 500, 1, "a")
    assert a == B.mean_ci(x, 500, 1, "a")
    assert a != B.mean_ci(x, 500, 1, "b")


def test_mean_ci_covers_true_mean_and_drops_nan():
    rng = np.random.default_rng(3)
    cover = 0
    for i in range(200):
        x = rng.normal(0.7, 0.2, 80)
        ci = B.mean_ci(x, 400, 7, f"cov{i}")
        cover += ci.lo <= 0.7 <= ci.hi
    assert 0.90 <= cover / 200 <= 0.99
    y = np.r_[rng.normal(size=50), np.nan, np.nan]
    assert B.mean_ci(y, 100, 1, "n").n == 50


def test_paired_comparison_detects_real_and_null_differences():
    rng = np.random.default_rng(4)
    b = rng.normal(0.6, 0.2, 300)
    a_real = b + 0.05 + rng.normal(0, 0.05, 300)
    a_null = b + rng.normal(0, 0.05, 300)
    real = B.paired_comparison(a_real, b, 1000, 4000, 1, "real")
    null = B.paired_comparison(a_null, b, 1000, 4000, 1, "null")
    assert real.p_raw < 0.01 and real.diff.lo > 0
    assert null.diff.lo < 0 < null.diff.hi and null.p_raw > 0.01
    assert real.relative.point == pytest.approx(a_real.mean() / b.mean() - 1)


def test_sign_flip_p_is_uniform_under_null():
    rng = np.random.default_rng(5)
    ps = [B.sign_flip_p(rng.normal(size=60), 500, np.random.default_rng(i)) for i in range(150)]
    assert 0.35 < np.mean(ps) < 0.65
    assert np.mean(np.array(ps) < 0.05) < 0.12
