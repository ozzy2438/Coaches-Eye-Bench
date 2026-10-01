import numpy as np
import pytest
from scipy.optimize import check_grad
from statsmodels.miscmodels.ordinal_model import OrderedModel

from ceb.models.m1 import OrdinalLogit, bands_to_codes


def _data(n=4000, p=5, K=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([1.0, -0.6, 0.3, 0.0, 0.8])[:p]
    theta = np.array([0.5, 1.8, 3.0])[: K - 1]
    eta = X @ beta
    u = rng.logistic(size=n)
    latent = eta + u
    y = np.searchsorted(theta, latent)
    return X, y, beta


def test_gradient_matches_finite_differences():
    X, y, _ = _data(n=300)
    m = OrdinalLogit(4, alpha=0.05)
    w = np.random.default_rng(1).integers(1, 4, len(y)).astype(float)
    x0 = np.r_[np.random.default_rng(2).normal(0, .3, X.shape[1]), [0.2, -0.5, -0.3]]
    f = lambda p: m._nll_grad(p, X, y, w)[0]  # noqa: E731
    g = lambda p: m._nll_grad(p, X, y, w)[1]  # noqa: E731
    assert check_grad(f, g, x0) < 1e-5


def test_unpenalised_fit_matches_statsmodels_orderedmodel():
    X, y, _ = _data()
    mine = OrdinalLogit(4, alpha=0.0).fit(X, y)
    sm = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False, maxiter=500)
    assert mine.converged_
    np.testing.assert_allclose(mine.coef_, sm.params[: X.shape[1]], atol=2e-3)
    sm_theta = np.cumsum(np.r_[sm.params[X.shape[1]], np.exp(sm.params[X.shape[1] + 1 :])])
    np.testing.assert_allclose(mine.theta_, sm_theta, atol=5e-3)


def test_recovers_truth_and_penalty_shrinks():
    X, y, beta = _data(n=20000)
    free = OrdinalLogit(4, alpha=0.0).fit(X, y)
    np.testing.assert_allclose(free.coef_, beta, atol=0.06)
    shrunk = OrdinalLogit(4, alpha=0.5).fit(X, y)
    assert np.linalg.norm(shrunk.coef_) < np.linalg.norm(free.coef_)


def test_weights_equal_row_duplication_and_probabilities_normalise():
    X, y, _ = _data(n=500)
    w = np.random.default_rng(3).integers(1, 4, len(y))
    a = OrdinalLogit(4, 0.01).fit(X, y, sample_weight=w.astype(float))
    b = OrdinalLogit(4, 0.01).fit(np.repeat(X, w, 0), np.repeat(y, w))
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-4)
    P = a.proba(X)
    np.testing.assert_allclose(P.sum(1), 1.0)
    assert (P >= 0).all()


def test_expected_vote_ranking_equals_eta_ranking():
    X, y, _ = _data(n=800)
    m = OrdinalLogit(4, 0.01).fit(X, y)
    ev = m.proba(X) @ np.array([0.0, 2.0, 5.0, 8.5])
    # monotone in eta (proportional odds): rank correlation exactly 1
    from scipy.stats import spearmanr
    assert spearmanr(ev, m.eta(X)).statistic == pytest.approx(1.0)


def test_bands_to_codes():
    bands = [[0, 0], [1, 3], [4, 6], [7, 10]]
    np.testing.assert_array_equal(bands_to_codes(np.array([0, 1, 3, 4, 6, 7, 10]), bands),
                                  [0, 1, 1, 2, 2, 3, 3])
    with pytest.raises(ValueError):
        bands_to_codes(np.array([11]), bands)
