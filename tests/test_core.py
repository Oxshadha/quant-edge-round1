"""Unit tests for the statistical building blocks.  Run: python -m pytest -q"""

import numpy as np
import pytest
from scipy import stats

from src.backtest import christoffersen_ind, diebold_mariano, fz0_loss, kupiec_uc, traffic_light
from src.copulas_fit import (CopulaResult, clayton_lambda_l, fit_all_families, fit_student, simulate_uniforms,
                             t_tail_dep)
from src.margins import empirical_quantile, pseudo_obs
from src.modwt import modwt_mra
from src.risk import join_with_copula, var_es
from src.tail_dependence import chi_avg, gaussian_chi

RNG = np.random.default_rng(123)
D = 5
R = np.full((D, D), 0.5)
np.fill_diagonal(R, 1.0)


def _sample(family, n=3000):
    params = {"gaussian": {"R": R}, "student": {"R": R, "nu": 4.0}, "clayton": {"theta": 2.0, "dim": D}}[family]
    return pseudo_obs(simulate_uniforms(CopulaResult(family, 0.0, 0, params), n, RNG))


@pytest.mark.parametrize("family", ["gaussian", "student", "clayton"])
def test_aic_recovers_true_family(family):
    """Regression test for the original bug: Clayton must NOT win on Gaussian (or t) data."""
    fits = fit_all_families(_sample(family))
    assert min(fits, key=lambda c: c.aic).family == family


def test_t_copula_recovers_nu():
    fit = fit_student(_sample("student", 6000))
    assert 3.0 < fit.params["nu"] < 5.5


def test_tail_dependence_formulas():
    assert clayton_lambda_l(1.0) == pytest.approx(0.5)
    assert t_tail_dep(0.0, 1.0) == pytest.approx(2 * stats.t.cdf(-np.sqrt(2), 2))
    assert t_tail_dep(0.5, 1e6) < 1e-6  # t -> Gaussian: no tail dependence


def test_modwt_mra_is_additive():
    x = RNG.standard_normal(777).cumsum()
    details, smooth = modwt_mra(x, "sym4", 6)
    assert np.allclose(np.sum(details, axis=0) + smooth, x, atol=1e-9)


def test_gaussian_chi_matches_simulation():
    z = RNG.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=400_000)
    emp, _ = chi_avg(z, 0.05)
    assert emp == pytest.approx(gaussian_chi(0.6, 0.05), abs=0.01)


def test_kupiec_reference_values():
    hits = np.zeros(1000)
    hits[:10] = 1
    assert kupiec_uc(hits, 0.01)[1] == pytest.approx(1.0)
    hits[:25] = 1
    assert kupiec_uc(hits, 0.01)[1] < 0.001


def test_christoffersen_detects_clustering():
    h = np.zeros(1000)
    h[100:120] = 1
    assert christoffersen_ind(h)[1] < 0.001


def test_traffic_light_zones():
    h = np.zeros(300, dtype=int)
    h[:10] = 1
    tl = traffic_light(h)
    assert tl["max_count"] == 10 and tl["red_share"] > 0


def test_var_es_on_normal():
    x = RNG.standard_normal(1_000_000)
    out = var_es(x)
    assert out["var99"] == pytest.approx(stats.norm.ppf(0.99), abs=0.02)
    assert out["es97_5"] == pytest.approx(stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025, abs=0.02)


def test_fz0_prefers_true_quantiles():
    y = RNG.standard_normal(200_000)
    a = 0.025
    v, e = stats.norm.ppf(1 - a), stats.norm.pdf(stats.norm.ppf(a)) / a
    true = fz0_loss(y, np.full_like(y, v), np.full_like(y, e), a).mean()
    wrong = fz0_loss(y, np.full_like(y, 0.8 * v), np.full_like(y, 0.8 * e), a).mean()
    assert true < wrong


def test_dm_detects_better_model():
    l = RNG.standard_normal(500)
    better = diebold_mariano(l - 0.5 + 0.1 * RNG.standard_normal(500), l)
    assert better["dm"] < 0 and better["p"] < 0.001


def test_join_keeps_margins_and_imposes_dependence():
    marg = RNG.standard_normal((20_000, D)) * np.arange(1, D + 1)
    joined = join_with_copula(marg, CopulaResult("gaussian", 0, 0, {"R": R}), RNG)
    assert np.allclose(np.sort(joined, 0), np.sort(marg, 0))
    tau = stats.kendalltau(joined[:, 0], joined[:, 1])[0]
    assert tau == pytest.approx(2 / np.pi * np.arcsin(0.5), abs=0.03)


def test_empirical_quantile_monotone():
    z = RNG.standard_normal(1000)
    q = empirical_quantile(z, np.linspace(0.01, 0.99, 50))
    assert np.all(np.diff(q) >= 0)


def test_ljung_box_and_arch_lm():
    from src.eda import arch_lm, ljung_box
    e = RNG.standard_normal(3000)
    assert ljung_box(e)[1] > 0.01 and arch_lm(e)[1] > 0.01  # white noise: no rejection
    s = np.empty(3000)
    s[0] = 1.0
    x = np.empty(3000)
    for t in range(3000):  # GARCH(1,1) process: clustering must be detected
        s[t] = 0.05 + 0.15 * (x[t - 1] ** 2 if t else 0) + 0.8 * (s[t - 1] if t else 1.0)
        x[t] = np.sqrt(s[t]) * RNG.standard_normal()
    assert ljung_box(x**2)[1] < 0.001 and arch_lm(x)[1] < 0.001
