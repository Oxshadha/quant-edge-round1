"""Elliptical and Archimedean copulas with *full* log-likelihoods, so AIC comparisons are valid.

Families
- Gaussian: R from normal scores; no tail dependence.
- Student-t: R from Kendall's tau (R_ij = sin(pi*tau_ij/2), Lindskog et al. 2003), nu by profile ML;
  symmetric tail dependence lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))).
- Clayton (exchangeable, d-dimensional): lower-tail dependence lambda_L = 2^(-1/theta), no upper tail.
- Empirical: resampled rank vectors (used for forecasting only; no likelihood).
All are fitted to rank pseudo-observations (canonical ML, Genest, Ghoudi & Rivest 1995).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

EPS = 1e-10


@dataclass
class CopulaResult:
    family: str
    loglik: float
    n_params: int
    params: dict = field(default_factory=dict)
    lambda_l: float = 0.0
    lambda_u: float = 0.0

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.n_params


def _nearest_corr(c: np.ndarray) -> np.ndarray:
    c = (c + c.T) / 2.0
    w, v = np.linalg.eigh(c)
    c = (v * np.clip(w, 1e-6, None)) @ v.T
    s = np.sqrt(np.diag(c))
    c = c / np.outer(s, s)
    np.fill_diagonal(c, 1.0)
    return c


def _offdiag_mean(m: np.ndarray) -> float:
    return float(m[~np.eye(m.shape[0], dtype=bool)].mean())


def kendall_matrix(u: np.ndarray) -> np.ndarray:
    d = u.shape[1]
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = tau[j, i] = stats.kendalltau(u[:, i], u[:, j])[0]
    return tau


def t_tail_dep(rho, nu):
    rho = np.asarray(rho, dtype=float)
    x = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return 2.0 * stats.t.cdf(x, df=nu + 1.0)


def clayton_lambda_l(theta: float) -> float:
    return float(2.0 ** (-1.0 / theta)) if theta > 0 else 0.0


# ---------------------------------------------------------------- Gaussian
def gaussian_loglik(u: np.ndarray, R: np.ndarray) -> float:
    x = stats.norm.ppf(np.clip(u, EPS, 1 - EPS))
    return float(stats.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R).logpdf(x).sum()
                 - stats.norm.logpdf(x).sum())


def fit_gaussian(u: np.ndarray) -> CopulaResult:
    d = u.shape[1]
    x = stats.norm.ppf(np.clip(u, EPS, 1 - EPS))
    R = _nearest_corr(np.corrcoef(x, rowvar=False))
    return CopulaResult("gaussian", gaussian_loglik(u, R), d * (d - 1) // 2,
                        {"R": R, "rho_avg": _offdiag_mean(R)})


# ---------------------------------------------------------------- Student-t
def t_loglik(u: np.ndarray, R: np.ndarray, nu: float) -> float:
    x = stats.t.ppf(np.clip(u, EPS, 1 - EPS), df=nu)
    return float(stats.multivariate_t(loc=np.zeros(R.shape[0]), shape=R, df=nu).logpdf(x).sum()
                 - stats.t.logpdf(x, df=nu).sum())


def fit_student(u: np.ndarray, R: np.ndarray | None = None) -> CopulaResult:
    d = u.shape[1]
    if R is None:
        R = _nearest_corr(np.sin(np.pi * kendall_matrix(u) / 2.0))
    res = optimize.minimize_scalar(lambda lnu: -t_loglik(u, R, float(np.exp(lnu))),
                                   bounds=(np.log(2.1), np.log(200.0)), method="bounded",
                                   options={"xatol": 1e-3})
    nu = float(np.exp(res.x))
    lam = float(np.mean(t_tail_dep(R[~np.eye(d, dtype=bool)], nu)))
    return CopulaResult("student", -float(res.fun), d * (d - 1) // 2 + 1,
                        {"R": R, "nu": nu, "rho_avg": _offdiag_mean(R)}, lam, lam)


# ---------------------------------------------------------------- Clayton
def clayton_loglik(u: np.ndarray, theta: float) -> float:
    u = np.clip(u, EPS, 1 - EPS)
    d = u.shape[1]
    s = np.sum(u ** (-theta), axis=1) - d + 1.0
    return float(np.sum(np.log1p(theta * np.arange(d)))* len(u)
                 - (1.0 + theta) * np.log(u).sum()
                 - (d + 1.0 / theta) * np.log(s).sum())


def fit_clayton(u: np.ndarray) -> CopulaResult:
    res = optimize.minimize_scalar(lambda lt: -clayton_loglik(u, float(np.exp(lt))),
                                   bounds=(np.log(1e-3), np.log(30.0)), method="bounded")
    theta = float(np.exp(res.x))
    return CopulaResult("clayton", -float(res.fun), 1, {"theta": theta, "dim": u.shape[1]},
                        clayton_lambda_l(theta), 0.0)


def fit_empirical(u: np.ndarray) -> CopulaResult:
    """Empirical copula: resample observed rank vectors (non-parametric, keeps any tail asymmetry)."""
    return CopulaResult("empirical", float("nan"), 0, {"U": np.asarray(u, dtype=float)})


FITTERS = {"gaussian": fit_gaussian, "student": fit_student, "clayton": fit_clayton}


def fit_all_families(u: np.ndarray) -> list[CopulaResult]:
    return [f(np.asarray(u, dtype=float)) for f in FITTERS.values()]


def select_copula(u: np.ndarray) -> CopulaResult:
    return min(fit_all_families(u), key=lambda c: c.aic)


# ---------------------------------------------------------------- simulation
def simulate_uniforms(c: CopulaResult, n: int, rng: np.random.Generator) -> np.ndarray:
    if c.family == "gaussian":
        L = np.linalg.cholesky(c.params["R"])
        return stats.norm.cdf(rng.standard_normal((n, L.shape[0])) @ L.T)
    if c.family == "student":
        L = np.linalg.cholesky(c.params["R"])
        nu = c.params["nu"]
        g = rng.standard_normal((n, L.shape[0])) @ L.T
        w = np.sqrt(nu / rng.chisquare(nu, size=n))[:, None]
        return stats.t.cdf(g * w, df=nu)
    if c.family == "clayton":  # Marshall–Olkin
        theta = c.params["theta"]
        d = c.params.get("dim")
        v = rng.gamma(1.0 / theta, 1.0, size=n)
        e = rng.exponential(1.0, size=(n, d))
        return (1.0 + e / v[:, None]) ** (-1.0 / theta)
    if c.family == "empirical":
        U = c.params["U"]
        return U[rng.integers(len(U), size=n)]
    raise ValueError(c.family)
