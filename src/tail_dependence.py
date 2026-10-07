"""Model-free tail co-movement, Gaussian benchmark, and stationary-bootstrap inference.

chi_L(q) = P(U_j <= q | U_i <= q) = C(q, q) / q   (lower tail, "co-crash" probability)
chi_U(q) = P(U_j >  1-q | U_i > 1-q)               (upper tail, "co-rally" probability)
As q -> 0, chi_L(q) -> lambda_L. At finite q a Gaussian copula still gives chi > 0, so we always
report the Gaussian-implied value at the same rho as the benchmark (Frahm, Junker & Schmidt 2005;
Schmidt & Stadtmueller 2006).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.margins import pseudo_obs


def chi_pairs(u: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise (i<j) empirical chi_L(q), chi_U(q)."""
    lo = (u <= q).astype(float)
    hi = (u > 1.0 - q).astype(float)
    n = u.shape[0]
    cl = (lo.T @ lo) / n / q
    cu = (hi.T @ hi) / n / q
    iu = np.triu_indices(u.shape[1], 1)
    return cl[iu], cu[iu]


def chi_avg(x: np.ndarray, q: float) -> tuple[float, float]:
    cl, cu = chi_pairs(pseudo_obs(x), q)
    return float(cl.mean()), float(cu.mean())


def gaussian_chi(rho: float, q: float) -> float:
    """C_Gauss(q,q;rho)/q — the co-crash probability a Gaussian copula implies at level q."""
    z = stats.norm.ppf(q)
    cdf = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([z, z])
    return float(cdf / q)


def gaussian_chi_avg(x: np.ndarray, q: float) -> float:
    """Average Gaussian-implied chi over pairs, using each pair's normal-score correlation."""
    ns = stats.norm.ppf(pseudo_obs(x))
    R = np.corrcoef(ns, rowvar=False)
    iu = np.triu_indices(R.shape[0], 1)
    return float(np.mean([gaussian_chi(r, q) for r in R[iu]]))


def stationary_bootstrap_idx(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap indices (circular)."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    i = rng.integers(n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            i = rng.integers(n)
        idx[t] = i
        i = (i + 1) % n
    return idx


def boot_summary(point: float, draws: np.ndarray) -> dict:
    """Normal-approximation CI (est +/- 1.96 se, se from the bootstrap) and two-sided p-value for H0: value = 0.

    The normal interval is used instead of percentiles because block resampling biases chi slightly
    downward (it breaks dependence at block joins); centring on the point estimate avoids that bias.
    """
    draws = np.asarray(draws, dtype=float)
    se = float(draws.std(ddof=1))
    return {"est": float(point), "lo": float(point - 1.96 * se), "hi": float(point + 1.96 * se),
            "se": se, "p0": float(2.0 * stats.norm.sf(abs(point) / se)) if se > 0 else float("nan")}
