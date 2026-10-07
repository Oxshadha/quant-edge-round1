"""Monte Carlo portfolio VaR / ES at 1 day and h days.

Margins: GARCH(1,1) volatility x empirical standardized shocks (filtered historical simulation,
Barone-Adesi et al. 1999). Dependence: a copula on the shocks. Portfolio: equal weight;
1-day = daily-rebalanced, h-day = buy-and-hold from the forecast date.
Losses are positive numbers: VaR_a = -q_a(R_p), ES_a = -E[R_p | R_p <= q_a].
"""

from __future__ import annotations

import numpy as np

from src.config import ES_LEVELS, VAR_LEVELS
from src.copulas_fit import CopulaResult, simulate_uniforms
from src.margins import MarginFit, empirical_quantile


def var_es(rp: np.ndarray) -> dict[str, float]:
    out = {}
    for a in sorted(set(VAR_LEVELS) | set(ES_LEVELS)):
        q = float(np.quantile(rp, a))
        tag = f"{round(100 * (1 - a), 1):g}".replace(".", "_")
        if a in VAR_LEVELS:
            out[f"var{tag}"] = -q
        if a in ES_LEVELS:
            out[f"es{tag}"] = -float(rp[rp <= q].mean())
            out[f"var{tag}"] = -q  # VaR at the ES level is needed for joint (VaR, ES) scoring
    return out


def shocks(c: CopulaResult, z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n x d standardized shocks: copula uniforms mapped through each asset's empirical z."""
    u = simulate_uniforms(c, n, rng)
    return np.column_stack([empirical_quantile(z[:, j], u[:, j]) for j in range(z.shape[1])])


def one_day(Z: np.ndarray, fits: list[MarginFit]) -> dict[str, float]:
    mu = np.array([f.mu for f in fits])
    sd = np.sqrt([f.sigma2_next for f in fits])
    r = mu + Z * sd
    return var_es(np.expm1(r).mean(axis=1))


def simulate_paths(c: CopulaResult, z: np.ndarray, fits: list[MarginFit], h: int, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Simulate h daily steps with the GARCH recursion; return n x d cumulative log returns."""
    mu = np.array([f.mu for f in fits])
    om = np.array([f.omega for f in fits])
    al = np.array([f.alpha for f in fits])
    be = np.array([f.beta for f in fits])
    s2 = np.tile([f.sigma2_next for f in fits], (n, 1))
    cum = np.zeros((n, len(fits)))
    for _ in range(h):
        eps = shocks(c, z, n, rng) * np.sqrt(s2)
        cum += mu + eps
        s2 = om + al * eps**2 + be * s2
    return cum


def join_with_copula(marg: np.ndarray, c: CopulaResult, rng: np.random.Generator) -> np.ndarray:
    """Keep each asset's simulated h-day marginal, impose the dependence of copula c (rank reordering)."""
    n, d = marg.shape
    u = simulate_uniforms(c, n, rng)
    ranks = np.argsort(np.argsort(u, axis=0), axis=0)
    srt = np.sort(marg, axis=0)
    return np.take_along_axis(srt, ranks, axis=0)


def portfolio_bh(cum_log: np.ndarray) -> np.ndarray:
    """Equal-weight buy-and-hold portfolio simple return from per-asset cumulative log returns."""
    return np.expm1(cum_log).mean(axis=1)
