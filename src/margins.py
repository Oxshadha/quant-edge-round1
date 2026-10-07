"""GARCH(1,1)-t margins, daily volatility updating, and filtered-historical-simulation shocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass
class MarginFit:
    ticker: str
    mu: float          # daily mean (decimal)
    omega: float       # GARCH params in decimal^2 units
    alpha: float
    beta: float
    nu: float
    std_resid: pd.Series
    sigma2_next: float  # one-step-ahead variance for the day after the window
    last_eps: float     # last residual r_T - mu (needed to roll the recursion forward)
    last_sigma2: float

    def update(self, r: float) -> None:
        """Roll the variance forecast forward by one observed return (no re-estimation)."""
        eps = r - self.mu
        self.last_sigma2 = self.sigma2_next
        self.last_eps = eps
        self.sigma2_next = self.omega + self.alpha * eps**2 + self.beta * self.sigma2_next


def fit_garch_t(returns: pd.Series, ticker: str = "") -> MarginFit:
    """Fit GARCH(1,1) with Student-t innovations by QML in arch (returns scaled to %)."""
    y = returns.dropna() * 100.0
    res = arch_model(y, mean="Constant", vol="Garch", p=1, q=1, dist="t", rescale=False).fit(
        disp="off", show_warning=False
    )
    p = res.params
    mu, omega, alpha, beta, nu = (float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]),
                                  float(p["beta[1]"]), float(p["nu"]))
    sig = np.asarray(res.conditional_volatility)
    eps = np.asarray(y) - mu
    z = pd.Series(eps / sig, index=y.index)
    sigma2_next = omega + alpha * eps[-1] ** 2 + beta * sig[-1] ** 2
    s2 = 1e-4  # %^2 -> decimal^2
    return MarginFit(ticker, mu / 100.0, omega * s2, alpha, beta, nu, z,
                     sigma2_next * s2, eps[-1] / 100.0, sig[-1] ** 2 * s2)


def fit_all_margins(returns: pd.DataFrame) -> dict[str, MarginFit]:
    return {c: fit_garch_t(returns[c], c) for c in returns.columns}


def std_resid_matrix(fits: dict[str, MarginFit]) -> pd.DataFrame:
    return pd.DataFrame({k: v.std_resid for k, v in fits.items()}).dropna(how="any")


def pseudo_obs(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Rank-based pseudo-observations u = rank/(n+1) (canonical maximum likelihood)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    return (np.argsort(np.argsort(x, axis=0), axis=0) + 1.0) / (n + 1.0)


def empirical_quantile(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniforms to standardized shocks through the empirical distribution of z (FHS)."""
    zs = np.sort(np.asarray(z, dtype=float))
    n = len(zs)
    grid = (np.arange(1, n + 1)) / (n + 1.0)
    return np.interp(u, grid, zs)
