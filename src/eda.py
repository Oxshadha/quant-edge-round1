"""Exploratory data analysis and preprocessing diagnostics.

Each table here backs a modelling decision:
  data_quality       -> the price file is clean enough to use as is
  return_stats       -> fat tails (Jarque-Bera) and volatility clustering (Ljung-Box on r^2, ARCH-LM): use GARCH-t
  co_extremes        -> sectors hit extremes together far more often than independence implies
  rolling_avg_corr   -> correlation rises when volatility rises: dependence is not constant
  garch_diagnostics  -> after GARCH filtering the residuals show no remaining clustering: filter before the copula
  wavelet_energy     -> how much of each sector's variance sits at each scale (MODWT wavelet variance)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import BANDS, J_LEVELS, WAVELET
from src.modwt import modwt_mra


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value (chi2 with `lags` d.f.)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    denom = x @ x
    rho = np.array([(x[k:] @ x[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def arch_lm(x: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """Engle's ARCH-LM test: regress e_t^2 on its lags; LM = n R^2 ~ chi2(lags)."""
    e2 = (np.asarray(x, dtype=float) - np.mean(x)) ** 2
    y = e2[lags:]
    X = np.column_stack([np.ones(len(y))] + [e2[lags - k:-k] for k in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    lm = len(y) * r2
    return float(lm), float(stats.chi2.sf(lm, lags))


def data_quality(raw: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker checks on the raw price file and the cleaned panel."""
    rows = []
    r = np.log(prices / prices.shift(1)).dropna()
    for c in prices.columns:
        s = raw[c]
        rows.append({
            "ticker": c,
            "first_date": str(s.first_valid_index().date()),
            "missing_in_raw_file": int(s.isna().sum()),
            "non_positive_prices": int((s <= 0).sum()),
            "zero_return_days": int((r[c] == 0).sum()),
            "largest_abs_daily_move_pct": float(100 * r[c].abs().max()),
            "date_of_largest_move": str(r[c].abs().idxmax().date()),
        })
    out = pd.DataFrame(rows).set_index("ticker")
    out.attrs["duplicate_dates"] = int(raw.index.duplicated().sum())
    out.attrs["rows_dropped_by_alignment"] = int(len(raw) - len(prices))
    return out


def return_stats(rets: pd.DataFrame) -> pd.DataFrame:
    """Annualised moments plus normality and volatility-clustering tests for each series."""
    rows = []
    for c in rets.columns:
        x = rets[c].to_numpy()
        jb = stats.jarque_bera(x)
        rows.append({
            "series": c,
            "ann_mean_pct": 100 * 252 * x.mean(),
            "ann_vol_pct": 100 * np.sqrt(252) * x.std(ddof=1),
            "skew": float(stats.skew(x)),
            "excess_kurtosis": float(stats.kurtosis(x)),
            "worst_day_pct": 100 * x.min(),
            "jarque_bera_p": float(jb.pvalue),
            "ljung_box_r_p": ljung_box(x)[1],
            "ljung_box_r2_p": ljung_box(x**2)[1],
            "arch_lm_p": arch_lm(x)[1],
        })
    return pd.DataFrame(rows).set_index("series")


def co_extremes(rets: pd.DataFrame, qs=(0.01, 0.05)) -> pd.DataFrame:
    """On days when one sector is in its worst (best) q-tail, how many of the other sectors are too?
    Under independence the answer would be (d-1) * q."""
    rows = []
    d = rets.shape[1]
    for q in qs:
        for side, m in (("crash", rets <= rets.quantile(q)), ("rally", rets >= rets.quantile(1 - q))):
            M = m.astype(int).to_numpy()
            co = M.T @ M
            diag = np.diag(co)
            rows.append({"q": q, "side": side, "avg_other_sectors_also_extreme": float(((co.sum(1) - diag) / diag).mean()),
                         "if_independent": (d - 1) * q, "days_all_sectors_extreme": int((M.sum(1) == d).sum())})
    return pd.DataFrame(rows)


def rolling_avg_corr(rets: pd.DataFrame, window: int = 250) -> pd.Series:
    """Average pairwise correlation over a trailing window."""
    d = rets.shape[1]
    iu = np.triu_indices(d, 1)
    x = rets.to_numpy()
    out = np.full(len(x), np.nan)
    for t in range(window, len(x) + 1):
        out[t - 1] = np.corrcoef(x[t - window:t], rowvar=False)[iu].mean()
    return pd.Series(out, index=rets.index, name="avg_corr")


def acf(x: np.ndarray, lags: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float) - np.mean(x)
    return np.array([(x[k:] @ x[:-k]) / (x @ x) for k in range(1, lags + 1)])


def garch_diagnostics(fits: dict) -> pd.DataFrame:
    """Fitted GARCH(1,1)-t parameters and residual tests after filtering."""
    rows = []
    for c, f in fits.items():
        z = f.std_resid.to_numpy()
        pers = f.alpha + f.beta
        rows.append({
            "ticker": c, "omega_x1e6": 1e6 * f.omega, "alpha": f.alpha, "beta": f.beta, "persistence": pers,
            "half_life_days": float(np.log(0.5) / np.log(pers)) if pers < 1 else np.inf, "nu": f.nu,
            "ljung_box_z_p": ljung_box(z)[1], "ljung_box_z2_p": ljung_box(z**2)[1], "arch_lm_z_p": arch_lm(z)[1],
            "z_excess_kurtosis": float(stats.kurtosis(z)),
        })
    return pd.DataFrame(rows).set_index("ticker")


def wavelet_energy(z: pd.DataFrame) -> pd.DataFrame:
    """Share of each series' variance in each band (MODWT-MRA, LA(8), J = 6)."""
    rows = []
    for c in z.columns:
        details, smooth = modwt_mra(z[c].to_numpy(), WAVELET, J_LEVELS)
        comps = {b: np.sum(details[lo - 1:hi], axis=0) + (smooth if hi == J_LEVELS else 0)
                 for b, (lo, hi) in BANDS.items()}
        tot = sum(np.var(v) for v in comps.values())
        rows.append({"ticker": c, **{b: float(np.var(v) / tot) for b, v in comps.items()}})
    return pd.DataFrame(rows).set_index("ticker")
