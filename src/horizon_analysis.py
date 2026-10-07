"""Part (a): does tail dependence change with the investment horizon?  (estimation sample <= 2019)

Two complementary views, both with stationary-bootstrap confidence intervals:
1. Scale view:  MODWT-MRA bands H1/H2/H3 of GARCH-filtered residuals (volatility clustering removed).
2. Return view: non-overlapping 1/5/10/21-day returns, i.e. what a holder at that horizon experiences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AGG_HORIZONS, BAND_LABELS, BANDS, BOOT_BLOCK, N_BOOT, SEED, TAIL_Q
from src.copulas_fit import fit_all_families, fit_empirical, fit_gaussian, fit_student, simulate_uniforms
from src.margins import empirical_quantile, pseudo_obs
from src.tail_dependence import boot_summary, chi_avg, gaussian_chi_avg, stationary_bootstrap_idx
from src.wavelets import decompose


def copula_table(z: pd.DataFrame) -> pd.DataFrame:
    """Full-likelihood Gaussian / t / Clayton fits on daily residuals and on each band."""
    series = {"daily": np.asarray(z), **decompose(z)}
    rows = []
    for name, x in series.items():
        fits = fit_all_families(pseudo_obs(x))
        best = min(fits, key=lambda c: c.aic)
        for c in fits:
            rows.append({
                "series": name, "family": c.family, "n_obs": len(x), "loglik": c.loglik,
                "n_params": c.n_params, "aic": c.aic, "delta_aic": c.aic - best.aic,
                "aic_winner": c.family == best.family,
                "nu": c.params.get("nu", np.nan), "theta": c.params.get("theta", np.nan),
                "rho_avg": c.params.get("rho_avg", np.nan),
                "lambda_l": c.lambda_l, "lambda_u": c.lambda_u,
            })
    return pd.DataFrame(rows)


def _band_stats(z: np.ndarray, q: float) -> dict[str, float]:
    out = {}
    for name, x in {"daily": z, **decompose(z)}.items():
        cl, cu = chi_avg(x, q)
        out[f"{name}_L"], out[f"{name}_U"] = cl, cu
        out[f"{name}_G"] = gaussian_chi_avg(x, q)
    return out


def _agg(r: np.ndarray, h: int) -> np.ndarray:
    n = (len(r) // h) * h
    return r[:n].reshape(-1, h, r.shape[1]).sum(axis=1)


def _horizon_stats(r: np.ndarray, q: float) -> dict[str, float]:
    out = {}
    for h in AGG_HORIZONS:
        a = _agg(r, h)
        cl, cu = chi_avg(a, q)
        out[f"h{h}_L"], out[f"h{h}_U"] = cl, cu
        out[f"h{h}_G"] = gaussian_chi_avg(a, q)
    return out


def _bootstrap(stat_fn, data: np.ndarray, q: float, seed: int) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    point = stat_fn(data, q)
    draws = [stat_fn(data[stationary_bootstrap_idx(len(data), BOOT_BLOCK, rng)], q) for _ in range(N_BOOT)]
    return point, draws


def _summaries(point: dict, draws: list[dict], names: list[str], base: str) -> tuple[list, list]:
    """CI per (name, tail) and tests of (name - base) and of asymmetry (L - U)."""
    D = pd.DataFrame(draws)
    level_rows, test_rows = [], []
    for nm in names:
        for tail in ("L", "U"):
            k = f"{nm}_{tail}"
            level_rows.append({"series": nm, "tail": tail, **boot_summary(point[k], D[k].to_numpy()),
                               "gaussian_implied": point[f"{nm}_G"],
                               **{f"excess_{kk}": vv for kk, vv in boot_summary(
                                   point[k] - point[f"{nm}_G"], (D[k] - D[f"{nm}_G"]).to_numpy()).items()}})
        asym = point[f"{nm}_L"] - point[f"{nm}_U"]
        test_rows.append({"test": f"{nm}: chi_L - chi_U", **boot_summary(asym, (D[f"{nm}_L"] - D[f"{nm}_U"]).to_numpy())})
        if nm != base:
            for tail in ("L", "U"):
                k, kb = f"{nm}_{tail}", f"{base}_{tail}"
                test_rows.append({"test": f"chi_{tail}: {nm} - {base}",
                                  **boot_summary(point[k] - point[kb], (D[k] - D[kb]).to_numpy())})
    return level_rows, test_rows


def tail_by_band(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arr = np.asarray(z, dtype=float)
    levels, tests = [], []
    bands = decompose(arr)
    for i, q in enumerate(TAIL_Q):
        point, draws = _bootstrap(_band_stats, arr, q, SEED + i)
        lv, ts = _summaries(point, draws, ["daily", *BANDS], base="H1")
        for row in lv:
            row.update(q=q, n_obs=len(arr if row["series"] == "daily" else bands[row["series"]]),
                       scale=BAND_LABELS.get(row["series"], "1 day"))
        for row in ts:
            row["q"] = q
        levels += lv
        tests += ts
    return pd.DataFrame(levels), pd.DataFrame(tests)


def tail_by_horizon(r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arr = np.asarray(r, dtype=float)
    levels, tests = [], []
    for i, q in enumerate(TAIL_Q):
        point, draws = _bootstrap(_horizon_stats, arr, q, SEED + 100 + i)
        names = [f"h{h}" for h in AGG_HORIZONS]
        lv, ts = _summaries(point, draws, names, base="h1")
        for row in lv:
            h = int(row["series"][1:])
            agg = _agg(arr, h)
            row.update(q=q, horizon_days=h, n_obs=len(agg))
        for row in ts:
            row["q"] = q
        levels += lv
        tests += ts
    return pd.DataFrame(levels), pd.DataFrame(tests)


def model_implied_by_horizon(r: pd.DataFrame, fits: dict, n_paths: int = 60_000) -> pd.DataFrame:
    """Tail co-movement of h-day returns implied by *daily* models simulated forward (GARCH + copula),
    to set against the observed values: does a daily model reproduce how dependence behaves at horizon h?
    Paths start from variances drawn from the in-sample conditional variances (stationary start)."""
    cols = list(r.columns)
    fl = [fits[c] for c in cols]
    z = pd.DataFrame({c: fits[c].std_resid for c in cols}).dropna()
    Z = z.to_numpy()
    mu = np.array([f.mu for f in fl])
    om, al, be = (np.array([getattr(f, a) for f in fl]) for a in ("omega", "alpha", "beta"))
    sig2 = ((r.loc[z.index].to_numpy() - mu) / Z) ** 2
    U = pseudo_obs(Z)
    rows = []
    for j, (name, cop) in enumerate({"Gaussian": fit_gaussian(U), "Student-t": fit_student(U),
                                     "Empirical (FHS)": fit_empirical(U)}.items()):
        rng = np.random.default_rng([SEED, 500, j])
        s2 = sig2[rng.integers(len(sig2), size=n_paths)]
        cum = np.zeros((n_paths, len(cols)))
        for step in range(1, max(AGG_HORIZONS) + 1):
            u = simulate_uniforms(cop, n_paths, rng)
            e = np.column_stack([empirical_quantile(Z[:, k], u[:, k]) for k in range(len(cols))]) * np.sqrt(s2)
            cum += mu + e
            s2 = om + al * e**2 + be * s2
            if step in AGG_HORIZONS:
                for q in TAIL_Q:
                    cl, cu = chi_avg(cum, q)
                    rows.append({"model": name, "horizon_days": step, "q": q, "chi_L": cl, "chi_U": cu})
    return pd.DataFrame(rows)
