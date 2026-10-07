"""VaR and ES backtests, and comparative (scoring-function) tests.

VaR: Kupiec (1995) unconditional coverage, Christoffersen (1998) independence and conditional
coverage, Engle & Manganelli (2004) dynamic quantile, Basel (1996) traffic light.
ES: McNeil & Frey (2000) exceedance-residual test, Acerbi & Szekely (2014) Z2.
Comparison: FZ0 joint (VaR, ES) loss (Fissler & Ziegel 2016; Patton, Ziegel & Chen 2019) and
Diebold & Mariano (1995) with Newey–West HAC variance.
Convention: r = realised portfolio return, var/es = positive loss numbers; a hit is r < -var.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- VaR tests
def kupiec_uc(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    n, x = len(hits), int(np.sum(hits))
    ph = x / n
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll1 = (x * np.log(ph) if x else 0.0) + ((n - x) * np.log(1 - ph) if x < n else 0.0)
    lr = max(0.0, -2.0 * (ll0 - ll1))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    def xlogy(a, p):
        return a * np.log(p) if a > 0 else 0.0

    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    lr = max(0.0, -2.0 * (ll0 - ll1))
    return lr, float(stats.chi2.sf(lr, 1))


def dq_test(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int = 4) -> tuple[float, float]:
    """Engle–Manganelli DQ: regress (hit - alpha) on const, lagged hits and VaR; chi2(k) under H0."""
    hit = np.asarray(hits, dtype=float) - alpha
    T = len(hit)
    X = [np.ones(T - lags)]
    X += [hit[lags - k:T - k] for k in range(1, lags + 1)]
    X.append(np.asarray(var, dtype=float)[lags:])
    X = np.column_stack(X)
    y = hit[lags:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dq = float(beta @ X.T @ X @ beta / (alpha * (1 - alpha)))
    return dq, float(stats.chi2.sf(dq, X.shape[1]))


def traffic_light(hits99: np.ndarray, window: int = 250) -> dict:
    """Rolling 250-day count of 99% VaR exceptions and Basel zone shares (green<=4, yellow 5-9, red>=10)."""
    h = np.asarray(hits99, dtype=int)
    if len(h) < window:
        return {}
    c = np.convolve(h, np.ones(window, dtype=int), mode="valid")
    return {
        "green_share": float(np.mean(c <= 4)),
        "yellow_share": float(np.mean((c >= 5) & (c <= 9))),
        "red_share": float(np.mean(c >= 10)),
        "max_count": int(c.max()),
        "last_count": int(c[-1]),
    }


def var_backtest(r: np.ndarray, var: np.ndarray, alpha: float) -> dict:
    hits = (r < -var).astype(int)
    lr_uc, p_uc = kupiec_uc(hits, alpha)
    lr_ind, p_ind = christoffersen_ind(hits)
    lr_cc = lr_uc + lr_ind
    out = {
        "n": int(len(hits)), "hits": int(hits.sum()), "expected_hits": float(alpha * len(hits)),
        "hit_rate": float(hits.mean()), "alpha": alpha,
        "p_uc": p_uc, "p_ind": p_ind, "p_cc": float(stats.chi2.sf(lr_cc, 2)),
    }
    if len(hits) > 50:
        out["dq"], out["p_dq"] = dq_test(hits, var, alpha)
    return out


# ---------------------------------------------------------------- ES tests
def mcneil_frey(r: np.ndarray, var: np.ndarray, es: np.ndarray, n_boot: int = 5000,
                seed: int = 0) -> dict:
    """Exceedance residuals e = (loss - ES) on VaR-breach days; H0: E[e]=0 vs H1: E[e]>0 (ES too low).
    One-sided bootstrap of the centred mean (Efron & Tibshirani 1993, as in McNeil & Frey 2000)."""
    hit = r < -var
    e = (-r[hit] - es[hit]) / es[hit]  # scale-free: relative shortfall
    if len(e) < 3:
        return {"n_exc": int(len(e)), "mean_rel_excess": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    t_obs = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))
    ec = e - e.mean()
    bs = rng.choice(ec, size=(n_boot, len(e)), replace=True)
    tb = bs.mean(1) / (bs.std(1, ddof=1) / np.sqrt(len(e)) + 1e-12)
    return {"n_exc": int(len(e)), "mean_rel_excess": float(e.mean()), "p": float(np.mean(tb >= t_obs))}


def acerbi_szekely_z2(r: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """Z2 = sum(r_t * I_t / (T * alpha * ES_t)) + 1; E[Z2]=0 under H0, negative = ES underestimated."""
    hit = (r < -var).astype(float)
    return float(np.sum(r * hit / (len(r) * alpha * es)) + 1.0)


def z2_pvalue(z2: float, T: int, alpha: float, nu: float = 5.0, n_sim: int = 4000, seed: int = 1) -> float:
    """Left-tail p-value of Z2 under a correctly specified Student-t(nu) null (Acerbi & Szekely 2014
    show the null distribution is nearly invariant to the tail shape)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_t(nu, size=(n_sim, T))
    q = stats.t.ppf(alpha, nu)
    es = -stats.t.expect(lambda y: y, args=(nu,), ub=q) / alpha
    hit = x < q
    z = (x * hit).sum(1) / (T * alpha * es) + 1.0
    return float(np.mean(z <= z2))


def es_backtest(r: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> dict:
    z2 = acerbi_szekely_z2(r, var, es, alpha)
    hit = r < -var
    ratio = float((-r[hit]).mean() / es[hit].mean()) if hit.any() else float("nan")
    return {"z2": z2, "p_z2": z2_pvalue(z2, len(r), alpha), "realised_over_predicted": ratio,
            **{f"mf_{k}": v for k, v in mcneil_frey(r, var, es).items()}}


# ---------------------------------------------------------------- comparative
def fz0_loss(r: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """FZ0 loss of Patton, Ziegel & Chen (2019) in return space (v=-VaR, e=-ES < 0). Lower is better."""
    v, e = -np.asarray(var), -np.asarray(es)
    hit = (r <= v).astype(float)
    return -hit * (v - r) / (alpha * e) + v / e + np.log(-e) - 1.0


def diebold_mariano(l1: np.ndarray, l2: np.ndarray, lag: int | None = None) -> dict:
    """DM test of equal expected loss; negative statistic = model 1 better. Newey–West HAC variance."""
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    if lag is None:
        lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    dc = d - d.mean()
    s = dc @ dc / T
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(s / T)
    return {"mean_diff": float(d.mean()), "dm": float(stat), "p": float(2 * stats.norm.sf(abs(stat)))}
