"""Out-of-sample engine (2020 onward): 1-day and 10-day VaR / ES forecasts.

Every forecast for day t uses data up to t-1 only. GARCH and copulas are re-estimated every
REFIT_EVERY days on the trailing WINDOW; between refits the GARCH variance is rolled forward
daily with realised returns (no re-estimation), so forecasts react to volatility immediately.

1-day models (same FHS-GARCH margins, different dependence) + one model-free benchmark:
  HS    historical simulation of the portfolio, last 250 days (industry simple benchmark)
  FHS   empirical copula of daily residuals = joint filtered historical simulation (non-parametric)
  G     Gaussian copula on daily residuals              (ignores tail dependence)
  T     Student-t copula on daily residuals             (tail dependence, single horizon)
  WT    Student-t copula on the H1 wavelet band         (tail dependence at the 2–8 day scale)

10-day models (buy-and-hold, non-overlapping windows), isolating each simplification in turn:
  G_sqrt   Gaussian 1-day VaR/ES x sqrt(10)               (ignores horizon AND tails: desk shortcut)
  G_path   Gaussian copula, simulated 10-day GARCH paths    (ignores tails)
  T_path   t copula on daily data, simulated paths          (tails, but daily dependence aggregated)
  T_join   control: T_path margins re-joined by the *daily* t copula (same machinery as WC)
  WC       wavelet-copula: T_path margins, joined by a t copula fitted on the horizon-matched band
  E_join   control: T_path margins re-joined by the *daily* empirical copula
  WC_emp   as WC but with the band's empirical copula (keeps crash/rally asymmetry)
The WC - T_join gap isolates horizon-specific dependence; T_path - G_path isolates tail dependence;
G_sqrt - G_path isolates the square-root-of-time rule.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import (H_DAYS, HORIZON_BAND, HS_WINDOW, N_SIM, N_SIM_H, OOS_START, REFIT_EVERY,
                        SEED, TICKERS, WINDOW)
from src.copulas_fit import fit_empirical, fit_gaussian, fit_student
from src.margins import fit_all_margins, pseudo_obs, std_resid_matrix
from src.risk import join_with_copula, one_day, portfolio_bh, shocks, simulate_paths, var_es
from src.wavelets import decompose_trailing

MODELS_1D = ["HS", "FHS", "G", "T", "WT"]
MODELS_H = ["G_sqrt", "G_path", "T_path", "T_join", "WC", "E_join", "WC_emp"]
COPULA_1D = ("FHS", "G", "T", "WT")


def run_oos(rets: pd.DataFrame, seed: int = SEED, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rets = rets[TICKERS]
    simple_p = np.expm1(rets).mean(axis=1)  # daily-rebalanced equal-weight portfolio return
    dates = rets.index
    oos_locs = [i for i, d in enumerate(dates) if d >= pd.Timestamp(OOS_START) and i >= WINDOW]
    band_h = HORIZON_BAND[H_DAYS]

    rec1, rech, refits = [], [], []
    fits = cop = Zs = zmat = None
    t0 = time.time()
    for k, loc in enumerate(oos_locs):
        dt = dates[loc]
        if k % REFIT_EVERY == 0:
            win = rets.iloc[loc - WINDOW:loc]
            fd = fit_all_margins(win)
            fits = [fd[t] for t in TICKERS]
            zdf = std_resid_matrix(fd)[TICKERS]
            zmat = zdf.to_numpy()
            bands = decompose_trailing(zmat)
            cop = {
                "FHS": fit_empirical(pseudo_obs(zmat)),
                "WE": fit_empirical(pseudo_obs(bands[band_h])),
                "G": fit_gaussian(pseudo_obs(zmat)),
                "T": fit_student(pseudo_obs(zmat)),
                "WT": fit_student(pseudo_obs(bands["H1"])),
                "WH": fit_student(pseudo_obs(bands[band_h])),
            }
            # independent random streams per refit and model: adding a model never changes another's draws
            Zs = {m: shocks(cop[m], zmat, N_SIM, np.random.default_rng([seed, k, 1, j]))
                  for j, m in enumerate(COPULA_1D)}
            refits.append({"date": dt, **{f"nu_{m}": cop[m].params["nu"] for m in ("T", "WT", "WH")},
                           **{f"rho_{m}": cop[m].params["rho_avg"] for m in ("G", "T", "WT", "WH")},
                           **{f"lambda_{m}": cop[m].lambda_l for m in ("T", "WT", "WH")}})
            if verbose and len(refits) % 10 == 1:
                print(f"  refit {len(refits):3d} @ {dt.date()}  nu_T={cop['T'].params['nu']:.1f} "
                      f"nu_WT={cop['WT'].params['nu']:.1f}  ({time.time() - t0:.0f}s)", flush=True)

        # ---- 1-day forecasts for date dt (information to dt-1)
        row = {"date": dt, "r_p": float(simple_p.iloc[loc])}
        hs = simple_p.iloc[loc - HS_WINDOW:loc].to_numpy()
        for m, res in [("HS", var_es(hs))] + [(m, one_day(Zs[m], fits)) for m in COPULA_1D]:
            row.update({f"{key}_{m}": v for key, v in res.items()})
        rec1.append(row)

        # ---- 10-day forecasts on non-overlapping windows starting at dt
        if k % H_DAYS == 0 and loc + H_DAYS <= len(dates):
            realised = float(np.expm1(rets.iloc[loc:loc + H_DAYS].sum(axis=0).to_numpy()).mean())
            g1 = one_day(Zs["G"], fits)
            hrow = {"date": dt, "end": dates[loc + H_DAYS - 1], "r_h": realised}
            hrow.update({f"{key}_G_sqrt": v * np.sqrt(H_DAYS) for key, v in g1.items()})
            rs = [np.random.default_rng([seed, k, 2, j]) for j in range(6)]
            cum_g = simulate_paths(cop["G"], zmat, fits, H_DAYS, N_SIM_H, rs[0])
            cum_t = simulate_paths(cop["T"], zmat, fits, H_DAYS, N_SIM_H, rs[1])
            sims = {"G_path": cum_g, "T_path": cum_t,
                    "T_join": join_with_copula(cum_t, cop["T"], rs[2]),
                    "WC": join_with_copula(cum_t, cop["WH"], rs[3]),
                    "WC_emp": join_with_copula(cum_t, cop["WE"], rs[4]),
                    "E_join": join_with_copula(cum_t, cop["FHS"], rs[5])}
            for m, cum in sims.items():
                hrow.update({f"{key}_{m}": v for key, v in var_es(portfolio_bh(cum)).items()})
            rech.append(hrow)

        # ---- roll GARCH variance forward with today's realised returns
        r_today = rets.iloc[loc].to_numpy()
        for f, r in zip(fits, r_today):
            f.update(float(r))

    if verbose:
        print(f"  OOS done: {len(rec1)} days, {len(rech)} x {H_DAYS}-day windows, {len(refits)} refits "
              f"({time.time() - t0:.0f}s)")
    return (pd.DataFrame(rec1).set_index("date"), pd.DataFrame(rech).set_index("date"),
            pd.DataFrame(refits).set_index("date"))
