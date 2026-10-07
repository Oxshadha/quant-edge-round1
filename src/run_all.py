"""
One-command reproduction for SAIFA Quant Edge 1.0, Round 1 — "Risk Across Tails and Timescales".

    python -m src.run_all          (or: make reproduce)

Stages
  1. In-sample (1999–2019): GARCH-t filtering, MODWT-MRA bands, full-likelihood copula fits,
     tail co-movement by scale and by holding horizon with stationary-bootstrap inference.
  2. Out-of-sample (2020–): rolling 1-day VaR/ES for HS, Gaussian, t and wavelet-t copula models;
     10-day ES on non-overlapping windows for four horizon treatments.
  3. Backtests, figures, headline numbers (outputs/results.json) and the desk recommendation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import figures
from src.config import H_DAYS, IN_SAMPLE_END, OUTPUT_DIR, TICKERS
from src.data import download_prices, log_returns
from src.evaluate import summarize_1d, summarize_h
from src.horizon_analysis import copula_table, model_implied_by_horizon, tail_by_band, tail_by_horizon
from src.margins import fit_all_margins, std_resid_matrix
from src.backtest import fz0_loss
from src.oos import MODELS_1D, MODELS_H, run_oos

ROBUST_SEEDS = (7, 11, 2026)  # extra Monte Carlo seeds: are model differences larger than simulation noise?


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def _pick(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m].iloc[0]


def headline(prices, cop, band_lv, band_ts, hor_lv, hor_ts, s1, sh, d1, dh) -> dict:
    """Every number quoted in the report, in one place."""
    q = 0.05
    tails = {}
    for s in ["daily", "H1", "H2", "H3"]:
        for t in ("L", "U"):
            r = _pick(band_lv, series=s, tail=t, q=q)
            tails[f"{s}_{t}"] = {k: float(r[k]) for k in ["est", "lo", "hi", "gaussian_implied",
                                                          "excess_est", "excess_lo", "excess_hi", "excess_p0"]}
    tests = {row.test: {"est": row.est, "lo": row.lo, "hi": row.hi, "p": row.p0}
             for row in band_ts[band_ts.q == q].itertuples()}
    hq = 0.10
    htails = {}
    for h in (1, 5, 10, 21):
        for t in ("L", "U"):
            r = _pick(hor_lv, series=f"h{h}", tail=t, q=hq)
            htails[f"h{h}_{t}"] = {k: float(r[k]) for k in ["est", "lo", "hi", "gaussian_implied", "excess_est", "excess_p0"]}
    htests = {row.test: {"est": row.est, "lo": row.lo, "hi": row.hi, "p": row.p0}
              for row in hor_ts[hor_ts.q == hq].itertuples()}
    win = cop[cop.aic_winner][["series", "family", "nu", "lambda_l"]].set_index("series").to_dict("index")
    tcop = cop[cop.family == "student"].set_index("series")[["nu", "lambda_l", "rho_avg"]].to_dict("index")
    return {
        "data": {"tickers": TICKERS, "start": str(prices.index.min().date()), "end": str(prices.index.max().date()),
                 "n_days": int(len(prices))},
        "copula_aic_winner": win, "t_copula_by_series": tcop,
        "tail_band_q05": tails, "tail_band_tests_q05": tests,
        "tail_horizon_q10": htails, "tail_horizon_tests_q10": htests,
        "oos_1d": s1, "oos_10d": sh,
        "oos_window": {"start": str(d1.index.min().date()), "end": str(d1.index.max().date()), "days": int(len(d1)),
                       "h_windows": int(len(dh))},
        "latest": {
            "date": str(d1.index.max().date()),
            **{f"{k}_{m}": float(d1[f"{k}_{m}"].iloc[-1]) for m in ("HS", "G", "T", "WT") for k in ("var99", "es97_5")},
            **{f"es97_5_10d_{m}": float(dh[f"es97_5_{m}"].iloc[-1]) for m in ("G_sqrt", "G_path", "T_path", "WC")},
        },
    }


def seed_row(d1: pd.DataFrame, dh: pd.DataFrame, seed) -> dict:
    r, rh = d1.r_p.to_numpy(), dh.r_h.to_numpy()
    row = {"seed": seed}
    for m in MODELS_1D:
        row[f"hits99_{m}"] = int((r < -d1[f"var99_{m}"]).sum())
        row[f"es97_5_{m}"] = float(d1[f"es97_5_{m}"].mean())
        row[f"fz0_{m}"] = float(fz0_loss(r, d1[f"var97_5_{m}"], d1[f"es97_5_{m}"], 0.025).mean())
    for m in MODELS_H:
        row[f"es97_5_10d_{m}"] = float(dh[f"es97_5_{m}"].mean())
        row[f"gap_10d_{m}"] = float((dh[f"es97_5_{m}"] / dh["es97_5_WC"] - 1).mean())
    return row


def recommendation(H: dict) -> str:
    s1, sh = H["oos_1d"], H["oos_10d"]
    gap = sh["es_gap_vs_WC"]
    tl = {m: s1[m]["traffic_light"] for m in ("HS", "G", "T", "WT")}
    best = min(("HS", "G", "T", "WT"), key=lambda m: s1[m]["fz0_97_5"])
    lat = H["latest"]
    t = H["tail_band_q05"]
    txt = (
        "DESK RECOMMENDATION (equal-weight US sector ETF book: XLE XLF XLK XLV XLI XLU)\n\n"
        f"Action for tomorrow: replace the Gaussian-copula daily 99% VaR / 97.5% ES with the {best}-model "
        f"(t-copula dependence, daily-updated GARCH volatility), and stop scaling 1-day risk by sqrt(10).\n"
        f"  - Daily limit: 97.5% ES today = {100 * lat[f'es97_5_{best}']:.2f}% of NAV "
        f"(Gaussian: {100 * lat['es97_5_G']:.2f}%).\n"
        f"  - 10-day ES 97.5% today = {100 * lat['es97_5_10d_WC']:.2f}% (wavelet-copula) vs "
        f"{100 * lat['es97_5_10d_G_sqrt']:.2f}% from Gaussian x sqrt(10).\n"
        f"Evidence (out-of-sample {H['oos_window']['start']} to {H['oos_window']['end']}):\n"
        f"  - 99% VaR exceptions: {best} {s1[best]['var99']['hits']} vs Gaussian {s1['G']['var99']['hits']} "
        f"(expected {s1['G']['var99']['expected_hits']:.0f}); Basel red-zone share of 250-day windows: "
        f"{best} {100 * tl[best]['red_share']:.0f}% vs Gaussian {100 * tl['G']['red_share']:.0f}%.\n"
        f"  - Crash co-movement does not fade with horizon (chi_L H1 {t['H1_L']['est']:.2f} -> H3 {t['H3_L']['est']:.2f}) "
        f"while rally co-movement does (chi_U {t['H1_U']['est']:.2f} -> {t['H3_U']['est']:.2f}); diversification "
        f"that looks fine on up-days is not there on down-days at any horizon.\n"
        f"  - 10-day ES: Gaussian x sqrt(10) differs from the wavelet-copula by {100 * gap['G_sqrt']['all']:+.0f}% on average "
        f"({100 * gap['G_sqrt']['calm']:+.0f}% in calm, {100 * gap['G_sqrt']['stressed']:+.0f}% in stressed windows).\n"
        "Review trigger: escalate if the 250-day count of 99% exceptions reaches 5 (Basel yellow).\n"
        "Governance: internal risk overlay; regulatory capital model unchanged until independent validation.\n"
    )
    return txt


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== Quant Edge Round 1: tails x timescales ==")
    prices = download_prices()
    rets = log_returns(prices)
    print(f"Data: {prices.index.min().date()} -> {prices.index.max().date()}, {len(prices)} days, {TICKERS}")

    print("[1/3] In-sample horizon analysis (<= 2019)")
    ris = rets.loc[:IN_SAMPLE_END]
    fits_is = fit_all_margins(ris)
    z = std_resid_matrix(fits_is)
    cop = copula_table(z)
    cop.to_csv(OUTPUT_DIR / "insample_copula_fits.csv", index=False)
    band_lv, band_ts = tail_by_band(z)
    band_lv.to_csv(OUTPUT_DIR / "tail_comovement_by_band.csv", index=False)
    band_ts.to_csv(OUTPUT_DIR / "tail_tests_by_band.csv", index=False)
    hor_lv, hor_ts = tail_by_horizon(ris)
    hor_lv.to_csv(OUTPUT_DIR / "tail_comovement_by_horizon.csv", index=False)
    hor_ts.to_csv(OUTPUT_DIR / "tail_tests_by_horizon.csv", index=False)
    implied = model_implied_by_horizon(ris, fits_is)
    implied.to_csv(OUTPUT_DIR / "model_implied_tail_by_horizon.csv", index=False)
    print(cop[cop.aic_winner].round(3).to_string(index=False))

    print("[2/3] Rolling out-of-sample forecasts (2020 ->)")
    d1, dh, rf = run_oos(rets)
    d1.to_csv(OUTPUT_DIR / "oos_1d_forecasts.csv")
    dh.to_csv(OUTPUT_DIR / f"oos_{H_DAYS}d_forecasts.csv")
    rf.to_csv(OUTPUT_DIR / "refit_log.csv")

    print("[3/3] Backtests, figures, recommendation")
    s1, sub = summarize_1d(d1)
    sh = summarize_h(dh)
    sub.to_csv(OUTPUT_DIR / "subperiod_hit_rates.csv", index=False)
    H = _jsonable(headline(prices, cop, band_lv, band_ts, hor_lv, hor_ts, s1, sh, d1, dh))
    (OUTPUT_DIR / "results.json").write_text(json.dumps(H, indent=2))

    figures.fig_tail(band_lv, hor_lv, 0.05, 0.10)
    figures.fig_oos_1d(d1)
    figures.fig_traffic(d1)
    figures.fig_10d(dh)
    figures.fig_refits(rf)

    print("Monte Carlo seed robustness (re-runs the OOS engine with other seeds)")
    rows = [seed_row(d1, dh, "main")]
    for sd in ROBUST_SEEDS:
        a, b, _ = run_oos(rets, seed=sd, verbose=False)
        rows.append(seed_row(a, b, sd))
    rob = pd.DataFrame(rows)
    rob.to_csv(OUTPUT_DIR / "seed_robustness.csv", index=False)
    H["seed_robustness"] = _jsonable(rob.drop(columns="seed").agg(["min", "max"]).to_dict())
    (OUTPUT_DIR / "results.json").write_text(json.dumps(H, indent=2))

    rec = recommendation(H)
    (OUTPUT_DIR / "manager_recommendation.txt").write_text(rec)
    print("\n" + rec)
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
