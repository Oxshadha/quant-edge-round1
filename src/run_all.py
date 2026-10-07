"""
One-command reproduction for SAIFA Quant Edge 1.0, Round 1: "Risk Across Tails and Timescales".

    python -m src.run_all          (or: make reproduce)

Stages
  0. Data loading, exploratory analysis and preprocessing diagnostics.
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
from src.config import H_DAYS, IN_SAMPLE_END, OOS_START, OUTPUT_DIR, TICKERS
from src import eda
from src.data import download_prices, load_raw, log_returns
from src.evaluate import summarize_1d, summarize_h
from src.horizon_analysis import copula_table, model_implied_by_horizon, tail_by_band, tail_by_horizon
from src.margins import fit_all_margins, std_resid_matrix
from src.backtest import fz0_loss
from src.oos import MODELS_1D, MODELS_H, run_oos
from src.wavelets import decompose

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
            **{f"es97_5_10d_{m}": float(dh[f"es97_5_{m}"].iloc[-1]) for m in MODELS_H},
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
    s1, sh, lat, t = H["oos_1d"], H["oos_10d"], H["latest"], H["tail_band_q05"]
    g, tc, full = s1["G"], s1["T"], "WC_emp"
    gap = sh["es_gap_vs_WC"]
    rel = lambda m: (1 + gap[m]["all"]) / (1 + gap[full]["all"]) - 1  # noqa: E731  (gaps are stored vs WC)
    return (
        "DESK RECOMMENDATION (equal-weight US sector ETF book: XLE XLF XLK XLV XLI XLU)\n\n"
        "From tomorrow: replace the Gaussian copula and the sqrt(10) rule with tail-aware, horizon-matched ES.\n"
        f"1. Daily limit: 1-day ES 97.5% from the t-copula on daily-updated GARCH margins = {100 * lat['es97_5_T']:.2f}% "
        f"of NAV today (Gaussian copula: {100 * lat['es97_5_G']:.2f}%). Out-of-sample {H['oos_window']['start']} to "
        f"{H['oos_window']['end']}: Gaussian 99% VaR breaches {g['var99']['hits']} vs {g['var99']['expected_hits']:.0f} "
        f"expected (t-copula {tc['var99']['hits']}).\n"
        f"2. 10-day limit: ES 97.5% from the wavelet empirical-copula model = {100 * lat['es97_5_10d_' + full]:.2f}% today "
        f"(sqrt(10) x Gaussian {100 * lat['es97_5_10d_G_sqrt']:.2f}%, Gaussian paths {100 * lat['es97_5_10d_G_path']:.2f}%). "
        f"On average the Gaussian path model sits {100 * -rel('G_path'):.0f}% below it.\n"
        f"3. Diversification: crash co-movement does not fade with horizon (chi_L {t['H1_L']['est']:.2f} at 2-8 days, "
        f"{t['H3_L']['est']:.2f} beyond 32 days) while rally co-movement does ({t['H1_U']['est']:.2f} -> "
        f"{t['H3_U']['est']:.2f}). Do not rely on sector rotation as a crash hedge.\n"
        "4. Trigger: escalate when the 250-day count of 99% breaches reaches 5 (Basel yellow). Internal overlay; "
        "regulatory capital model unchanged until independent validation.\n"
    )


def stage_data():
    """Load the bundled prices and compute log returns."""
    raw = load_raw()
    prices = download_prices()
    rets = log_returns(prices)
    print(f"Data: {prices.index.min().date()} -> {prices.index.max().date()}, {len(prices)} days, {TICKERS}")
    return raw, prices, rets


def stage_eda(raw, prices, rets) -> dict:
    """Exploratory analysis and preprocessing diagnostics (estimation sample <= 2019)."""
    ris = rets.loc[:IN_SAMPLE_END]
    port = np.log1p(np.expm1(ris).mean(axis=1)).rename("Portfolio")
    fits = fit_all_margins(ris)
    z = std_resid_matrix(fits)
    out = {
        "quality": eda.data_quality(raw[TICKERS], prices),
        "stats": eda.return_stats(ris.join(port)),
        "corr": ris.corr(),
        "co_extremes": eda.co_extremes(ris),
        "garch": eda.garch_diagnostics(fits),
        "energy": eda.wavelet_energy(z),
    }
    for k, df in out.items():
        df.to_csv(OUTPUT_DIR / f"eda_{k}.csv")
    avg_corr = eda.rolling_avg_corr(rets)
    figures.fig_data_overview(rets, avg_corr, OOS_START)
    figures.fig_diagnostics(port, ris, z, eda.acf)
    bands = {b: pd.Series(v[:, 0], index=z.index) for b, v in decompose(z[["XLF"]], trim=False).items()}
    figures.fig_wavelet_example(z["XLF"], bands, "2007-01-01", "2009-12-31")
    vol = (np.expm1(rets).mean(axis=1).rolling(250).std() * np.sqrt(252)).reindex(avg_corr.index)
    ok = avg_corr.notna() & vol.notna()
    out["corr_vol_link"] = float(np.corrcoef(avg_corr[ok], vol[ok])[0, 1])
    out["avg_corr_range"] = (float(avg_corr.min()), float(avg_corr.max()))
    out["summary"] = {"duplicate_dates": out["quality"].attrs.get("duplicate_dates", 0),
                      "rows_dropped_by_alignment": out["quality"].attrs.get("rows_dropped_by_alignment", 0),
                      "corr_vol_link": out["corr_vol_link"], "avg_corr_min": out["avg_corr_range"][0],
                      "avg_corr_max": out["avg_corr_range"][1]}
    out["fits"], out["z"], out["ris"] = fits, z, ris
    return out


def stage_horizon(ris, fits_is, z) -> dict:
    """Part (a): copula fits and tail co-movement by scale and by horizon."""
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
    figures.fig_tail(band_lv, hor_lv, 0.05, 0.10)
    return {"cop": cop, "band_lv": band_lv, "band_ts": band_ts, "hor_lv": hor_lv, "hor_ts": hor_ts,
            "implied": implied}


def stage_oos(rets) -> dict:
    """Part (b): rolling out-of-sample 1-day and 10-day forecasts."""
    d1, dh, rf = run_oos(rets)
    d1.to_csv(OUTPUT_DIR / "oos_1d_forecasts.csv")
    dh.to_csv(OUTPUT_DIR / f"oos_{H_DAYS}d_forecasts.csv")
    rf.to_csv(OUTPUT_DIR / "refit_log.csv")
    return {"d1": d1, "dh": dh, "rf": rf}


def stage_evaluate(prices, rets, hz: dict, oo: dict, eda_summary: dict | None = None, robustness: bool = True) -> dict:
    """Backtests, figures, seed robustness, results.json and the desk recommendation."""
    d1, dh, rf = oo["d1"], oo["dh"], oo["rf"]
    s1, sub = summarize_1d(d1)
    sh = summarize_h(dh)
    sub.to_csv(OUTPUT_DIR / "subperiod_hit_rates.csv", index=False)
    H = _jsonable(headline(prices, hz["cop"], hz["band_lv"], hz["band_ts"], hz["hor_lv"], hz["hor_ts"], s1, sh, d1, dh))
    H["eda"] = _jsonable(eda_summary or {})
    figures.fig_oos_1d(d1)
    figures.fig_traffic(d1)
    figures.fig_10d(dh)
    figures.fig_refits(rf)
    if robustness:
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
    return {"H": H, "sub": sub, "rec": rec}


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== Quant Edge Round 1: tails x timescales ==")
    raw, prices, rets = stage_data()
    print("[1/4] Exploratory analysis and preprocessing diagnostics")
    ed = stage_eda(raw, prices, rets)
    print("[2/4] In-sample horizon analysis (<= 2019)")
    hz = stage_horizon(ed["ris"], ed["fits"], ed["z"])
    print(hz["cop"][hz["cop"].aic_winner].round(3).to_string(index=False))
    print("[3/4] Rolling out-of-sample forecasts (2020 ->)")
    oo = stage_oos(rets)
    print("[4/4] Backtests, figures, recommendation")
    ev = stage_evaluate(prices, rets, hz, oo, ed["summary"])
    print("\n" + ev["rec"])
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
