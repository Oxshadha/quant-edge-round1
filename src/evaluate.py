"""Turn OOS forecasts into backtest tables (1-day and 10-day)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import diebold_mariano, es_backtest, fz0_loss, traffic_light, var_backtest
from src.oos import MODELS_1D, MODELS_H

SUBPERIODS = {
    "COVID crash (Feb–Jun 2020)": ("2020-02-15", "2020-06-30"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
    "Other days": None,
}


def _model_block(r, df, m, suffix=""):
    s = suffix
    out = {
        "var95": var_backtest(r, df[f"var95_{m}{s}"].to_numpy(), 0.05),
        "var99": var_backtest(r, df[f"var99_{m}{s}"].to_numpy(), 0.01),
        "es97_5": es_backtest(r, df[f"var97_5_{m}{s}"].to_numpy(), df[f"es97_5_{m}{s}"].to_numpy(), 0.025),
        "mean_var99": float(df[f"var99_{m}{s}"].mean()),
        "mean_es97_5": float(df[f"es97_5_{m}{s}"].mean()),
    }
    out["fz0_97_5"] = float(fz0_loss(r, df[f"var97_5_{m}{s}"], df[f"es97_5_{m}{s}"], 0.025).mean())
    return out


def summarize_1d(d1: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    r = d1["r_p"].to_numpy()
    res = {}
    for m in MODELS_1D:
        res[m] = _model_block(r, d1, m)
        res[m]["traffic_light"] = traffic_light((r < -d1[f"var99_{m}"].to_numpy()).astype(int))
    loss = {m: fz0_loss(r, d1[f"var97_5_{m}"], d1[f"es97_5_{m}"], 0.025) for m in MODELS_1D}
    res["dm_fz0_97_5"] = {f"{a}_vs_{b}": diebold_mariano(loss[a], loss[b])
                         for a, b in [("T", "G"), ("WT", "G"), ("FHS", "G"), ("WT", "T"),
                                                ("FHS", "HS"), ("G", "HS"), ("T", "HS"), ("WT", "HS")]}

    rows = []
    idx = d1.index
    used = np.zeros(len(d1), dtype=bool)
    for name, span in SUBPERIODS.items():
        mask = (idx >= span[0]) & (idx <= span[1]) if span else ~used
        used |= mask
        for m in MODELS_1D:
            sub = d1[mask]
            rows.append({"period": name, "model": m, "days": int(mask.sum()),
                         "hit95_pct": 100 * float((sub.r_p < -sub[f"var95_{m}"]).mean()),
                         "hit99_pct": 100 * float((sub.r_p < -sub[f"var99_{m}"]).mean())})
    return res, pd.DataFrame(rows)


def summarize_h(dh: pd.DataFrame) -> dict:
    r = dh["r_h"].to_numpy()
    res = {m: _model_block(r, dh, m) for m in MODELS_H}
    loss = {m: fz0_loss(r, dh[f"var97_5_{m}"], dh[f"es97_5_{m}"], 0.025) for m in MODELS_H}
    res["dm_fz0_97_5"] = {f"{a}_vs_WC": diebold_mariano(loss[a], loss["WC"], lag=0)
                         for a in MODELS_H if a != "WC"}
    # "What does ignoring it do?": ES97.5 of each simplification relative to the wavelet-copula model
    wc = dh["es97_5_WC"]
    stressed = wc >= wc.quantile(0.75)
    gaps = {}
    for m in MODELS_H:
        rel = dh[f"es97_5_{m}"] / wc - 1.0
        gaps[m] = {"all": float(rel.mean()), "calm": float(rel[~stressed].mean()),
                   "stressed": float(rel[stressed].mean())}
    res["es_gap_vs_WC"] = gaps
    res["n_windows"] = int(len(dh))
    return res
