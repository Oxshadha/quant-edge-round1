"""Report figures (static PNG). Palette: fixed categorical order, recessive grid, single y-axis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, REPORT_ASSETS

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4
INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
GOOD, WARN, CRIT = "#008300", "#eda100", "#e34948"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def _save(fig, name):
    for d in (REPORT_ASSETS, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _tail_panel(ax, df, xcol, xlabels, title):
    x = np.arange(len(xlabels))
    for off, tail, col, lab in [(-0.12, "L", S1, r"Crash co-movement $\chi_L$"),
                                (0.12, "U", S2, r"Rally co-movement $\chi_U$")]:
        d = df[df["tail"] == tail].set_index(xcol).loc[xlabels]
        ax.errorbar(x + off, d.est, yerr=[d.est - d.lo, d.hi - d.est], fmt="o", ms=6, color=col,
                    elinewidth=1.6, capsize=3, label=lab)
    g = df[df["tail"] == "L"].set_index(xcol).loc[xlabels].gaussian_implied
    ax.scatter(x, g, marker="_", s=500, color=INK2, linewidths=2, label="Gaussian-copula implied", zorder=3)
    ax.set_xticks(x, xlabels)
    ax.set_title(title, loc="left")
    ax.set_ylim(0, max(0.65, float(df.hi.max()) + 0.05))


def fig_tail(band_lv: pd.DataFrame, hor_lv: pd.DataFrame, q_band: float, q_hor: float):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    b = band_lv[band_lv.q == q_band].copy()
    b["lab"] = b.series.map({"daily": "Daily\nresiduals", "H1": "H1\n2–8 d", "H2": "H2\n8–32 d", "H3": "H3\n>32 d"})
    _tail_panel(axes[0], b, "lab", ["Daily\nresiduals", "H1\n2–8 d", "H2\n8–32 d", "H3\n>32 d"],
                f"(a) Wavelet scale bands, q = {q_band:g}")
    h = hor_lv[hor_lv.q == q_hor].copy()
    h["lab"] = h.horizon_days.astype(int).astype(str) + "-day"
    _tail_panel(axes[1], h, "lab", ["1-day", "5-day", "10-day", "21-day"],
                f"(b) Non-overlapping returns, q = {q_hor:g}")
    axes[0].set_ylabel("Avg. pairwise tail co-movement")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig1_tail_by_horizon.png")


def fig_oos_1d(d1: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.0), gridspec_kw={"height_ratios": [1.0, 1.0]})
    for ax, sl, title in [(axes[0], slice(None), "(a) Full out-of-sample period"),
                          (axes[1], slice("2020-02-01", "2020-07-31"), "(b) COVID crash, Feb–Jul 2020")]:
        d = d1.loc[sl]
        ax.bar(d.index, d.r_p * 100, width=1.0, color=GREY, label="Portfolio daily return")
        for m, col, lab in [("HS", S1, "Historical simulation (250d)"), ("G", S2, "Gaussian copula"),
                            ("WT", S3, "Wavelet t-copula (H1)")]:
            ax.plot(d.index, -d[f"var99_{m}"] * 100, color=col, lw=1.2, label=f"−VaR 99%: {lab}")
        hit = d.r_p < -d["var99_WT"]
        ax.scatter(d.index[hit], d.r_p[hit] * 100, s=14, color=CRIT, zorder=4, label="Breach of WT VaR")
        ax.set_ylabel("% return")
        ax.set_title(title, loc="left")
    axes[0].legend(ncol=3, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig2_oos_var99.png")


def fig_traffic(d1: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.axhspan(-0.5, 4.5, color=GOOD, alpha=0.07)
    ax.axhspan(4.5, 9.5, color=WARN, alpha=0.10)
    ax.axhspan(9.5, 40, color=CRIT, alpha=0.07)
    for m, col, lab in [("HS", S1, "Historical simulation"), ("G", S2, "Gaussian copula"),
                        ("T", S4, "t-copula (daily)"), ("WT", S3, "Wavelet t-copula (H1)")]:
        c = (d1.r_p < -d1[f"var99_{m}"]).astype(int).rolling(250).sum()
        ax.plot(c.index, c, color=col, lw=1.4, label=lab)
    ax.text(d1.index[5], 2, "green", color=GOOD, fontsize=8)
    ax.text(d1.index[5], 6.5, "yellow", color=INK2, fontsize=8)
    ax.text(d1.index[5], 11.5, "red", color=CRIT, fontsize=8)
    ax.set_ylim(0, None)
    ax.set_ylabel("99% exceptions, last 250 days")
    ax.set_title("Basel traffic light: rolling 250-day count of 99% VaR exceptions", loc="left")
    ax.legend(ncol=4, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig3_traffic_light.png")


def fig_10d(dh: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    for m, col, lab in [("G_sqrt", S2, r"Gaussian 1-day × $\sqrt{10}$"), ("T_path", S4, "t-copula, simulated paths"),
                        ("WC", S3, "Wavelet-copula (H2-matched)")]:
        ax.step(dh.index, dh[f"es97_5_{m}"] * 100, where="post", color=col, lw=1.3, label=f"ES 97.5%: {lab}")
    loss = -dh.r_h * 100
    ax.scatter(dh.index, loss, s=10, color=INK2, label="Realised 10-day loss", zorder=3)
    hit = dh.r_h < -dh["var97_5_WC"]
    ax.scatter(dh.index[hit], loss[hit], s=22, facecolors="none", edgecolors=CRIT, linewidths=1.4,
               label="Beyond WC VaR 97.5%", zorder=4)
    ax.set_ylabel("% of portfolio")
    ax.set_title("10-day Expected Shortfall 97.5% vs realised 10-day losses (non-overlapping windows)", loc="left")
    ax.legend(ncol=2, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig4_es10d.png")


def fig_refits(rf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.2, 2.6))
    for m, col, lab in [("T", S4, "Daily residuals"), ("WT", S3, "H1 band (2–8 d)"), ("WH", S1, "H2 band (8–32 d)")]:
        ax.plot(rf.index, rf[f"lambda_{m}"], color=col, lw=1.4, marker="o", ms=2.5, label=lab)
    ax.set_ylabel(r"t-copula $\lambda$ (avg. pair)")
    ax.set_title(r"Rolling t-copula tail dependence by scale (window 1000 days, refit every 20)", loc="left")
    ax.legend(ncol=3, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig5_rolling_lambda.png")
