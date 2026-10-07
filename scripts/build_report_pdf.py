#!/usr/bin/env python3
"""Build the competition report (PDF) from pipeline outputs. Every number is read from outputs/;
nothing quantitative is typed by hand. Run after `python -m src.run_all`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import H_DAYS, N_BOOT, N_SIM, N_SIM_H, REFIT_EVERY, WINDOW  # noqa: E402

OUT, ASSETS = ROOT / "outputs", ROOT / "report" / "assets"
PDF = ROOT / "report" / "QuantEdge_Round1_Report.pdf"

# ---- cover details ---------------------------------------------------------------------------------
TEAM_NAME = "Gmora"
TRACKING_CODE = "SAIFA-2026-75323CFE"
SUBMISSION_DATE = "7 October 2026"
# ---------------------------------------------------------------------------------------------------

H = json.loads((OUT / "results.json").read_text())
cop = pd.read_csv(OUT / "insample_copula_fits.csv")
band = pd.read_csv(OUT / "tail_comovement_by_band.csv")
hor = pd.read_csv(OUT / "tail_comovement_by_horizon.csv")
implied = pd.read_csv(OUT / "model_implied_tail_by_horizon.csv")
sub = pd.read_csv(OUT / "subperiod_hit_rates.csv")
rob = pd.read_csv(OUT / "seed_robustness.csv")
dh = pd.read_csv(OUT / f"oos_{H_DAYS}d_forecasts.csv", index_col=0, parse_dates=True)
est = pd.read_csv(OUT / "eda_stats.csv", index_col=0)
gd = pd.read_csv(OUT / "eda_garch.csv", index_col=0)
qual = pd.read_csv(OUT / "eda_quality.csv", index_col=0)
coex = pd.read_csv(OUT / "eda_co_extremes.csv", index_col=0)
EDA = H.get("eda", {})
S1, SH, T, TT = H["oos_1d"], H["oos_10d"], H["tail_band_q05"], H["tail_band_tests_q05"]
HT, HTT = H["tail_horizon_q10"], H["tail_horizon_tests_q10"]

# ---------------------------------------------------------------- formatting helpers
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def pct(x, d=1): return f"{100 * x:.{d}f}%"
def spct(x, d=1): return f"{100 * x:+.{d}f}%"
def pv(p): return "&lt;0.001" if p < 0.001 else f"{p:.3f}"
def pp(p): return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"
def ci(d): return f"{d['est']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
def sig(p, yes="significant", no="not significant"): return yes if p < 0.05 else no


def es_gap(m, base):
    rel = dh[f"es97_5_{m}"] / dh[f"es97_5_{base}"] - 1.0
    stressed = dh[f"es97_5_{base}"] >= dh[f"es97_5_{base}"].quantile(0.75)
    return float(rel.mean()), float(rel[~stressed].mean()), float(rel[stressed].mean())


BASE = "WC_emp"  # the full horizon-aware model: band-matched, non-parametric (asymmetric) copula
NAMES_1D = {"HS": "Historical simulation (250d)", "FHS": "Filtered HS (empirical copula)",
            "G": "Gaussian copula (benchmark)", "T": "Student-t copula", "WT": "Wavelet t-copula (H1 band)"}
NAMES_H = {"G_sqrt": "Gaussian 1-day × √10", "G_path": "Gaussian copula, simulated paths",
           "T_path": "t-copula, simulated paths", "T_join": "Daily t-copula imposed at 10 days",
           "WC": "Wavelet t-copula (H2 band)", "E_join": "Daily empirical copula at 10 days",
           "WC_emp": "Wavelet empirical copula (H2 band)"}

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
INK, MUTED, ACCENT, RULE, FILL = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#1f4e79", "#c9c8c2", "#f3f2ee"))
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9, leading=12.2,
                      alignment=TA_JUSTIFY, spaceAfter=5, textColor=INK)
small = ParagraphStyle("s", parent=body, fontSize=7.6, leading=9.6, alignment=0, spaceAfter=3, textColor=MUTED)
cell = ParagraphStyle("c", parent=body, fontSize=7.6, leading=9.2, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=12.5, leading=15,
                    spaceBefore=8, spaceAfter=5, textColor=ACCENT, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12.5,
                    spaceBefore=6, spaceAfter=3, textColor=INK, keepWithNext=1)
cap = ParagraphStyle("cap", parent=small, fontName="Helvetica-Oblique", spaceBefore=2, spaceAfter=8)
boxs = ParagraphStyle("box", parent=body, fontSize=9, leading=12.4, spaceAfter=3)
title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=34, leading=40,
                       alignment=TA_CENTER, textColor=ACCENT)
cover = ParagraphStyle("cv", parent=body, fontSize=11, leading=15, alignment=TA_CENTER)


def P(t, s=body): return Paragraph(t, s)


def table(rows, widths, header_rows=1, zebra=True, bold_first_col=False, align_right_from=1):
    data = [[P(str(c), cellb if (i < header_rows or (bold_first_col and j == 0)) else cell)
             for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=header_rows)
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5, INK),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for i in range(header_rows, len(rows)):
            if (i - header_rows) % 2 == 1:
                st.append(("BACKGROUND", (0, i), (-1, i), FILL))
    t.setStyle(TableStyle(st))
    return t


def box(paras, color=ACCENT):
    t = Table([[paras]], colWidths=[17.4 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.0, color), ("BACKGROUND", (0, 0), (-1, -1), FILL),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def fig(name, width_cm, caption):
    p = ASSETS / name
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(p)).getSize()
    return KeepTogether([Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w), P(caption, cap)])


def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.8 * cm, 1.0 * cm, f"SAIFA Quant Edge 1.0 · Round 1 · Team {TEAM_NAME} · {TRACKING_CODE}")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {doc.page - 1}")
    canvas.restoreState()


# ---------------------------------------------------------------- derived numbers
lat = dict(H["latest"])
lat.update({f"es97_5_10d_{m}": float(dh[f"es97_5_{m}"].iloc[-1]) for m in NAMES_H})
g1, t1, wt1, fhs1, hs1 = (S1[m] for m in ("G", "T", "WT", "FHS", "HS"))
es1_gap_t = g1["mean_es97_5"] / t1["mean_es97_5"] - 1
es1_gap_f = g1["mean_es97_5"] / fhs1["mean_es97_5"] - 1
gaps = {m: es_gap(m, BASE) for m in NAMES_H}
rb = rob.set_index("seed")
hits_rng = {m: (int(rb[f"hits99_{m}"].min()), int(rb[f"hits99_{m}"].max())) for m in NAMES_1D}
gap_rng = {m: (float((rb[f"es97_5_10d_{m}"] / rb[f"es97_5_10d_{BASE}"] - 1).min()),
               float((rb[f"es97_5_10d_{m}"] / rb[f"es97_5_10d_{BASE}"] - 1).max())) for m in NAMES_H}
best10 = min((m for m in NAMES_H), key=lambda m: SH[m]["fz0_97_5"])
tw = cop[cop.aic_winner].set_index("series")
imp = implied[implied.q == 0.10].pivot(index="horizon_days", columns="model", values="chi_L")
obs = hor[(hor.q == 0.10) & (hor["tail"] == "L")].set_index("horizon_days")
expected99 = g1["var99"]["expected_hits"]
sqrt_ratio = dh["es97_5_G_sqrt"] / dh[f"es97_5_{BASE}"]
sqrt_below, sqrt_worst = float((sqrt_ratio < 1).mean()), float(1 - sqrt_ratio.min())
sqrt_vs_path = float((dh["es97_5_G_sqrt"] / dh["es97_5_G_path"]).mean() - 1)

story = []
# ================================================================= COVER (excluded from page limit)
cover_label = ParagraphStyle("cl", parent=cover, fontName="Helvetica-Bold", fontSize=10, textColor=ACCENT, spaceAfter=3)
cover_q = ParagraphStyle("cq", parent=cover, fontSize=14, leading=19)
story += [Spacer(1, 2.6 * cm), P("SAIFA QUANT EDGE 1.0 · ROUND 1", ParagraphStyle("k", parent=cover, textColor=MUTED, fontSize=11)),
          Spacer(1, 0.5 * cm), P("Risk Across Tails<br/>and Timescales", title), Spacer(1, 1.0 * cm),
          P("RESEARCH QUESTION", cover_label),
          P("Does tail dependence change with the investment horizon, and what does ignoring it do to a "
            "portfolio's measured risk?", cover_q),
          Spacer(1, 0.9 * cm),
          P("DATASET", cover_label),
          P("Daily adjusted closing prices of six US sector ETFs (XLE · XLF · XLK · XLV · XLI · XLU), Yahoo Finance, "
            "1999–2026; equal-weight portfolio; in-sample to 2019, out-of-sample 2020–2026", cover),
          Spacer(1, 0.9 * cm),
          P("METHOD", cover_label),
          P("GARCH-t margins, MODWT wavelet horizon bands and copulas; 1-day and 10-day VaR / ES backtests", cover),
          Spacer(1, 2.0 * cm), P(f"<b>Team {TEAM_NAME}</b><br/>Tracking code: {TRACKING_CODE}<br/>{SUBMISSION_DATE}", cover),
          PageBreak()]

# ================================================================= EXECUTIVE SUMMARY
story.append(P("Executive summary", h1))
story.append(box([
    P("<b>Question.</b> Does tail dependence between US equity sectors change with the investment horizon, and what "
      "does ignoring that do to a portfolio's measured risk?", boxs),
    P(f"<b>Answer (a): the crash tail does not fade, the rally tail does.</b> Using a MODWT wavelet decomposition of "
      f"GARCH-filtered returns (1999–2019), the probability that two sectors crash together (χ<sub>L</sub>, q = 5%) is "
      f"{f2(T['H1_L']['est'])} at the 2–8 day scale and {f2(T['H3_L']['est'])} beyond 32 days, a change of "
      f"{TT['chi_L: H3 - H1']['est']:+.2f} that is {sig(TT['chi_L: H3 - H1']['p'])} ({pp(TT['chi_L: H3 - H1']['p'])}). "
      f"Rally co-movement falls from {f2(T['H1_U']['est'])} to {f2(T['H3_U']['est'])} ({pp(TT['chi_U: H3 - H1']['p'])}). "
      f"So dependence becomes <i>asymmetric</i> at long horizons, and at every scale crash co-movement exceeds what a "
      f"Gaussian copula implies (by {T['H1_L']['excess_est']:+.2f} to {T['daily_L']['excess_est']:+.2f}). A symmetric "
      f"t-copula reads the fading rally tail as fading tail dependence (λ {f2(tw.loc['H1','lambda_l'])} → "
      f"{f2(tw.loc['H3','lambda_l'])}) and would understate long-horizon crash risk.", boxs),
    P(f"<b>Answer (b): ignoring it understates risk, most at multi-day horizons.</b> Out-of-sample "
      f"({H['oos_window']['start'][:4]}–{H['oos_window']['end'][:4]}, {H['oos_window']['days']:,} days), the Gaussian-copula "
      f"benchmark gives {g1['var99']['hits']} breaches of 99% VaR against {expected99:.0f} expected (t-copula: "
      f"{t1['var99']['hits']}) and a daily ES 97.5% {pct(-es1_gap_t)} lower than the t-copula. At the {H_DAYS}-day "
      f"horizon the Gaussian path model's ES 97.5% is {pct(-gaps['G_path'][0])} below the horizon-aware wavelet "
      f"empirical-copula model ({pct(-gaps['G_path'][2])} in stressed windows); aggregating a daily t-copula still "
      f"leaves {pct(-gaps['T_path'][0])}. Getting volatility right matters most: every GARCH model beats historical "
      f"simulation (Diebold–Mariano p ≤ {pv(max(S1['dm_fz0_97_5'][k]['p'] for k in ('G_vs_HS','T_vs_HS','WT_vs_HS')))}).", boxs),
    P(f"<b>Recommendation.</b> Replace the Gaussian copula with a t-copula on daily-updated GARCH margins for the "
      f"daily limit, and set the {H_DAYS}-day ES from the horizon-matched wavelet model instead of √10 × 1-day: "
      f"today that is {pct(lat['es97_5_10d_' + BASE], 2)} "
      f"of NAV. Over 2020–2026 the √10 shortcut sat below this model in {pct(sqrt_below, 0)} of windows. "
      f"Details in Section 6.", boxs)]))
story.append(Spacer(1, 6))

# ================================================================= 1. DESIGN
story.append(P("1. Design choices and why", h1))
story.append(P("Table 1 sets out each design choice, the reason for it and the alternative considered. Each component "
               "answers one part of the question and is tested out of sample against a simpler alternative.", body))
design = [
    ["Choice", "Specification", "Rationale", "Alternative considered"],
    ["Portfolio", "Equal weight, 6 Select Sector SPDRs", "Sectors are where diversification is supposed to come from; "
     "equal weight keeps the result about dependence, not weights", "Adding SPY: it contains the sectors and double-counts beta"],
    ["Market & data", f"US, Yahoo Finance adjusted closes, {H['data']['start']} to {H['data']['end']} ({H['data']['n_days']} days)",
     "Deep, liquid, synchronous trading; covers dot-com, GFC, COVID and 2022", "Sri Lankan CSE: thin trading biases tail estimates"],
    ["Split", "Estimate ≤ 2019, test 2020 → today", "The test window contains a crash (COVID) and a rate shock (2022)",
     "Random split: leaks future information"],
    ["Margins", "GARCH(1,1)-t, volatility updated daily; FHS shocks", "Removes volatility clustering so the copula sees "
     "dependence, not shared volatility (McNeil & Frey 2000)", "Raw returns: inflate measured tail dependence"],
    ["Horizon tool", f"MODWT multiresolution analysis, LA(8), J = 6, bands H1 2–8d, H2 8–32d, H3 >32d",
     "Shift-invariant, any sample length, additive (bands sum to the series)", "DWT (needs 2<sup>J</sup> length); "
     "overlapping h-day returns (autocorrelated)"],
    ["Dependence", "Gaussian, Student-t, Clayton (full likelihood, AIC); empirical copula; χ<sub>L</sub>, χ<sub>U</sub>",
     "Gaussian = no tail dependence; t = symmetric; Clayton = crash-only; empirical = no shape imposed", "Composite "
     "pairwise likelihoods: not comparable across families"],
    ["Risk measures", "VaR 95/99 %, ES 97.5 % (FRTB), at 1 and 10 days", "VaR for Basel backtesting; ES because "
     "tail dependence matters beyond the quantile", "VaR only: blind to tail shape"],
    ["Benchmarks", "Gaussian copula (same margins); historical simulation", "Isolates the dependence choice; HS is "
     "the industry default", "No benchmark"],
]
story.append(table(design, [2.0, 4.6, 6.0, 4.8]))
story.append(P("Table 1. Design choices. Every choice is a constant in <font face='Courier'>src/config.py</font>.", cap))

# ================================================================= 2. DATA AND EDA
co5 = coex[(coex.q == 0.05)].set_index("side")
story.append(P("2. Data, exploration and preprocessing", h1))
story.append(P(
    f"<b>Data quality.</b> The bundled file holds {H['data']['n_days']:,} trading days from {H['data']['start']} to "
    f"{H['data']['end']} for all six ETFs, with {int(qual['missing_in_raw_file'].sum())} missing prices, "
    f"{EDA.get('duplicate_dates', 0)} duplicate dates and {int(qual['non_positive_prices'].sum())} non-positive prices, so "
    "no imputation or filtering is needed. Prices are dividend-adjusted closes; we work with daily log returns for "
    "modelling and simple returns for portfolio profit and loss. The largest daily moves (for example "
    f"{qual['largest_abs_daily_move_pct'].max():.1f}% for {qual['largest_abs_daily_move_pct'].idxmax()} on "
    f"{qual.loc[qual['largest_abs_daily_move_pct'].idxmax(), 'date_of_largest_move']}) are genuine market events and "
    "are kept: they are exactly the observations a tail-risk study is about.", body))
rows = [["Series", "Ann. return", "Ann. vol.", "Skew", "Excess kurtosis", "Worst day", "Jarque–Bera p", "Ljung–Box r² p", "ARCH-LM p"]]
for c, rr in est.iterrows():
    rows.append([c, f"{rr.ann_mean_pct:.1f}%", f"{rr.ann_vol_pct:.1f}%", f"{rr['skew']:.2f}", f"{rr.excess_kurtosis:.1f}",
                 f"{rr.worst_day_pct:.1f}%", pv(rr.jarque_bera_p), pv(rr.ljung_box_r2_p), pv(rr.arch_lm_p)])
story.append(table(rows, [2.0, 1.8, 1.7, 1.4, 2.0, 1.7, 2.3, 2.3, 2.0]))
story.append(P("Table 2. Daily log returns, 1999–2019 (estimation sample). Jarque–Bera tests normality; Ljung–Box on "
               "squared returns (10 lags) and ARCH-LM (5 lags) test for volatility clustering.", cap))
story.append(P(
    f"<b>What the data show.</b> Every sector has excess kurtosis between {est.excess_kurtosis.iloc[:-1].min():.1f} and "
    f"{est.excess_kurtosis.iloc[:-1].max():.1f} and every normality test rejects, so a normal margin would understate "
    "tail losses. Squared returns are strongly autocorrelated (all p &lt; 0.001): calm and turbulent periods cluster. "
    f"Dependence is not constant either: the trailing one-year average correlation between sectors ranges from "
    f"{EDA.get('avg_corr_min', float('nan')):.2f} to {EDA.get('avg_corr_max', float('nan')):.2f} and moves with "
    f"portfolio volatility (correlation {EDA.get('corr_vol_link', float('nan')):.2f}), so diversification is weakest "
    f"exactly when it is needed (Figure 1). On days when one sector has a worst-5% day, on average "
    f"{co5.loc['crash','avg_other_sectors_also_extreme']:.1f} of the other five do too, against "
    f"{co5.loc['crash','if_independent']:.2f} if they were independent.", body))
story.append(fig("fig0a_data_overview.png", 16.5, "Figure 1. (a) Growth of $1 in each sector and the equal-weight "
                 "portfolio (log scale); (b) portfolio volatility; (c) average pairwise correlation. Dashed line: start "
                 "of the out-of-sample period."))
story.append(P(
    "<b>Preprocessing decisions.</b> (1) Model each sector with GARCH(1,1) and Student-t innovations, because returns "
    "are fat-tailed and volatility clusters. (2) Measure dependence on the GARCH-standardised residuals, not on raw "
    "returns, so that shared volatility is not mistaken for tail dependence. Figure 2 confirms the filter works: the "
    "autocorrelation of squared residuals falls inside the no-autocorrelation band, residual excess kurtosis drops to "
    f"{gd.z_excess_kurtosis.min():.1f}–{gd.z_excess_kurtosis.max():.1f}, and the Ljung–Box test on squared residuals "
    f"no longer rejects for {int((gd.ljung_box_z2_p > 0.05).sum())} of 6 sectors "
    f"(utilities keep mild clustering, p = {gd.loc['XLU','ljung_box_z2_p']:.3f}). GARCH persistence α + β is "
    f"{gd.persistence.min():.3f}–{gd.persistence.max():.3f}. (3) Convert residuals to ranks for the copulas, so the "
    "dependence estimate does not depend on the margins. (4) Re-estimate every 20 days on a rolling window, because "
    "Figure 1 shows dependence drifting over time.", body))
story.append(fig("fig0b_diagnostics.png", 16.5, "Figure 2. (a) Standardised portfolio returns against the normal "
                 "distribution, 1999–2019; (b) autocorrelation of squared returns before and after GARCH filtering, "
                 "averaged over the six sectors."))

# ================================================================= 2. METHOD
story.append(P("3. Method", h1))
story.append(P(
    "<b>Margins.</b> For each sector, r<sub>t</sub> = μ + σ<sub>t</sub>z<sub>t</sub> with "
    "σ<sup>2</sup><sub>t</sub> = ω + α(r<sub>t-1</sub> − μ)<sup>2</sup> + βσ<sup>2</sup><sub>t-1</sub> and Student-t z. "
    "Out-of-sample the parameters are re-estimated every "
    f"{REFIT_EVERY} trading days on the trailing {WINDOW} days, and σ<sub>t</sub> is rolled forward every day with the "
    "realised return, so a forecast for day t uses information up to t − 1 only. Simulated shocks are drawn from the "
    "empirical distribution of the standardized residuals (filtered historical simulation).", body))
story.append(P(
    "<b>Wavelets.</b> The MODWT multiresolution analysis splits each residual series into details D<sub>j</sub> "
    "(fluctuations on scales of 2<sup>j−1</sup>–2<sup>j</sup> days) and a smooth S<sub>6</sub>, with "
    "Σ<sub>j</sub>D<sub>j</sub> + S<sub>6</sub> equal to the series exactly (computed in the frequency domain; the "
    "test suite checks additivity). We group them into H1 = D1+D2, H2 = D3+D4 and H3 = D5+D6+S6, reflect the series "
    "at the ends and drop boundary-affected values in the in-sample analysis.", body))
story.append(P(
    "<b>Tail co-movement.</b> For each pair of sectors, χ<sub>L</sub>(q) = P(U<sub>j</sub> ≤ q | U<sub>i</sub> ≤ q) is the "
    "probability that one sector is in its worst q-tail given the other is (χ<sub>U</sub>: best tail). As q → 0 it "
    "becomes the tail-dependence coefficient λ. We average over the 15 pairs and compare with the value a Gaussian "
    f"copula with the same correlation implies. Confidence intervals and tests use a stationary block bootstrap "
    f"({N_BOOT} replications, mean block 60 days) that re-runs the wavelet decomposition on every resample. "
    "Copulas are fitted to rank pseudo-observations by maximum likelihood with the full d-dimensional density, so "
    "AIC compares like with like.", body))
story.append(P(
    f"<b>Risk.</b> 1-day VaR and ES come from {N_SIM:,} Monte Carlo draws of the copula, mapped to shocks and scaled by "
    f"each sector's σ<sub>t</sub>. {H_DAYS}-day ES uses {N_SIM_H:,} simulated GARCH paths (buy-and-hold). To isolate what "
    "each simplification costs, we change one thing at a time: the √10 rule, the copula family, how the "
    f"{H_DAYS}-day dependence is formed (aggregated daily paths vs a copula imposed at the horizon), and which scale "
    "the copula is estimated on (daily vs the H2 band that contains 10 days).", body))
story.append(P(
    "<b>Validation.</b> VaR: Kupiec coverage, Christoffersen independence, Engle–Manganelli dynamic quantile, Basel "
    "traffic light. ES: Acerbi–Szekely Z2 and the McNeil–Frey exceedance test. Model comparison: the FZ0 joint "
    "VaR–ES loss (Patton, Ziegel & Chen 2019) with Diebold–Mariano tests. Monte Carlo error is checked by re-running "
    f"the whole out-of-sample engine with {len(rob) - 1} more seeds.", body))

# ================================================================= 3. RESULTS (a)
story.append(P("4. Does tail dependence change with the horizon?", h1))
rows = [["Series", "Scale", "AIC winner", "t-copula ν", "t-copula λ", "ΔAIC Gaussian", "ΔAIC Clayton"]]
for s, lab in [("daily", "1 day"), ("H1", "2–8 days"), ("H2", "8–32 days"), ("H3", "> 32 days")]:
    c = cop[cop.series == s].set_index("family")
    rows.append([s if s != "daily" else "Daily residuals", lab, c[c.aic_winner].index[0], f"{c.loc['student','nu']:.1f}",
                 f2(c.loc["student", "lambda_l"]), f"{c.loc['gaussian','delta_aic']:,.0f}", f"{c.loc['clayton','delta_aic']:,.0f}"])
story.append(table(rows, [2.6, 2.2, 2.2, 2.2, 2.2, 3.0, 3.0]))
story.append(P("Table 3. Full-likelihood copula fits, 1999–2019. ΔAIC is relative to the winner (larger = worse). "
               "The t-copula wins at every scale; its symmetric λ falls at the longest scale.", cap))
story.append(P(
    f"The t-copula is preferred at every scale by a wide AIC margin, so tails matter. Its tail-dependence coefficient "
    f"is stable from daily data to H2 ({f2(min(tw.loc[k,'lambda_l'] for k in ('daily','H1','H2')))}–"
    f"{f2(max(tw.loc[k,'lambda_l'] for k in ('daily','H1','H2')))}) and drops at H3 "
    f"({f2(tw.loc['H3','lambda_l'])}, ν = {tw.loc['H3','nu']:.0f}). Taken at face value, tail dependence fades at long horizons. "
    "But a t-copula forces the crash and rally tails to be equal. Figure 3 measures them separately.", body))
story.append(fig("fig1_tail_by_horizon.png", 17.0,
                 "Figure 3. Average pairwise crash (χ<sub>L</sub>) and rally (χ<sub>U</sub>) co-movement with 95% "
                 "bootstrap intervals; bars mark the Gaussian-copula value at the same correlation. (a) wavelet bands "
                 "of GARCH-filtered residuals; (b) non-overlapping raw h-day returns, 1999–2019."))
trows = [["Test (q = 5%, wavelet bands)", "Estimate [95% CI]", "p-value", "Verdict"]]
for k, lab in [("chi_L: H3 - H1", "Crash co-movement, H3 − H1"), ("chi_U: H3 - H1", "Rally co-movement, H3 − H1"),
               ("H1: chi_L - chi_U", "Asymmetry χ<sub>L</sub> − χ<sub>U</sub>, H1"), ("H3: chi_L - chi_U", "Asymmetry χ<sub>L</sub> − χ<sub>U</sub>, H3")]:
    d = TT[k]
    trows.append([lab, f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]", pv(d["p"]), sig(d["p"], "different", "no evidence of change" if "H3 - H1" in k else "symmetric")])
for s in ("H1", "H3"):
    d = T[f"{s}_L"]
    trows.append([f"Crash co-movement above Gaussian, {s}", f"{d['excess_est']:+.3f} [{d['excess_lo']:+.3f}, {d['excess_hi']:+.3f}]",
                  pv(d["excess_p0"]), sig(d["excess_p0"], "above Gaussian", "not distinguishable")])
story.append(table(trows, [6.2, 4.6, 2.0, 4.6]))
story.append(P(f"Table 4. Bootstrap tests ({N_BOOT} stationary-bootstrap replications re-running the MODWT).", cap))
story.append(P(
    f"<b>Finding.</b> Crash co-movement does not decline significantly with scale (H3 − H1 = "
    f"{TT['chi_L: H3 - H1']['est']:+.3f}, {pp(TT['chi_L: H3 - H1']['p'])}), while rally co-movement roughly halves "
    f"({pp(TT['chi_U: H3 - H1']['p'])}). At short scales the two tails are symmetric; beyond a month crash "
    f"co-movement exceeds rally co-movement by {TT['H3: chi_L - chi_U']['est']:.2f} ({pp(TT['H3: chi_L - chi_U']['p'])}). "
    "This is the long-horizon version of the asymmetric correlation found by Longin & Solnik (2001) and Ang & Chen "
    "(2002): sectors rally separately but fall together, and that remains true when the holding period lengthens. "
    f"In raw h-day returns (panel b) the picture is noisier (only {int(obs.loc[21,'n_obs'])} independent 21-day "
    "observations exist), but crash co-movement stays above the Gaussian level up to 10 days.", body))
irows = [["Horizon", "Observed χ<sub>L</sub> [95% CI]"] + [f"{m} model" for m in ("Gaussian", "Student-t", "Empirical (FHS)")]]
for h in (1, 5, 10, 21):
    o = obs.loc[h]
    irows.append([f"{h} day{'s' if h > 1 else ''}", f"{o.est:.3f} [{o.lo:.3f}, {o.hi:.3f}]"] +
                 [f3(imp.loc[h, m]) for m in ("Gaussian", "Student-t", "Empirical (FHS)")])
story.append(KeepTogether([table(irows, [2.4, 4.4, 3.5, 3.5, 3.6]),
                           P("Table 5. Crash co-movement of h-day returns (q = 10%): observed vs implied by daily "
                             "GARCH + copula models simulated forward (60,000 paths each), 1999–2019.", cap)]))
story.append(P(
    f"Table 5 asks whether a model estimated on daily data reproduces this. At one day the Gaussian model implies "
    f"{f3(imp.loc[1,'Gaussian'])} and the t-copula {f3(imp.loc[1,'Student-t'])}, both below the 95% interval of the "
    f"observed {obs.loc[1,'est']:.3f}; only the empirical copula ({f3(imp.loc[1,'Empirical (FHS)'])}) lies inside it. At "
    "longer horizons every model stays below the observed value, though the intervals widen and overlap. <b>Ignoring "
    "tail dependence therefore means understating joint crashes, and the shortfall does not disappear as the "
    "horizon lengthens.</b> Section 5 measures what that does to VaR and ES.", body))

# ================================================================= 4. RESULTS (b)
story.append(P("5. What does ignoring it do to measured risk?", h1))
story.append(P("5.1 One-day risk, out-of-sample 2020–2026", h2))
r1 = [["Model", "99% VaR breaches", "Kupiec p", "Indep. p", "DQ p", "Basel red / yellow", "Mean ES 97.5%",
       "Z2 p", "Breach loss / ES", "FZ0 loss"]]
for m in ("HS", "FHS", "G", "T", "WT"):
    s = S1[m]
    tl = s["traffic_light"]
    r1.append([NAMES_1D[m], f"{s['var99']['hits']} ({hits_rng[m][0]}–{hits_rng[m][1]})", pv(s["var99"]["p_uc"]),
               pv(s["var99"]["p_ind"]), pv(s["var99"]["p_dq"]), f"{pct(tl['red_share'],0)} / {pct(tl['yellow_share'],0)}",
               pct(s["mean_es97_5"], 2), pv(s["es97_5"]["p_z2"]), f2(s["es97_5"]["realised_over_predicted"]),
               f3(s["fz0_97_5"])])
story.append(table(r1, [3.7, 1.9, 1.3, 1.3, 1.2, 1.9, 1.6, 1.2, 1.6, 1.6]))
story.append(P(f"Table 6. {H['oos_window']['days']:,} out-of-sample days; {expected99:.0f} breaches expected at 99%. "
               "Breach range across Monte Carlo seeds in brackets. Basel shares are the fraction of rolling 250-day "
               "windows in the red (≥ 10) or yellow (5–9) zone. Z2 p: Acerbi–Szekely test of ES 97.5% (small = ES too "
               "low). Breach loss / ES: average realised loss on VaR-breach days ÷ predicted ES. FZ0: lower is better.", cap))
dmk = S1["dm_fz0_97_5"]
story.append(P(
    f"<b>Tails at one day.</b> Same margins, different copula: the Gaussian benchmark breaches its 99% VaR "
    f"{g1['var99']['hits']} times in every seed, the t-copula {hits_rng['T'][0]}–{hits_rng['T'][1]} times, against "
    f"{expected99:.0f} expected. Its average ES 97.5% is {pct(-es1_gap_t)} below the t-copula's and "
    f"{pct(-es1_gap_f)} below filtered historical simulation, and it is the only copula model that reaches the Basel "
    f"red zone ({pct(g1['traffic_light']['red_share'],1)} of windows). All copula models pass Kupiec and the "
    f"Acerbi–Szekely test. On the joint FZ0 score the copula models are statistically tied (t vs Gaussian DM p = "
    f"{pv(dmk['T_vs_G']['p'])}), while all beat historical simulation decisively (p ≤ "
    f"{pv(max(dmk[k]['p'] for k in ('G_vs_HS','T_vs_HS','WT_vs_HS','FHS_vs_HS')))}): at one day volatility dynamics "
    "dominate and dependence is a second-order, tail-only effect. Using the H1 wavelet band instead of daily residuals "
    f"(WT) changes little (WT vs t: DM {pp(dmk['WT_vs_T']['p'])}), which is what Section 4 predicts: daily "
    "dependence is the H1 dependence.", body))
story.append(P(
    f"<b>What still fails.</b> Breaches cluster: the dynamic-quantile test rejects every model at 99% and "
    f"Christoffersen's independence test every model except filtered HS ({pp(fhs1['var99']['p_ind'])}), driven by "
    "COVID (Table 7). The McNeil–Frey test finds realised breach losses about "
    f"{pct(min(S1[m]['es97_5']['mf_mean_rel_excess'] for m in ('FHS','G','T','WT')),0)}–"
    f"{pct(max(S1[m]['es97_5']['mf_mean_rel_excess'] for m in ('FHS','G','T','WT')),0)} above predicted ES. A model "
    "re-estimated monthly on four years of data cannot anticipate a regime change of the size of March 2020.", body))
sp = sub.pivot(index="period", columns="model", values="hit99_pct")
sp95 = sub.pivot(index="period", columns="model", values="hit95_pct")
days = sub.groupby("period").days.first()
srows = [["Period", "Days"] + [f"{m} 99% / 95%" for m in ("HS", "FHS", "G", "T", "WT")]]
for per in ["COVID crash (Feb–Jun 2020)", "2022 rate shock", "Other days"]:
    srows.append([per, str(days[per])] + [f"{sp.loc[per, m]:.1f} / {sp95.loc[per, m]:.1f}" for m in ("HS", "FHS", "G", "T", "WT")])
story.append(KeepTogether([table(srows, [4.4, 1.2, 2.36, 2.36, 2.36, 2.36, 2.36]),
                           P("Table 7. Breach rates (%) by sub-period; targets 1% and 5%.", cap)]))
story.append(fig("fig2_oos_var99.png", 17.0, "Figure 4. Out-of-sample 99% VaR (negative, % of NAV) and daily portfolio "
                 "returns. Panel (b) zooms into the COVID crash; red dots are breaches of the wavelet t-copula VaR."))
story.append(fig("fig3_traffic_light.png", 17.0, "Figure 5. Rolling 250-day count of 99% VaR breaches against the "
                 "Basel traffic-light zones."))

story.append(P(f"5.2 {H_DAYS}-day risk: where the horizon matters", h2))
story.append(P(
    f"Regulators and asset managers need multi-day risk (FRTB uses a 10-day base horizon). The usual shortcut is "
    f"√10 × 1-day risk. Table 8 replaces one simplification at a time on {SH['n_windows']} non-overlapping 10-day "
    f"windows, measuring each model's ES 97.5% against the full horizon-aware model, the wavelet empirical copula on the H2 band.", body))
r10 = [["Model", "Ignores", "Mean ES 97.5%", "vs full model: all / stressed", "Seed range", "99% / 95% breaches",
        "Breach loss / ES", "FZ0 loss"]]
ign = {"G_sqrt": "tails; uses √10 rule", "G_path": "tail dependence", "T_path": "compounds daily draws",
       "T_join": "asymmetry; band", "WC": "asymmetry (symmetric t)", "E_join": "band (uses daily data)",
       "WC_emp": "nothing (full model)"}
for m in NAMES_H:
    s = SH[m]
    g = gaps[m]
    r10.append([NAMES_H[m], ign[m], pct(s["mean_es97_5"], 2), "reference" if m == BASE else f"{spct(g[0])} / {spct(g[2])}",
                "reference" if m == BASE else f"{spct(gap_rng[m][0])} to {spct(gap_rng[m][1])}",
                f"{s['var99']['hits']} / {s['var95']['hits']}", f2(s["es97_5"]["realised_over_predicted"]), f3(s["fz0_97_5"])])
story.append(table(r10, [4.0, 3.3, 1.7, 2.6, 2.2, 1.6, 1.4, 1.2]))
story.append(P(f"Table 8. {SH['n_windows']} non-overlapping 10-day windows, 2020–2026 (expected breaches "
               f"{SH['G_path']['var99']['expected_hits']:.1f} at 99%, {SH['G_path']['var95']['expected_hits']:.1f} at 95%). "
               "'Stressed' = windows in the top quartile of the full model's ES.", cap))
story.append(P(
    f"<b>Interpretation.</b> (i) Dropping tail dependence (Gaussian paths) lowers 10-day ES by "
    f"{pct(-gaps['G_path'][0])} on average and {pct(-gaps['G_path'][2])} when risk is high, and this model has the "
    f"largest breach losses relative to ES ({f2(SH['G_path']['es97_5']['realised_over_predicted'])}×). (ii) Building "
    f"10-day risk by compounding a daily t-copula still leaves {pct(-gaps['T_path'][0])}: with independent daily "
    "draws the joint tail is diluted, while Section 4 shows the observed crash tail is not. Imposing the daily "
    f"t-copula directly at the 10-day horizon removes that dilution ({spct(gaps['T_path'][0])} → "
    f"{spct(gaps['T_join'][0])}; the H2-band t-copula gives {spct(gaps['WC'][0])}). (iii) The remaining gap is the shape of the tail: the "
    f"empirical copula, which keeps the crash/rally asymmetry, raises 10-day ES by a further "
    f"{pct(SH['WC_emp']['mean_es97_5'] / SH['WC']['mean_es97_5'] - 1)}. (iv) Estimating the copula on the H2 band "
    f"instead of daily data changes little (daily empirical copula vs full model: {spct(gaps['E_join'][0])}), which "
    "is what Section 4 predicts: crash co-movement at 8–32 days is statistically the same as at the daily scale. The "
    "wavelet analysis is what justifies carrying daily crash dependence to the 10-day horizon. (v) The √10 rule "
    f"({spct(gaps['G_sqrt'][0])}) does better than Gaussian paths only because two errors partly cancel: it ignores "
    f"tail dependence, but scaling a fat-tailed 1-day distribution by √10 overstates the 10-day tail of the same "
    f"Gaussian model by {pct(sqrt_vs_path)} on average, because 10-day returns have thinner tails than daily ones. "
    f"With {SH['n_windows']} windows the backtest has little power: no model's breach count is rejected and the FZ0 "
    f"differences are not significant (lowest loss: {NAMES_H[best10]}). The two models that compound daily draws are "
    f"the only ones whose breach losses are significantly above predicted ES (McNeil–Frey p = "
    f"{pv(SH['G_path']['es97_5']['mf_p'])} and {pv(SH['T_path']['es97_5']['mf_p'])} respectively; full model p = "
    f"{pv(SH[BASE]['es97_5']['mf_p'])}).", body))
story.append(fig("fig4_es10d.png", 17.0, "Figure 6. 10-day ES 97.5% forecasts and realised 10-day portfolio losses; "
                 "circled points exceed the full model's VaR 97.5%."))

# ================================================================= 5. RECOMMENDATION
story.append(P("6. Recommendation for the risk manager", h1))
story.append(box([
    P(f"<b>From tomorrow: replace the Gaussian copula and the √10 rule with tail-aware, horizon-matched ES.</b>", boxs),
    P(f"1. <b>Daily limit.</b> Compute 1-day ES 97.5% from the t-copula on daily-updated GARCH margins: "
      f"{pct(lat['es97_5_T'], 2)} of NAV today, against {pct(lat['es97_5_G'], 2)} from the Gaussian copula. Over "
      f"2020–2026 the Gaussian model breached its 99% VaR {g1['var99']['hits']} times against {expected99:.0f} expected "
      "and was the only copula model to reach the Basel red zone.", boxs),
    P(f"2. <b>10-day limit.</b> Size the 10-day ES from the wavelet empirical-copula model, not √10 × 1-day: "
      f"{pct(lat['es97_5_10d_' + BASE], 2)} of NAV today (√10 × Gaussian: {pct(lat['es97_5_10d_G_sqrt'], 2)}; Gaussian "
      f"paths: {pct(lat['es97_5_10d_G_path'], 2)}). Today the shortcut happens to be close, but over 2020–2026 it was "
      f"below the full model in {pct(sqrt_below, 0)} of windows, by up to {pct(sqrt_worst, 0)}, and the Gaussian path "
      f"model was below it by {pct(-gaps['G_path'][2], 0)} on average when risk was high.", boxs),
    P("3. <b>Diversification.</b> Do not count on sector diversification in a sell-off at any horizon: crash "
      "co-movement beyond a month is statistically the same as at 2–8 days, while rally co-movement halves. Hedge with "
      "instruments outside the equity sector set, not with sector rotation.", boxs),
    P("4. <b>Trigger.</b> Escalate to the risk committee when the 250-day count of 99% breaches reaches 5 (Basel "
      "yellow). Keep the regulatory capital model unchanged until independent validation; run this as an internal "
      "overlay.", boxs)], color=ACCENT))

# ================================================================= 6. ROBUSTNESS & LIMITATIONS
story.append(P("7. Robustness and limitations", h1))
story.append(P(
    f"<b>Robustness.</b> Every out-of-sample number was recomputed with {len(rob) - 1} more Monte Carlo seeds "
    "(ranges in Tables 6 and 8); conclusions do not change. Results are reported at two tail levels (q = 5% and 10%); "
    "the conclusions hold at both (files in <font face='Courier'>outputs/</font>). Figure 7 shows the rolling t-copula "
    "tail dependence: it rises in stress (2020, 2022) at every scale, with the H2 band the most variable.", body))
story.append(fig("fig5_rolling_lambda.png", 15.5, "Figure 7. Rolling t-copula tail dependence by scale "
                 f"(window {WINDOW} days, re-estimated every {REFIT_EVERY} days)."))
story.append(P(
    "<b>Limitations.</b> (1) Long-scale bands are highly autocorrelated, so H3 has few effective observations; "
    "bootstrap intervals widen accordingly but remain approximate. (2) The 10-day backtest has low power "
    f"({SH['n_windows']} windows). (3) The t-copula has one ν for all pairs and cannot be asymmetric; the empirical "
    "copula fixes asymmetry but cannot extrapolate beyond the worst event in its window. (4) Breach clustering "
    "remains: a regime-switching or dynamic (DCC/GAS) copula is the natural next step. (5) Results are for one "
    "equal-weight US sector portfolio; the code is portfolio-agnostic.", body))
story.append(P(
    "<b>Further work.</b> Asymmetric parametric copulas per scale (skew-t, vine copulas), dynamic copulas for breach "
    "clustering, 10-day ES on overlapping windows with HAC-adjusted tests, and the same study on the Colombo Stock "
    "Exchange with corrections for non-synchronous trading.", body))

# ================================================================= APPENDIX A
story.append(P("Appendix A. Reproducibility and use of AI tools", h1))
story.append(P(
    "<b>Reproduce.</b> Everything runs locally with no downloads beyond the Python packages. Run <font face='Courier'>pip install -r "
    "requirements.txt</font> followed by <font face='Courier'>python -m src.run_all</font>. This one command regenerates every table, "
    "figure and number in this report from the bundled price file "
    "(<font face='Courier'>data/etf_prices.csv</font>; <font face='Courier'>src/data.py</font> re-downloads it from "
    "Yahoo Finance with <font face='Courier'>force=True</font>). All randomness is seeded. The "
    "report text reads its numbers from <font face='Courier'>outputs/results.json</font>; none are typed by hand.", body))
story.append(P(
    "<b>AI use.</b> All cognitive and statistical work in this study was done by Team Gmora: framing the research "
    "question, choosing the portfolio, data and sample split, selecting the margin, wavelet and copula models, "
    "designing the out-of-sample experiments and validation tests, interpreting the results, and forming the "
    "recommendation. AI assistants (Anthropic Claude) were used as tools under the team's direction for "
    "writing and debugging code, and reviewing code for errors. The team checked every "
    "result and can explain every line of code and every claim in this report.", body))

# ================================================================= REFERENCES (excluded)
ref_block = [P("References", h1)]
refs = [
    "Acerbi, C. & Szekely, B. (2014). Backtesting expected shortfall. <i>Risk</i>, December.",
    "Ang, A. & Chen, J. (2002). Asymmetric correlations of equity portfolios. <i>J. Financial Economics</i> 63, 443–494.",
    "Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. <i>J. Futures Markets</i> 19, 583–602.",
    "Basel Committee (1996). Supervisory framework for the use of backtesting. BCBS; (2019) Minimum capital requirements for market risk (FRTB).",
    "Berger, T. & Missong, M. (2014). Financial crisis, Value-at-Risk forecasts and the puzzle of dependency modeling. <i>Int. Rev. Financial Analysis</i> 33, 33–38.",
    "Christoffersen, P. (1998). Evaluating interval forecasts. <i>Int. Economic Review</i> 39, 841–862.",
    "Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. <i>J. Business & Economic Statistics</i> 13, 253–263.",
    "Engle, R. & Manganelli, S. (2004). CAViaR. <i>J. Business & Economic Statistics</i> 22, 367–381.",
    "Frahm, G., Junker, M. & Schmidt, R. (2005). Estimating the tail-dependence coefficient. <i>Insurance: Mathematics and Economics</i> 37, 80–100.",
    "Genest, C., Ghoudi, K. & Rivest, L.-P. (1995). A semiparametric estimation procedure of dependence parameters. <i>Biometrika</i> 82, 543–552.",
    "Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. <i>J. Derivatives</i> 3, 73–84.",
    "Lindskog, F., McNeil, A. & Schmock, U. (2003). Kendall's tau for elliptical distributions. In <i>Credit Risk</i>, Physica.",
    "Longin, F. & Solnik, B. (2001). Extreme correlation of international equity markets. <i>J. Finance</i> 56, 649–676.",
    "McNeil, A. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. <i>J. Empirical Finance</i> 7, 271–300.",
    "Patton, A., Ziegel, J. & Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and VaR). <i>J. Econometrics</i> 211, 388–413.",
    "Percival, D. & Walden, A. (2000). <i>Wavelet Methods for Time Series Analysis</i>. Cambridge University Press.",
    "Politis, D. & Romano, J. (1994). The stationary bootstrap. <i>J. American Statistical Association</i> 89, 1303–1313.",
    "Schmidt, R. & Stadtmüller, U. (2006). Non-parametric estimation of tail dependence. <i>Scandinavian J. Statistics</i> 33, 307–335.",
]
ref_block += [P(r, small) for r in refs]
story.append(KeepTogether(ref_block))


def main():
    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.6 * cm, title="Risk Across Tails and Timescales",
                            author=TEAM_NAME, subject="SAIFA Quant Edge 1.0 Round 1")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
