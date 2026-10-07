# Quant research lead: how the literature handles multi-horizon tail dependence, and how this repo compares

## 0. Bottom line

The pipeline order is the right one: GARCH-t margins, then a wavelet decomposition of the standardised residuals, then a copula per band, then a VaR/ES backtest. It is the standard pipeline in Cai et al. 2020, Shahzad et al. 2016 and Jammazi & Reboredo 2016. The implementation, though, produces wrong headline numbers. I re-estimated on the repo's own data (`data/etf_prices.csv`, 1999-01-05 to 2026-10-06, n=6981) with scratch scripts that did not touch the repo, and found the following:

| Report claim | What correct estimation gives |
|---|---|
| "Clayton dominates AIC on every band; λL ≈ 0.66–0.69" | **This is an artefact.** The proper 7-dimensional maximum-likelihood fit on the H1 band (≤2019) gives: Gaussian log-likelihood (LL) 15296 (AIC −30551); **t-copula ν=7.43, LL 16238 (AIC −32432, the winner)**; Clayton θ=0.80, LL 8508 (AIC −17013, worst by far). The t-copula's λ is about 0.19 at the average ρ. |
| "Student ν = 30" on every band | ν sits at the top of the search grid because the likelihood formula is wrong. Correct ν is 7–8 at H1. |
| "Tail dependence barely changes with horizon" | **It does change.** t-copula on the GARCH-residual bands, sectors only, boundary values dropped: H1 ν=8.0, λ=0.130; H2 ν=9.2, λ=0.107; **H3 ν=19.7, λ=0.015**. On non-overlapping raw returns: 1-day ν=4.0, λ=0.27; 5-day λ=0.22; 10-day λ=0.21; 20-day λ=0.18. |
| Lower vs upper tail symmetry (no claim in the report) | Empirical tail dependence on non-overlapping k-day returns, q=0.10: lower tail is flat (0.446 at k=1, 0.453 at k=20, 0.478 at k=60), upper tail falls (0.410 to 0.318 to 0.294). **Asymmetry grows with horizon.** Longer horizons keep the crash co-movement but lose the rally co-movement. This is the Longin-Solnik / Ang-Chen pattern at longer timescales. |
| The H1 wavelet-copula fixes VaR coverage | The backtest failures (Christoffersen independence p≈1e-6) come mainly from **volatility frozen for 120 days**, not from the copula. A plain univariate filtered historical simulation (FHS) on the portfolio, with daily GARCH volatility updates (750-day window, refit every 20 days), gives **95%: 4.83% hits, Kupiec p=0.74, independence p=0.31; 99%: 0.82%, p=0.45, independence p=0.10.** It beats both repo models. A frozen-volatility GARCH-t reproduces the repo's failure pattern: 6.89% and 2.30% hits, independence rejected. |
| Wavelet bands are "MRA H1=D1+D2…" | They are sums of wavelet *coefficients*, not multiresolution-analysis (MRA) details. H1+H2+H3 does not rebuild the series: on XLF, corr(x, H1+H2+H3) = **−0.305**. |
| Square-root-of-time understates risk | Unconditional, the √h rule *overstates* the portfolio's 97.5% ES a little: empirical/√h ratio is 0.97 at 5 days, 0.89 at 10 days, 0.95 at 20 days. The story has to be conditional and dependence-driven, not "√h always understates". |

One more general finding: raw returns show much more tail dependence (ν=4) than GARCH-filtered residuals (ν≈7.6). A large part of measured "tail dependence" is shared volatility clustering. McNeil-Frey style filtering is essential, and this is a good teaching point for the report.

## 1. Code defects, with file:line (method relevance only)

1. **`src/run_all.py:50`** sets `QUANT_EDGE_FAST` to default `"1"`, so refits always happen every 120 days. **`src/risk.py:34`** uses `f.last_vol` from the end of the window, so VaR and ES stay constant for 120 days (see `outputs/oos_forecasts.csv`). This is the main cause of hit clustering. The locked design called for refits every 20 days plus a daily one-step GARCH update.
2. **`src/copulas_fit.py:104-125`** (Clayton):
   - The "log-likelihood" is a *sum of 21 pairwise* composite likelihoods, each at its own θ.
   - AIC uses k=1, while the stored θ is the mean of the pairwise values.
   - That is compared against a full 7-dimensional Gaussian likelihood with k=21. The comparison is invalid; composite likelihoods need CL-AIC (Varin & Vidoni 2005, *Biometrika*).
   - The simulation at lines 159-162 is an exchangeable Clayton with one θ, which forces equal tail dependence on every pair.
3. **`src/copulas_fit.py:76-90`** (Student-t): the code computes a multivariate-normal log-density of t-scores minus t log-densities. That is not the t-copula density. `scale` on line 83 is unused. The ν grid is capped at 30, and the fit always lands there (`outputs/band_copula_results.csv`). Tail dependence is computed from the *average* ρ (line 91).
4. **`src/modwt.py:48-54`**: H-bands are sums of MODWT coefficients W_j, not MRA details D_j. Circular filtering contaminates the first L_j−1 values, where L_j = (2^j−1)(L−1)+1. For db2 at j=6 that is **189 of 750 window values (25%)**, wrapped from the end of the window. This is acceptable for cross-sectional band correlation only if the boundary values are dropped, as Percival & Walden 2000 (ch. 5) and Cornish, Bretherton & Percival 2006 do.
5. **Conceptual mismatch** (`run_all.py:114-122`): only H1 is used for risk. Its copula drives *daily* margins, and H2/H3 never reach any risk number. So the second half of the question, "what does ignoring this do to measured risk", is never answered at any horizon other than 1 day.
6. **ES "test"** (`backtest.py:93-105`): a ratio of means on hit days is not a test. The realized-to-predicted ratio is about 1.52–1.58 for the wavelet-copula, so its ES also under-predicts. The report's line "Gaussian understates ES by 28%" compares two models that both fail.
7. **Portfolio**: SPY is close to a linear combination of the sectors, so the 7-asset equal-weight portfolio double-counts market beta. It also inflates average correlation and makes an exchangeable copula even less suitable. Drop SPY, or use it only as a factor or hedge instrument.
8. The locked design in `reports/Wavelet copula horizon risk.md` asked for `copulae` maximum-likelihood fits, N=20,000 simulations, refits every 20 days and a McNeil-Frey ES test. The implementation diverged on all four, and the report does not say so.

## 2. Literature map: what to cite and what each paper establishes

**Copula fundamentals**
- **Sklar (1959)**, Publ. Inst. Statist. Univ. Paris 8. Any joint distribution splits into its margins plus a copula.
- **Nelsen (2006)**, *An Introduction to Copulas*, Springer; **Joe (2014)**, *Dependence Modeling with Copulas*, CRC. Families and their tail behaviour:
  - Gaussian has λ=0.
  - t is symmetric, with λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) (Demarta & McNeil 2005, *Int. Stat. Rev.*).
  - Clayton has λL = 2^(−1/θ); Gumbel has λU = 2 − 2^(1/θ); survival/rotated versions flip the tail.
  - BB1 and BB7 have both tails.
  - Mixtures: Hu (2006), *Applied Financial Economics*.
- **Estimation**: IFM (Joe & Xu 1996); canonical maximum likelihood with rank pseudo-observations (Genest, Ghoudi & Rivest 1995, *Biometrika*). The repo's rank PIT is CML, which is fine.
- **Pitfalls**: Embrechts, McNeil & Straumann (2002), "Correlation and dependence in risk management: properties and pitfalls". **Frahm, Junker & Schmidt (2005), *IME* 37:80-100**: parametric λ estimates depend heavily on the family chosen, so report nonparametric estimates (**Schmidt & Stadtmüller 2006, *Scand. J. Stat.* 33:307-335**) with bootstrap confidence intervals. https://wisostat.uni-koeln.de/fileadmin/sites/statistik/pdf_publikationen/FrahmJunkerSchmidt.pdf

**Filtering, EVT and risk measures**
- **McNeil & Frey (2000), *J. Empirical Finance* 7:271-300**: GARCH filter, then EVT/POT on the residuals, plus a bootstrap ES backtest.
- **Barone-Adesi, Giannopoulos & Vosper (1999), *J. Futures Markets***: filtered historical simulation (FHS).
- **Kuester, Mittnik & Paolella (2006), *J. Financial Econometrics***: across many VaR methods, GARCH-filtered EVT/FHS approaches rank best.
- **Coherence and ES**: Artzner et al. (1999), *Math. Finance*; Acerbi & Tasche (2002), *JBF*. Elicitability: Gneiting (2011), *JASA* (ES is not elicitable on its own); **Fissler & Ziegel (2016), *Ann. Stat.*** (VaR and ES jointly are).
- **ES model comparison**: Patton, Ziegel & Chen (2019), *J. Econometrics* 211:388-413, FZ0 loss. https://arxiv.org/abs/1707.05108
- **Backtesting**:
  - Kupiec (1995), *J. Derivatives*; Christoffersen (1998), *IER*; BCBS (1996) traffic light.
  - Acerbi & Szekely (2014), MSCI, Z1/Z2 ES tests.
  - Nolde & Ziegel (2017), *Ann. Appl. Stat.* 11:1833-1874, comparative backtests. https://boris.unibe.ch/108729/1/AOAS1041.pdf
- **FRTB (BCBS MAR33)**: ES at 97.5%, 10-day base horizon. Liquidity horizons of 10/20/40/60/120 days are layered by √(LH_j − LH_{j−1})/10 scaling. Large-cap equity price is 10 days, small-cap 20. ES is calibrated to a stressed 12-month period. https://www.bis.org/committees/bcbs/basel-framework/81333/chapter.pdf
- **Time aggregation**:
  - Drost & Nijman (1993), *Econometrica* 61:909-927: temporal aggregation of GARCH.
  - Diebold, Hickman, Inoue & Schuermann (1998), *Risk*: "scaling by √h is worse than you think".
  - **Danielsson & Zigrand (2006), *JBF* 30:2701-2713**: √t understates risk under jumps, and more so at longer horizons and higher confidence. https://www.riskresearch.org/papers/DanielssonZigrand2006/

**Asymmetric and extreme dependence**
- **Longin & Solnik (2001), *J. Finance* 56:649-676**: correlation rises in bear-market extremes, not in bull-market ones.
- **Ang & Chen (2002), *JFE* 63:443-494**: exceedance correlations.
- **Hong, Tu & Zhou (2007), *RFS* 20:1547-1581**: a formal test of that asymmetry.
- These support the finding that λL stays while λU falls with horizon.

**Wavelets**
- **Percival & Walden (2000)**, *Wavelet Methods for Time Series Analysis*, CUP; **Gençay, Selçuk & Whitcher (2002)**, *An Introduction to Wavelets and Other Filtering Methods in Finance and Economics*, Academic Press.
  - MODWT over DWT: it is shift-invariant, works for any sample length, and splits variance across scales.
  - LA(8) is the usual filter because its near-linear phase lets coefficients be time-aligned.
  - Boundary coefficients should be dropped.
  - Wavelet variance and correlation with confidence intervals: Whitcher, Guttorp & Percival (2000), *JGR*.
- **Gençay, Selçuk & Whitcher (2005), *JIMF* 24:55-70**, "Multiscale systematic risk": the link between beta and portfolio return strengthens at longer scales. https://ideas.repec.org/a/eee/jimfin/v24y2005i1p55-70.html
- **Kim & In (2005), *J. Empirical Finance* 12:435-444**: the sign of the stock-inflation relation changes across scales.
- **Rua & Nunes (2009), *J. Empirical Finance* 16:632-639**: international stock comovement is stronger at low frequencies. https://www.bportugal.pt/sites/default/files/anexos/papers/wp200904.pdf
- **Fernández-Macho (2012), *Physica A* 391:1097-1104**: wavelet *multiple* correlation, a single scale-by-scale statistic for a whole set of assets. This is directly usable for the 6 sectors. https://addi.ehu.es/handle/10810/5575
- **Forecasting caution**: wavelet decompositions use future data at the series edge, which leaks into forecasts unless they are computed causally (Zhang, Gençay & Yazgan 2017, *Economics Letters* 158:41-46). Scale-specific predictability: Bandi, Perron, Tamoni & Tebaldi (2019), *J. Econometrics* 208:120-140.

**Wavelet + copula / multiscale risk**
- **Berger (2015), *Physica A* 436:338-350**: contagion at different time scales. **Berger (2016), *J. Risk* 18:53-77**: wavelet decomposition and portfolio management.
- **Berger & Gençay (2018), *JEDC* 92:30-46**: daily VaR is driven mainly by *short-run* (wavelet scales 1–2) volatility. This is the honest rationale for using H1 for 1-day VaR, provided volatility is updated daily. https://ideas.repec.org/a/eee/dyncon/v92y2018icp30-46.html
- **Berger & Missong (2014), *IRFA* 33:33-38**: copula choice matters less for VaR than the GARCH/EVT margins.
- **Conlon, Cotter & Gençay (2016), *Eur. J. Finance* 22:1534-1560**: optimal hedge ratios depend on the hedging horizon. https://ideas.repec.org/p/ucd/wpaper/201218.html
- Boubaker & Sghaier (2013), *JBF* 37:361-377 (copula portfolio optimisation with long memory).
- Applied papers already in the repo notes: Cai et al. 2020 (MODWT plus time-varying symmetrised Joe-Clayton copula); Jammazi & Reboredo 2016, *Energy*; Shahzad et al. 2016, *Physica A*; Aloui & Jammazi 2015; Alqaralleh & Canepa 2021, *JRFM*.
- Note: these papers mostly report scale-by-scale dependence or hedging effectiveness. **None gives a standard recipe for combining scales into one VaR.** The team can turn that into a clear design choice: diagnose per band, measure risk at matched horizons.

**Dynamic, high-dimensional and industry practice**
- Patton (2006), *IER* 47:527-556: time-varying SJC copula.
- **Christoffersen, Errunza, Jacobs & Langlois (2012), *RFS* 25:3711-3751**: dynamic asymmetric copulas show diversification benefits shrinking over time. https://ideas.repec.org/a/oup/rfinst/v25y2012i12p3711-3751.html
- Creal, Koopman & Lucas (2013), *JAE*: GAS / score-driven copulas.
- **Oh & Patton (2017), *JBES* 35:139-154**: factor copulas, with SPY as the natural common factor.
- Aas, Czado, Frigessi & Bakken (2009), *IME* 44:182-198 (pair-copula/vine constructions); Dißmann et al. (2013), *CSDA* (vine selection); Brechmann & Czado (2013), *Stat. & Risk Modeling* (vine VaR on the Euro Stoxx 50).
- Engle (2002), *JBES* (DCC); Cappiello, Engle & Sheppard (2006), *JFEc* (asymmetric DCC).
- Regime-switching copulas: Okimoto (2008), *JFQA*; Chollete, Heinen & Valdesogo (2009), *JFEc*.
- **Industry**:
  - Banks run HS/FHS ES at 97.5% / 10 days with liquidity-horizon cascading and a stressed period (MAR33). P&L attribution and traffic-light backtesting sit on 1-day VaR at 99% and 97.5% (MAR32).
  - RiskMetrics (1996) used EWMA with √h scaling. RiskMetrics 2006 (Zumbach) moved to long-memory volatility with horizon-dependent scaling.
  - Asset-manager factor models (MSCI Barra, Bloomberg PORT) use horizon-specific covariance with Newey-West adjustment and volatility-regime adjustment. **No mainstream vendor estimates scale-specific copulas.** That gap is a legitimate "creativity" angle.

**ML and modern methods**
- CAViaR (Engle & Manganelli 2004, *JBES*); joint VaR-ES models (Patton-Ziegel-Chen 2019).
- Conformal VaR calibration: arXiv 2507.05470 (temporal conformal prediction), arXiv 2602.03903 (regime-weighted conformal calibration).
- Normalizing-flow / neural copulas (e.g., Wiese, Knobloch & Korn 2019, arXiv:1907.03361).

## 3. Feasibility triage

**Today (hours):**
- Correct Gaussian and t-copula maximum-likelihood fits.
- Daily volatility updates.
- FHS benchmark.
- Nonparametric λL/λU by band and by horizon, with block-bootstrap confidence intervals.
- Exceedance-correlation chart.
- 10-day ES: √10 scaling vs simulated paths.
- FZ0 loss with Diebold-Mariano test; McNeil-Frey or Acerbi-Szekely ES test.

**Round 2 and later:**
- Pairwise rotated-Clayton/Gumbel or BB7, or an R-vine via pyvinecopulib.
- GAS/DCC dynamic copula.
- Skew-t copula or a factor copula with SPY as the factor.
- Regime switching.
- Conformal calibration layer.

**Not worth it now:** deep copulas or normalizing flows. They are hard to explain live, and there is no evidence they help at 6 dimensions with daily data.

## 4. Seven upgrades, ranked by impact per hour (deadline today)

1. **Update volatility daily** (about 1 hour). Refit every 20 days and run the GARCH one-step recursion each day (σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}), feeding σ_t into `invert_t_margin`. Set the default to non-FAST. *Basis:* McNeil & Frey 2000; Berger & Gençay 2018. *Evidence:* frozen volatility gives 6.89%/2.30% hits with clustering; daily updating removes the clustering (independence p 0.18 and 0.26).
2. **Replace the copula likelihoods** (1–2 hours). Gaussian stays. For the t-copula: R from Kendall's τ via sin(πτ/2), then a 1-D ν search with `scipy.stats.multivariate_t`; this is about 15 lines (see `exp2.py`). Drop 7-dimensional exchangeable Clayton from AIC selection, or use a valid composite-likelihood AIC. Report λ from ν and *pairwise* ρ. *Basis:* Demarta & McNeil 2005; Genest et al. 1995; Varin & Vidoni 2005. The headline changes to: "t wins; tail dependence falls with scale, λ 0.13 → 0.11 → 0.02."
3. **Answer the asymmetry part with nonparametric tail dependence** (1–2 hours). Compute λL and λU (Schmidt-Stadtmüller, q=0.05 and 0.10) per band and on non-overlapping 1/5/10/20/60-day returns, with stationary-block-bootstrap confidence intervals. Add an Ang-Chen exceedance-correlation plot. *Basis:* Longin & Solnik 2001; Ang & Chen 2002; Hong, Tu & Zhou 2007; Frahm et al. 2005. This gives the "simple idea explained well": crash co-movement persists across horizons while rally co-movement fades, so a symmetric (Gaussian or t) model fitted to daily data understates long-horizon downside dependence relative to upside.
4. **Quantify "what ignoring horizon does to measured risk"** (2–3 hours). Compute 10-day 97.5% ES (FRTB base horizon) three ways:
   - (a) 1-day model × √10, the industry shortcut;
   - (b) 10-day Monte Carlo paths with GARCH dynamics and the daily copula (FHS-style);
   - (c) a horizon-matched copula, i.e. dependence fitted on the H2 band (8–32 days) or on 10-day non-overlapping blocks.

   Backtest on non-overlapping 10-day OOS windows (about 170; low power, and say so). *Basis:* Danielsson & Zigrand 2006; Drost & Nijman 1993; MAR33. This is the missing half of the brief.
5. **Fix the wavelet step** (about 1 hour). Either rename the bands as wavelet-coefficient bands (scale-by-scale dependence of W_j, as in Whitcher et al. and Fernández-Macho), or produce real MRA details by inverse MODWT. Drop the L_j−1 boundary values. Switch to LA8. Run the band diagnostic on the long in-sample period, because J=6 on a 750-day window leaves about 12 effective observations per level-6 cycle. Optionally add wavelet multiple correlation by scale as one figure. *Basis:* Percival & Walden 2000; Gençay et al. 2002; Zhang, Gençay & Yazgan 2017.
6. **Use a proper benchmark and proper tests** (1–2 hours). Add a univariate FHS on the portfolio as the "simple benchmark"; it is what banks run, and it currently beats the model. Add the McNeil-Frey bootstrap or Acerbi-Szekely Z2 ES test, FZ0 average loss with a Diebold-Mariano test between models, and a Basel traffic light per 250 days. *Basis:* Kuester et al. 2006; Patton, Ziegel & Chen 2019; Nolde & Ziegel 2017. Reframe the claim: the copula's value is in multi-horizon and stress ES, not in beating FHS on 1-day 95% VaR.
7. **Portfolio hygiene** (15 minutes plus a rerun). Drop SPY, or keep it only as a factor or hedge in the recommendation. Justify equal weight. Then correct every report claim affected: λL 0.68, "Clayton dominates", the 28% ES figure, "MRA", and refit every 120 days. Disclose the gaps between the locked design and the implementation. The brief says unreproducible or unsupported claims are penalised and the team may be quizzed live.

**Resulting recommendation template:** "Set the 10-day ES limit from horizon-matched downside dependence, not from √10 × 1-day Gaussian or t. Size index-put hedges using lower-tail co-crash probabilities measured at the 2–6-week scale, since the upside diversification seen in daily data does not carry over to monthly horizons." Fill in the numbers once upgrades 3–4 are rerun.

## Files

Scratch analysis scripts (none of the repo's files were changed):
- /tmp/claude-0/-home-user-quant-edge-round1/9008468d-2c52-5318-9ae3-7346ebbf91aa/scratchpad/exp.py — MRA reconstruction check, empirical λ by horizon, proper 7-dimensional likelihoods, √h ES check, FHS / frozen-volatility backtests
- /tmp/claude-0/-home-user-quant-edge-round1/9008468d-2c52-5318-9ae3-7346ebbf91aa/scratchpad/exp2.py — t-copula ν and λ by band and by aggregation horizon
- Output: /tmp/claude-0/-home-user-quant-edge-round1/9008468d-2c52-5318-9ae3-7346ebbf91aa/tasks/bwr4g317e.output

Repo files referenced:
- /home/user/quant-edge-round1/src/copulas_fit.py
- /home/user/quant-edge-round1/src/modwt.py
- /home/user/quant-edge-round1/src/risk.py
- /home/user/quant-edge-round1/src/run_all.py
- /home/user/quant-edge-round1/src/backtest.py
- /home/user/quant-edge-round1/outputs/band_copula_results.csv
- /home/user/quant-edge-round1/outputs/backtest_summary.json
- /home/user/quant-edge-round1/reports/Wavelet copula horizon risk.md