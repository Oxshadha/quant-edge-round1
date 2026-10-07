# Code and methodology review: quant-edge-round1, SAIFA Quant Edge R1

**Bottom line:** the report's main claim does not survive testing. The claim is "Clayton dominates at every horizon, λ_L≈0.66–0.69, the wavelet-copula hits 5.24% and the Gaussian fails at 7.00%". It comes from three bugs working together:
1. **Frozen volatility:** each VaR forecast is held unchanged for 120 trading days.
2. **Invalid AIC comparison:** the Clayton and Student-t log-likelihoods are computed wrongly, so AIC cannot rank them.
3. **The wavelet step does nothing to the forecast:** Clayton fitted without wavelets gives the same result.

When volatility is updated daily, the Gaussian benchmark passes and the Clayton model is too conservative. The pipeline runs, has no look-ahead and reproduces exactly, so the fixes are feasible today.

## What I ran (scratch only, repo untouched)
- **Setup:** I copied the repo to `/tmp/claude-0/-home-user-quant-edge-round1/9008468d-2c52-5318-9ae3-7346ebbf91aa/scratchpad/repo`.
- **`make reproduce`:** it worked. Install took 30s and the run took 8.6s, with exit 0 and a 4-page PDF.
- **Match to committed outputs:** differences are at most 1.5e-6, and hit counts are identical (89/40/119/54). `manager_recommendation.txt` is byte-identical.
- **`QUANT_EDGE_FAST=0`:** I also ran the 20-day refit mode.
- **My scripts:** in `.../scratchpad/ana/`.
  - `diag1.py` computes correct 7-dimensional copula log-likelihoods and checks the MODWT.
  - `ablate.py` runs the ablation backtests (stale vs daily volatility, with and without wavelets, t-copula, FHS = filtered historical simulation).
  - `emp_tail.py` computes empirical λ_L by band and by horizon, with block-bootstrap confidence intervals.

## CRITICAL

**C1. VaR is frozen for 120 days and uses the wrong sigma.**
- **Where:** `src/margins.py:34` and `src/run_all.py:121-127, 131-150`.
- **What happens:**
  - `last_vol` is σ_T, the volatility for the last in-sample day, not the one-step-ahead forecast σ_{T+1}.
  - The VaR from each refit is reused for `step` days.
- **Evidence:**
  - `oos_forecasts.csv` has only **15 distinct VaR values over 1,699 days**.
  - The COVID crash (Feb–Mar 2020) was forecast with GARCH fitted through 2019-12-31. The next refit was 2020-06-24.
  - This is why Christoffersen independence is rejected everywhere (p_ind 1.7e-6 to 1.9e-10).
  - It is also why realised loss on breach days is about 1.5× the predicted ES.
- **Re-run with the GARCH variance updated daily** (parameters fixed between refits, same 120-day copula refits):

| Model | 95% hit | Kupiec p | 99% hit | Kupiec p |
|---|---|---|---|---|
| Gaussian benchmark | 5.53% | 0.32 | 1.35% | 0.16 |
| t-copula (raw residuals) | 5.77% | — | 1.29% | — |
| Repo's wavelet-copula (Clayton) | **3.47%** | **0.002** (too conservative) | 0.65% | — |

  - Loss/ES ratios fall to about 1.0–1.1.
- **Result:** the headline reverses.
- **Fix:** store ω, α, β, μ and ν from each fit. Run σ²_{t+1} = ω + α(r_t−μ)² + βσ²_t every day. Keep the simulated standardised shocks Z for the refit block and rescale them daily. About 30 lines; **2–3h**.

**C2. The AIC model selection is invalid.**
- **Where:** `src/copulas_fit.py:104-125` (Clayton) and `76-90` (Student-t).
- **Clayton:** the log-likelihood is the sum of 21 pairwise bivariate log-likelihoods. Each pair uses its own θ, but the model then uses mean θ and k=1. That number is compared with a full 7-dimensional Gaussian likelihood.
- **Student-t:** the "log-likelihood" is a multivariate normal logpdf of t-scores minus the t marginals. That is not the t-copula density. `scale` at line 83 is unused, and ν always ends at the grid maximum of 30 (see `band_copula_results.csv`).
- **Correct 7-dimensional log-likelihoods on in-sample data:**

| Data | Gaussian | Student-t | Exchangeable Clayton (MLE) |
|---|---|---|---|
| Raw residuals | 15,419 | **16,388** (ν=8) | 8,456 (θ=0.80, λ_L=0.42) |
| H1 band | 15,296 | **16,197** (ν=8) | 8,508 |
| H2 band | 17,229 | **17,931** (ν=8) | — |
| H3 band | 16,514 | **16,706** (ν=16) | 7,618 |

- **Result:** the t-copula wins every band and Clayton is by far the worst. "Clayton dominates AIC on every band" and λ_L 0.66–0.69 are artefacts.
- **What the data actually show:**
  - The t-copula gives λ_L ≈ 0.17 (H1, H2) and ≈ 0.05 (H3), using average ρ.
  - Empirical λ_L at q=0.05 is 0.42 (raw), 0.39 (H1), 0.40 (H2) and 0.37 (H3).
- **Fix:** use `scipy.stats.multivariate_t.logpdf` minus Σ `t.logpdf` and optimise ν. Use the closed-form d-dimensional Clayton density: Π(1+kθ) · Πu^{-(1+θ)} · (Σu^{-θ}−d+1)^{-(d+1/θ)}. Or drop Clayton from the AIC contest. **1.5–2h.**

**C3. The wavelet step does not drive the forecast, and the benchmark mixes two changes.**
- **Where:** `src/run_all.py:113-122`.
- **What happens:** the only wavelet input is the uniforms of the H1 band. They set one exchangeable θ, which is then applied to 1-day *total* returns.
- **Ablation:** Clayton on raw residuals, with no wavelets, gives a 5.24% 95% hit rate (p=0.655), the same as the published wavelet-copula.
- **Coherence problem:** the H1 band is a 2–8 day filtered series with lag-1 autocorrelation −0.42. Imposing its dependence on 1-day returns that contain all bands has no justification.
- **Benchmark problem:** the comparison changes the copula family and the wavelet step at the same time.
- **Fix:** present wavelets as the horizon *diagnostic*. Do the risk part with h-day aggregated returns and a copula per horizon. Add a 2×2 ablation table: Gaussian/t × raw/H1. **3–4h.**

## HIGH

**H1. The result depends on the refit cadence.**
- With `QUANT_EDGE_FAST=0` (20-day refits):
  - Wavelet-copula 95%: **3.71%** (Kupiec p=0.011, rejected).
  - Gaussian 95%: 6.12% (p=0.040).
  - Wavelet-copula 99%: 1.41% (p=0.108).
- So "WC stays near 5%" is a knife-edge result of the 120-day refits.
- The code defaults to fast mode even when the env var is unset (`run_all.py:50`).
- C1 fixes this.

**H2. The research question is only half answered.**
- Only 1-day VaR/ES is computed. "10d ES" in the recommendation is never computed.
- There are no confidence intervals on λ by band (0.676 / 0.686 / 0.659).
- The H3 band has lag-1 autocorrelation of 0.98, so its effective sample size is tiny.
- My block bootstrap on non-overlapping returns (empirical λ_L at q=0.10):

| Horizon | λ_L | 90% CI |
|---|---|---|
| 1-day | 0.490 | 0.470–0.518 |
| 5-day | 0.482 | 0.436–0.527 |
| 10-day | 0.456 | 0.405–0.502 |
| 20-day | 0.495 | 0.397–0.565 |

  The intervals overlap.
- **Fix:** add bootstrap CIs per band and h-day VaR/ES: √h-scaled Gaussian vs a horizon-specific copula. That comparison is the "what does ignoring it do" answer. **3–4h.**

**H3. The MODWT "bands" are not what the report says.**
- **Where:** `src/modwt.py:48-54`.
- The code sums wavelet coefficients W_j plus the scaling coefficients V_6. The report says "H1=D1+D2", which would be MRA details.
- I checked that ΣW_j+V_J ≠ x (max error 8.4). Energy is preserved, though, so the filters themselves are correct.
- The circular boundary affects L_6 = 190 coefficients in a 750-day window (25%).
- `pywt.mra` fails because 750 is not a multiple of 2^6.
- **Fix:** implement inverse MODWT, or relabel as "MODWT coefficient bands". Trim boundary coefficients. **1–2h.**

**H4. The report builder hard-codes numbers and claims.**
- **Where:** `scripts/build_report_pdf.py:102, 124-130, 141, 213-217`.
- Hard-coded: the date range, "Clayton dominates", λ values, "1,699 OOS days", "~28%", "stays near the 5% target".
- This breaks "one command reproduces every number": if any input changes, the PDF will contradict the outputs.
- **Fix:** template all of these from the JSON outputs. **1h.**

**H5. The recommendation logic is arbitrary and contradicts the data.**
- **Where:** `src/run_all.py:227-231`.
- The rule fires only because ES_WC/ES_G = 1.279, just above an arbitrary 1.25 threshold.
- Its text says "elevated coarse-horizon tail dependence", but λ(H3) < λ(H1).
- "Size hedges to H1 band", "SA high-ρ review" and "10d ES" are not actionable.
- **Fix:** write one numeric, testable action. **0.5–1h.**

**H6. The report overclaims or omits results.**
- **Wavelet-copula 99% VaR is rejected:** 2.35% hits, p=1.8e-6, but it is presented as a success.
- **ES is understated by both models:** realised loss on breach days is 1.52× (WC) and 1.56× (Gaussian) the predicted ES. This is not mentioned.
- **The H1 AIC winner switched to Gaussian** in the last two refits (2026-03-19 and 2026-09-10), and `point_risk.json` has `wc_family=gaussian`. Neither is mentioned.
- **"~28% ES understatement"** is computed against a miscalibrated model.

## MEDIUM

- **M1. PIT mismatch.** The copula is fitted on empirical-rank uniforms (`margins.py:53-57`), but simulation inverts a parametric standardised t (`margins.py:60-66`). Make these consistent. 0.5h.
- **M2. Exchangeable Clayton.** One θ, the mean of 21 τ-based values with τ clipped at 1e-4, gives every pair the same λ_L=0.68, including XLE–XLU. SPY is roughly a combination of the sectors, which inflates average dependence. Use a full-matrix t-copula and justify or drop SPY.
- **M3. No proper ES backtest.** There is only a ratio. Add Acerbi–Szekely Z2 or the McNeil–Frey exceedance test. 1h.
- **M4. Benchmark set is thin.** Add FHS. With daily volatility it gives 4.12% / 0.65% hits.
- **M5. Dependencies and Makefile.**
  - `requirements.txt` is unpinned (`>=`).
  - copulae, statsmodels and pyyaml are unused.
  - Tested working versions: arch 8.0.0, numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, PyWavelets 1.10.0, reportlab 5.0.1, yfinance 1.7.0.
  - `make reproduce` depends on `install`, so it reinstalls over the network every time.
  - 0.25h.
- **M6. Stray and risky files in the ZIP.**
  - `make_submission_zip.sh:29` adds `20261002114514_36365bae65d9.pdf`, which is byte-identical to the organisers' Challenge_Book.pdf. Remove it.
  - The ZIP also includes about 99KB of AI-generated `research_notes/`, plus `reports/Wavelet copula horizon risk.md`. That file claims "Tail dependence climbs with horizon", which contradicts the report.
  - Otherwise the ZIP matches the repo and is 837KB.
  - The script exits 0 even though the file it references is missing (a silent warning).
  - 0.1h.
- **M7. Silent failures.**
  - `margins.py:36-41` swallows exceptions; its `bfill` fallback uses later data from inside the window; `success` is never checked.
  - `run_all.py:111` (`if len(z)<200: continue`) silently drops OOS days.

## LOW

- **L1.** Portfolio return is the mean of log returns (`risk.py:35`, `run_all.py:131`). It is consistent between forecast and realised values, but state it.
- **L2. Dead code and duplication.**
  - `ALPHAS` is imported but unused; 0.05/0.01 are hard-coded (`risk.py:44-45`, `run_all.py:155-160`).
  - `N_SIM_SENS` is unused.
  - Imports inside functions at `run_all.py:118, 287`.
  - `select_copula` refits families that `fit_all_families` already fitted (`run_all.py:60, 72`).
  - `inch`, `ListFlowable` and `ListItem` are unused in the builder.
  - The `i % 5` print.
- **L3. No tests.** Suggested set: λ formulas, Kupiec reference value, MC VaR vs analytic Gaussian VaR, MODWT energy, a no-look-ahead check. 1h.
- **L4. Report presentation.** Appendix B dumps raw CSV with 16-digit floats. Figure 1 has three bars with no CIs. Only 4 of the 10 allowed pages are used.
- **L5. RNG.** One RNG stream is shared sequentially across models, so results depend on call order. Use common random numbers. N=10k gives only 100 tail draws for ES99; state the MC standard error.
- **L6. Makefile.** `make clean` wipes committed outputs. `make` is not available on Windows; the README's manual path covers that.

## Genuinely good

- **No look-ahead.** The window is `returns.iloc[loc-WINDOW:loc]`, so GARCH, MODWT, PIT and copula are all fitted on past data only. The in-sample period (≤2019) and the OOS period (2020+) are separate.
- **Reproducibility.** It works from a clean venv and matches the committed outputs to 1e-6. Data is cached in the repo. The ZIP is 0.8MB.
- **Correct pieces:**
  - Clayton λ_L = 2^(−1/θ).
  - t-copula tail-dependence formula.
  - Marshall–Olkin Clayton sampler and t-copula sampler.
  - VaR/ES sign and quantile direction (`risk.py:37-42`).
  - Kupiec and Christoffersen tests.
  - MODWT filter scaling and upsampling, which is energy-preserving.
- **Design and documentation:** clean modular layout, central config, dataclasses, AI disclosure, an honest limitations section and governance framing. The portfolio choice is sensible.

## Fix order for today, by impact per hour

1. **C1, daily GARCH volatility update.** 2h.
2. **C2, correct t-copula and Clayton likelihoods.** 1.5h.
3. **Re-run, then template every report number from JSON (H4), and rewrite sections 3–5 honestly.** 2h. The honest story: Gaussian/t calibrate at 1-day once volatility updates; tail dependence is about 0.4 empirically and roughly flat across horizons within the CIs; the risk impact shows up in 99% ES and at multi-day horizons.
4. **ZIP and dependencies: remove the stray PDF, pin requirements.** 0.25h.
5. **If time allows:**
   - 2×2 ablation table (1h).
   - Bootstrap CIs on λ by band plus h-day VaR (2–3h).
   - Relabel or fix the MODWT bands (0.5h).