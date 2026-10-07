# SAIFA Quant Edge R1: final evaluation of `quant-edge-round1`

*Head of Quant Research, final judge. Repo checked directly at 16:43 UTC (22:13 SLT, 7 Oct 2026), which leaves **about 1 h 45 min before the 23:59 SLT deadline**. Nothing in the repo was changed.*

---

## 1. Executive verdict

- **What is good.** The submission looks finished. It reproduces with one command (`make reproduce` runs in about 9 s and matches the committed outputs to about 1e-6), it has no look-ahead, and it is laid out cleanly.
- **What is wrong.** All three headline claims are produced by bugs:
  - **"Clayton dominates AIC, λ_L≈0.66–0.69":** the AIC compares a pairwise composite likelihood against a full likelihood. I re-ran the test myself: on data simulated from a Gaussian copula with no tail dependence, the repo's selector picks **Clayton with λ_L=0.595**.
  - **"The wavelet-copula stays near 5%":** VaR is held constant for 120 days. There are only 15 distinct VaR values across 1,699 days.
  - **"Gaussian understates ES by 28%":** this compares two models that both fail the backtest.
- **Brief coverage.** Only half the research question is answered. Nothing shows what ignoring the horizon does to risk, because every risk number is a 1-day number.
- **Competitiveness.** As it stands it might survive a quick paper screening because it looks polished. It will not survive a statistician judge or the live round, where the AIC bug can be shown in five minutes.
- **Biggest risk.** The team submits false claims that a live round will test.
- **The one thing to do today.** Remove or correct every claim built on the bugs, and add a correct likelihood plus a daily volatility update only if they run by 23:05. Then submit an honest, smaller story: crash co-movement is about 0.4–0.47 at every horizon, above what a Gaussian copula implies, and is statistically flat. The backtest failures came from frozen volatility, not from the copula family.

---

## 2. Scorecard (current submission)

| Criterion | Score | Justification |
|---|---|---|
| Creativity | 5/10 | Wavelet plus copula is the framework the brief itself suggests. The portfolio is sensible: US sector ETFs with a long history. There is no original twist: no pairwise or asymmetric view, no regime view, no local (CSE) angle. The good idea in the data (crash co-movement persists across horizons while rally co-movement fades) was never found. |
| Technical rigour | 3/10 | **Correct:** Kupiec and Christoffersen (`src/backtest.py`), no look-ahead (`run_all.py`, window `iloc[loc-WINDOW:loc]`), Clayton and t tail-dependence formulas, the samplers, the MODWT filters (energy is preserved). **Wrong:** invalid AIC (`copulas_fit.py:104-125`); the t-copula likelihood is not a t-copula (`:76-90`, ν pinned at 30); frozen σ_T used as the forecast (`margins.py:34`, `risk.py:34`, `run_all.py:125-126`). No ES test, no loss-based comparison, no confidence intervals, and the 99% rejection is hidden. |
| Clarity | 4/10 | Readable layout, but only 4 of the 10 allowed pages. The report has no answer-first summary, the recommendation is jargon ("SA high-ρ review", "LH anchor"), Appendix B is a raw CSV dump with 16-digit floats, and Figure 1 shows three near-identical bars with no CIs. It overclaims ("stays near the 5% target" while 99% is rejected at p=1.8e-6). |
| Code quality | 6/10 | **Good:** modular `src/`, dataclasses, central config, deterministic, cached data, ZIP is 0.84 MB. **Bad:** silent `except` fallbacks (`margins.py:36-41`); the default `QUANT_EDGE_FAST="1"` (`run_all.py:50`) overrides the configured 20-day refit; the report builder hard-codes results (`build_report_pdf.py:124,141,214-217`); dependencies are unpinned and `copulae`, `statsmodels` and `pyyaml` are unused; no tests. |
| Brief compliance | 5/10 | Deliverables exist and one command reproduces them. Missing or weak: part (b) of the question, justification of portfolio, period, measure and models, a "simple benchmark" that differs in only one respect, an actionable single recommendation, and a team name or tracking code on the cover. The ZIP ships the organisers' own brief PDF. |

**Overall: about 23/50 now.** About 33–36/50 is realistic after today's plan and about 40+ after the Round-2 plan.

---

## 3. Critical issues (verified)

### Resolved disagreements between specialists

- **Line numbers.** The strategist's citations (`margins.py:414`, `modwt.py:275-281`, `backtest.py:374-386`, `run_all.py:253-267`) and the validator's `copulas_fit.py:170-191 / 144-152` are **wrong**: the files are only 66, 54, 105 and 162 lines long. The code critic's line numbers are correct and are used below.
- **Status of `make_submission_zip.sh`.** The script is `scripts/make_submission_zip.sh`, not at the repo root. Line 29 adds `20261002114514_36365bae65d9.pdf`. That file is **no longer in the repo**, but it **is inside the committed ZIP** (73,541 bytes; checked with `unzip -l`). Whatever the script's exit code, the fix is the same: delete line 29 (and line 28) and rebuild the ZIP.
- **Hit rate in `FAST=0` mode** (critic 3.71%, p=0.011; validator 3.94%, p=0.038). Both are Monte Carlo runs with the RNG consumed in a different order (L5). Both reject at 5%, so the conclusion holds either way: the 95% pass only appears at 120-day refits.
- **√h ES scaling** (strategist: understates by 6% over 2020–26, full-sample ratio 0.98; literature agent: overstates, ratio 0.89 at 10 days). The two use different estimators (overlapping vs non-overlapping returns) and different samples. Neither number comes from the repo pipeline. **Ruling:** the unconditional √10 error is small (within about ±10%) and its sign is not stable, so do not headline it.
- **Does tail dependence change with horizon?** The three estimates disagree:
  - The parametric t-copula says yes: λ falls from 0.17 to 0.01, H3−H1 significant.
  - The model-free C(q,q)/q says no: 0.39 to 0.34, difference CI [−0.11, 0.02].
  - The literature agent's non-overlapping returns show lower-tail co-movement roughly flat (0.446 at 1 day, 0.453 at 20 days) and upper-tail co-movement falling (0.41 to 0.32).

  **Ruling:** these are consistent. A *symmetric* copula averages the two tails, so it reads the fading upper tail as fading dependence while crash co-movement persists. The defensible headline: *downside co-movement does not decline significantly with horizon; symmetric models fitted at longer horizons would wrongly conclude it does.* Two caveats must be stated: the upper-tail figures have no CIs yet, and H3 coefficients have lag-1 autocorrelation of 0.98, so the effective sample is tiny.

### Must-fix list

| # | Issue | Evidence (verified) | Fix | Time |
|---|---|---|---|---|
| **C1** | **Invalid AIC selection makes Clayton always win.** | `copulas_fit.py:104-125` sums 21 bivariate log-likelihoods, each with its own θ, uses k=1 and stores mean θ. `band_copula_results.csv`: Clayton "LL" 21,685 vs Gaussian 15,296 on H1. The correct 7-dimensional exchangeable-Clayton LL is **8,508** (worst of the three); the correct t-copula is **16,197–16,238 at ν≈7.4–8** (best). I re-ran the selector on data from a Gaussian copula (ρ=0.6): **Clayton chosen, λ_L=0.595**, AIC −28,621 vs −19,269. | Replace the t log-likelihood with `multivariate_t(shape=R, df=ν).logpdf(t⁻¹(u)) − Σ t.logpdf`, using a grid over ν from 3 to 30. Use the closed-form d-dimensional Clayton density, or drop Clayton from the AIC contest. Report λ from the t-copula and the empirical χ_L. | 30–45 min |
| **C2** | **t-copula likelihood is wrong; ν always lands at the grid maximum of 30.** | `copulas_fit.py:85-86` evaluates a multivariate *normal* logpdf on t-scores; `scale` (`:83`) is unused. The output gives λ≈0.0099 on every band. | Same fix as C1. | (in C1) |
| **C3** | **VaR and ES are frozen for up to 120 days, using σ_T rather than σ_{T+1}.** | `run_all.py:50` defaults to fast mode; `run_all.py:125-126` holds forecasts; `margins.py:34` sets `last_vol = vol.iloc[-1]`; `risk.py:34` uses it. `oos_forecasts.csv` has **15 distinct values per VaR column across 1,699 days**. The COVID forecasts were fitted through 2019-12-31. Consequences: Christoffersen p_ind ≤7.6e-6 everywhere; realised loss over predicted ES is 1.52–1.84; the 99% hit rate is 24% during Feb–Jun 2020. | Store ω, α, β, μ per asset at each refit. Each day run σ²_{t+1}=ω+α(r_t−μ)²+βσ²_t. Keep the refit's simulated standardised shocks Z and rescale them daily. About 30 lines, ported from `scratchpad/exp_b.py`. | 45–60 min (gated) |
| **C4** | **The report states results the outputs contradict.** | `build_report_pdf.py:124` ("Clayton dominates AIC"), `:141`, `:214-217` ("0.66–0.69", "~28%", "stays near the 5% target"). WC 99%: 40 hits vs 17 expected, Kupiec p=1.8e-6, not mentioned. `point_risk.json`: `wc_family: "gaussian", wc_lambda_l: 0.0` on the latest window, which contradicts "H1 Clayton". | Delete these sentences and template every number from the JSON outputs. Report the 99% rejection and the ES ratios. | 30 min |
| **C5** | **Part (b), what ignoring the horizon does to risk, is never answered.** | Only 1-day VaR/ES is computed. The "10d ES" in the recommendation is never computed. §5 swaps "ignoring horizon" for "ignoring tail dependence". | Today: an empirical χ_L-by-horizon table with block-bootstrap CIs plus a Gaussian-implied comparison, and state the finding (see §4). Round 2: 10-day ES backtest. | 30 min today |
| **C6** | **The benchmark comparison changes two things at once.** | WC is Clayton on the H1 band; the benchmark is a Gaussian copula on raw residuals (`run_all.py:113-122`). Ablations:<br>• Clayton on raw residuals, no wavelet: 5.24%, the same as the published result.<br>• Clayton-H1 vs Clayton-raw: DM p=0.80.<br>• Gaussian-H1 vs Gaussian-raw: DM p=0.50.<br>The wavelet step adds nothing to 1-day risk. | Say so explicitly. Position wavelets as the *horizon diagnostic*, not the forecasting engine. | text, 10 min |
| **C7** | **The recommendation is a template, not an action.** | `run_all.py:227-231`: an if/else rule that fires because ES ratio 1.279 > 1.25. The text says "elevated coarse-horizon tail dependence", but λ(H3) < λ(H1). No number, trigger or owner. | Rewrite (see §8). | 10 min |
| **C8** | **ZIP and submission hygiene.** | The ZIP contains the organisers' brief PDF and about 99 KB of AI `research_notes/`. It also contains `reports/Wavelet copula horizon risk.md`, which claims "tail dependence climbs with horizon" (contradicts the report) and a "LOCKED DESIGN" that the code departs from (20-day refits, `copulae`, N=20,000, McNeil–Frey). The cover reads `author="SAIFA Quant Edge Team"` (`build_report_pdf.py:66`), with no team name or tracking code. | Delete lines 27–29 of the ZIP script. Put the real team name on the cover. Pin `requirements.txt` to the tested versions: arch 8.0.0, numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, PyWavelets 1.10.0, reportlab 5.0.1, yfinance 1.7.0, matplotlib (whatever version is installed). Drop the unused packages. | 15 min |

### High (disclose today, fix in Round 2)

- **H1. The bands are not MRA details.** `modwt.py:48-54` sums MODWT *coefficients* W_j plus V_6 and calls them D_j. The bands do not add back to the series; on XLF, corr(x, H1+H2+H3) = −0.305. The circular boundary contaminates about 189 of the 750 window values. Relabel them "MODWT coefficient bands" today.
- **H2. PIT mismatch.** The copula is fitted on rank uniforms (`margins.py:53-57`) but simulated through a parametric t inverse (`:60-66`).
- **H3. Silent fallbacks.** `margins.py:36-41` swallows GARCH failures and its `bfill` uses later data from inside the window. `run_all.py:111` (`len(z)<200: continue`) silently drops OOS days.
- **H4. Portfolio double-counts market beta.** SPY sits alongside its own sectors, which inflates average correlation and makes an exchangeable copula even less suitable. Justify it as the market anchor or drop it.

---

## 4. Methodology vs literature and industry practice

**What is sound**
- **Pipeline order.** GARCH-t margins, then a wavelet decomposition of standardised residuals, then a copula per band, then a VaR/ES backtest. This matches the literature: Cai et al. 2020, Jammazi & Reboredo 2016, Shahzad et al. 2016. Filtering first follows McNeil & Frey 2000.
- **Filtering matters.** Raw returns imply ν≈4, while filtered residuals give ν≈7.6. A large part of naive "tail dependence" is shared volatility clustering. This is a good teaching point to include.
- **Rank pseudo-observations** are canonical maximum likelihood (Genest, Ghoudi & Rivest 1995).
- **MODWT over DWT** is the right choice for arbitrary sample lengths and shift invariance (Percival & Walden 2000).
- **The out-of-sample design** is right: a rolling window, a ≤2019 / 2020+ split, and OOS that covers COVID, the 2022 rate shock and 2025.

**What is naive (expect a challenge)**
1. **Model selection across non-comparable likelihoods.** Composite likelihoods need a composite-likelihood AIC (Varin & Vidoni 2005). Parametric λ is also highly family-dependent; Frahm, Junker & Schmidt 2005 recommend the nonparametric estimator (Schmidt & Stadtmüller 2006) with bootstrap CIs.
2. **Exchangeable Clayton.** One θ forces the same λ_L on XLE–XLU and XLK–XLF. Practice is a full-matrix t-copula, pair-copulas or vines (Aas et al. 2009), or a factor copula with SPY as the factor (Oh & Patton 2017).
3. **Static parameters without daily σ updating.** Both industry FHS and academic practice update conditional volatility daily (Barone-Adesi et al. 1999; Kuester, Mittnik & Paolella 2006). A univariate GARCH-t or FHS on the portfolio with daily σ already passes:
   - GARCH-t at 99%: 1.24% hits, Kupiec p=0.35, independence p=0.26.
   - FHS: 4.83% / 0.82% hits.

   Berger & Missong 2014 find that margins matter more than the copula for VaR. The data here agree.
4. **Using band dependence for 1-day returns.** The H1 band (2–8 day scale, lag-1 autocorrelation −0.42) supplies the dependence for 1-day *total* returns. No paper gives a recipe for recombining scales into a single VaR. The literature reports scale-by-scale dependence and hedging, e.g. Berger 2015/2016, Conlon, Cotter & Gençay 2016, Rua & Nunes 2009. The defensible design is: **diagnose per scale, then measure risk at matched horizons.** Berger & Gençay 2018 give the honest rationale for using H1 for daily VaR: short scales drive daily risk.
5. **The horizon question needs horizon risk.** FRTB (BCBS MAR33) uses 97.5% ES at a 10-day base with liquidity-horizon cascading. Diebold et al. 1998 and Danielsson & Zigrand 2006 show √h scaling fails under fat tails and jumps. A risk-literate judge will ask "where is the 10-day ES?"
6. **Boundary handling.** Percival & Walden and Cornish, Bretherton & Percival 2006 drop the L_j−1 boundary coefficients; the repo does not. Wavelets computed at the series edge also leak future data unless done causally (Zhang, Gençay & Yazgan 2017). The repo avoids this only because the copula is refit on past windows.

**The honest scientific answer the data support**
- Crash co-movement measured by empirical χ_L at the 5% level:

  | Horizon | χ_L | 95% CI |
  |---|---|---|
  | 1 day | 0.47 | [0.43, 0.51] |
  | 5 days | 0.48 | [0.40, 0.56] |
  | 10 days | 0.44 | [0.32, 0.59] |
  | 21 days | 0.40 | [0.28, 0.53] |

- It is high, roughly 0.05 above the Gaussian-implied 0.34 at the same quantile, and statistically flat across horizons.
- Upside co-movement fades with horizon (0.41 to 0.29–0.32), the Longin–Solnik / Ang–Chen asymmetry stretched over time.
- So symmetric models (Gaussian, and t fitted per band) understate long-horizon *downside* co-movement. A Gaussian copula ignores it at every horizon.

That is the "simple idea explained well" the judges asked for.

---

## 5. Statistical validation

**Current state.**
- The repo has Kupiec and Christoffersen (independence and conditional coverage), both correctly coded and independently replicated.
- The ES "test" is a ratio of means (`backtest.py:93-105`), not a test.
- There are no confidence intervals, no DQ test, no traffic light, no scoring-function comparison, and no horizon test.

**What the judges (UoC Statistics / ISMF faculty and industry alumni) will expect.** Formal hypothesis tests, proper p-value use, CIs on every estimated quantity, honest reporting of failures, and a formal "beats the benchmark" test, not just hit rates.

**Numbers already computed by the specialists** (scratch only, not yet reproducible from the repo)

*99% level, 1,699 OOS days:*

| Model | Hit % | Kupiec p | Indep. p | DQ p | AS Z2 (p) | McNeil–Frey p | FZ0 | 250-day red % / max |
|---|---|---|---|---|---|---|---|---|
| WC as submitted | 2.35 | <0.001 | <0.001 | <0.001 | −2.79 (<0.001) | <0.001 | −1.56 | 4.5 / 27 |
| Gaussian as submitted | 3.18 | <0.001 | <0.001 | <0.001 | −5.30 (<0.001) | <0.001 | −0.05 | 11.4 / 33 |
| HS-250 | 1.53 | 0.042 | <0.001 | <0.001 | −0.94 | 0.001 | −2.86 | 3.5 / 10 |
| Univariate GARCH-t, daily σ | 1.24 | 0.35 | 0.26 | 0.28 | −0.34 (0.086) | 0.10 | **−3.41** | 0 / 7 |
| Clayton-H1, daily σ, refit 20 | 0.71 | 0.20 | 0.07 | 0.011 | +0.25 (0.84) | 0.18 | −3.36 | 0 / 6 |
| Gaussian, daily σ, refit 20 | 1.29 | 0.24 | 0.031 | <0.001 | −0.50 (0.018) | 0.002 | −3.32 | 2.5 / 11 |

*95% level, daily σ:* Clayton hits 3.47–3.65% (Kupiec p=0.002–0.007, **too conservative**). Gaussian hits 5.53–5.77% (passes), but its ES is too low (Z2 p=0.013).

*Diebold–Mariano tests (FZ0 loss, HAC):*
- As submitted, WC beats Gaussian (p≈0.02) but **loses to univariate GARCH-t** (p=0.041).
- With daily σ, WC vs Gaussian p=0.58 and WC vs GARCH-t p=0.39: no significant difference.
- Daily σ beats stale σ for both models (p=0.032 and p=0.007).

*Sub-periods (WC as submitted, 95%):* COVID 33.7%, 2022 8.4%, 2023–26 2.0%. The full-sample "pass" is an average of two failures.

*Horizon tests, ≤2019, stationary bootstrap with MODWT re-run on each resample:*
- Model-free C(q,q)/q at q=0.05: H1 0.387 [0.354, 0.415], H3 0.342 [0.280, 0.427]. H3−H1 = −0.045 [−0.109, 0.022], p≈0.17.
- t-copula λ: H1 0.17 [0.13, 0.23], H3 0.006 [0, 0.03], difference significant.
- The repo's own λ (H3−H1 = −0.020, p≈0.21) is meaningless; it is Kendall's τ relabelled.

**Recommended validation suite, ranked by value per hour**

1. **Daily σ update.** Without it, every other test is testing the bug.
2. **Basel traffic light.** Rolling 250-day 99% exceptions with green ≤4, yellow 5–9, red ≥10, plus the binomial CDF. 15 min.
3. **Kupiec, Christoffersen and Engle–Manganelli DQ** at 95% and 99%. 20 min.
4. **ES backtests.** Acerbi–Szekely Z2 with simulated critical values, and McNeil–Frey exceedance residuals. 45 min.
5. **Comparative backtest.** Quantile loss and FZ0 (Fissler & Ziegel 2016; Patton, Ziegel & Chen 2019), then Diebold–Mariano with HAC. The Model Confidence Set has little power here because COVID dominates; report it only as a robustness check. 30 min.
6. **Simple benchmarks.** HS-250 and a univariate GARCH-t or FHS on the portfolio. This is the honest "simple benchmark". 30 min.
7. **2×2 ablation**, Gaussian/t × raw/H1, to isolate what the wavelet step contributes. 30 min.
8. **Horizon inference.** Empirical λ_L and λ_U at q ∈ {0.10, 0.05, 0.02} by band and on non-overlapping 1/5/10/21-day returns, with stationary-bootstrap CIs and a test of the H3−H1 difference. 1–1.5 h.
9. **Sub-period table.** 15 min.
10. **10-day ES backtest.** √10 vs simulated paths vs horizon-matched copula, on about 170 non-overlapping windows; state the low power. 2–3 h.
11. **Copula goodness of fit and sensitivity** (Genest–Rémillard–Beaudoin 2009; filter, J, window, refit cadence). Optional.

**Today:** do items 2, 3 and 8 (the χ_L part only), plus item 1 if its gate passes. Everything else goes to Round 2.

---

## 6. Brief compliance matrix

| Requirement | Status | Action |
|---|---|---|
| Copula + wavelet framework | ⚠ implemented, but selection invalid and bands mislabelled | C1, H1 |
| (a) Does tail dependence change with horizon? | ✗ artefact numbers, no CIs | Empirical χ_L table with CIs plus a verdict |
| (b) What ignoring it does to measured risk | ✗ missing | Today: state it qualitatively with the Gaussian-implied gap. Round 2: 10-day ES |
| Justify portfolio, assets, market, period, measure, models | ✗ one-liners | One design-rationale table (½ page) |
| OOS test vs a simple benchmark | ⚠ confounded, frozen σ, 99% failure hidden | Report honestly, add GARCH-t/HS benchmark, ES test |
| One concrete recommendation for tomorrow | ✗ canned jargon | §8 |
| One command reproduces every number | ⚠ prose hard-coded | Template from JSON |
| README, dependencies, data | ⚠ unpinned, unused packages | Pin versions, state runtime and Python version |
| AI disclosure | ⚠ vague | Honest paragraph (tools, scope, what was verified, bugs found) |
| ≤10 pages | ✓ (4 pages, which is thin) | Grow to 6–7 today |
| ZIP ≤25 MB | ✓ 0.84 MB, but includes the brief and AI notes | Delete script lines 27–29 |
| Team ID, tracking code, Drive link | ✗ / pending | Cover page, upload by 23:30 |

---

## 7. Recommended approach

### Track A: tonight (22:15 to 23:59 SLT, hard stop)

Assumes 3 people. If solo, do A-W, A-C2 and the ZIP fix only, and list C3 as a disclosed limitation.

| Time (SLT) | Who | Task | Gate |
|---|---|---|---|
| 22:15–22:20 | All | Create a branch. Create the Drive folder now and test "anyone with link". Confirm the tracking code and leader email. | — |
| 22:20–22:55 | **A-C2** (coder 2) | `copulas_fit.py`: correct t-copula log-likelihood (`scipy.stats.multivariate_t`, ν grid 3–30) and either the closed-form d-dimensional Clayton or Clayton removed from the AIC. Then add `src/horizon_tail.py`, about 40 lines lifted from `/tmp/claude-0/strat/horizon.py` + `boot.py`: empirical χ_L at q=0.05 on non-overlapping 1/5/10/21-day portfolio-constituent returns, a 500-draw block bootstrap with a fixed seed, and the Gaussian-implied comparison. It writes `outputs/horizon_tail.csv`. Call it from `run_all`. | Runs in under 2 min |
| 22:20–23:05 | **A-C1** (coder 1) | Daily GARCH σ recursion in the OOS loop (port the logic from `scratchpad/exp_b.py`). Keep step=120 so the run stays under a minute. Add a 250-day traffic light and DQ to `backtest.py` (from `val_lib.py`). | **If it is not running cleanly by 23:05, revert it and disclose stale σ as the top limitation.** |
| 22:20–23:10 | **A-W** (writer) | Rewrite `build_report_pdf.py` text. Delete the claims at `:124`, `:141` and `:214-217`. Pull every number from JSON/CSV. Add: an exec summary; a design-rationale table (portfolio = liquid ETFs with a common history from Dec 1998; US rather than CSE because thin trading biases tail estimates; period covers dot-com, GFC, COVID and 2022; 99% VaR with 97.5%/99% ES); the χ_L table; honest backtest text (the 99% result, the ES ratios, sub-periods); the new recommendation; limitations (H1 relabel, exchangeable copula, stale σ if not fixed, the composite-likelihood bug found and fixed); and the AI disclosure. Put the team name on the cover. Relabel "MRA" as "MODWT coefficient bands". | — |
| 23:05–23:20 | All | Merge, `make reproduce`, read every number in the PDF against the JSON. | PDF builds |
| 23:20–23:30 | A-C1 | Pin `requirements.txt`, remove unused packages, delete ZIP-script lines 27–29, rebuild, `unzip -l`. Make sure `outputs/` is regenerated by the final run. | ZIP <25 MB, no brief PDF inside |
| 23:30–23:45 | Leader | Upload ZIP and folder, test the link in a private window, submit as "General / Final Submission". | Confirmation screen |
| 23:45–23:59 | — | Buffer. **Do not edit after submitting.** | — |

**Floor version if everything slips:** delete the false claims, add the honest limitations paragraph, the new recommendation and the AI paragraph, fix the ZIP. Text-only edits take 30–40 min.

### Track B: next round and the ideal approach

1. **Margins.** Refit GARCH-t (or GJR-GARCH-t for leverage) every 20 days, update daily, with Ljung–Box on z and z² and PIT uniformity diagnostics. EVT/POT tails on the residuals (McNeil & Frey 2000).
2. **Dependence diagnostic by scale.** Proper MODWT MRA via inverse MODWT, LA(8) filter, boundary coefficients dropped. Wavelet *multiple* correlation by scale (Fernández-Macho 2012). Empirical λ_L and λ_U by scale and horizon with bootstrap CIs. A pairwise heatmap per horizon showing which sector hedges fail in a crash.
3. **Copulas.** Full-matrix t and skew-t; survival Clayton/Gumbel, BB1 or BB7 per pair, or an R-vine (pyvinecopulib, Dißmann et al. 2013); a factor copula with SPY as the factor (Oh & Patton 2017). Selection with correct likelihoods plus a Genest–Rémillard–Beaudoin goodness-of-fit test.
4. **Horizon risk, the core of part (b).** 10-day 97.5% ES computed three ways:
   - √10 × 1-day;
   - simulated GARCH paths with the daily copula (FHS-style);
   - a horizon-matched copula on 10-day blocks or the H2 band.

   Backtest on non-overlapping 10-day OOS windows and report the % gap in stress vs calm. Add a FRTB liquidity-horizon cascade as the industry tie-in.
5. **Dynamics.** DCC or asymmetric DCC (Cappiello, Engle & Sheppard 2006), a GAS t-copula (Creal, Koopman & Lucas 2013), or a two-regime copula (Okimoto 2008) to separate COVID-type regimes.
6. **Validation.** The full §5 suite, plus a sensitivity table (filter, J, window, refit cadence) and unit tests: λ formulas, Kupiec reference value, Monte Carlo vs analytic Gaussian VaR, MODWT energy, a no-look-ahead assertion.
7. **Creativity angles.** Transfer to the CSE/ASPI with non-synchronous-trading corrections. An optional conformal calibration layer on VaR (arXiv 2507.05470). Skip normalising-flow copulas: hard to defend live, no gain at d=6–7.

---

## 8. Report rewrite plan (10-page target; tonight aim for 6–7)

| Pages | Section | Content and figures |
|---|---|---|
| 0.75 | 1. Executive summary | The question; a two-line answer; three numbers with CIs; the recommendation in bold |
| 1.0 | 2. Design and justification | Table with columns choice / alternative / reason: portfolio, assets, market (US vs CSE), period, measure, margins, wavelet, bands, copulas, benchmark |
| 1.0 | 3. Data and margins | Summary statistics; GARCH-t estimates; Ljung–Box and PIT diagnostics; the point that filtering removes about half the apparent tail dependence (ν 4 → 7.6) |
| 2.0 | 4. Tail dependence across horizons | **Fig. 1:** χ_L and χ_U by horizon with 95% CIs, Gaussian-implied line overlaid (replaces the three-bar chart). **Table:** corrected AIC by band (t wins), t-λ with CIs, test of H3−H1. **Fig. 2:** pairwise lower-tail heatmap at 1 day vs 21 days. |
| 1.5 | 5. What ignoring horizon or tails does to risk | Gaussian-implied vs empirical co-crash probability; 99% ES under Gaussian vs t-copula; 10-day ES √10 vs simulated, flagged preliminary if only partly done |
| 2.0 | 6. Out-of-sample backtest | **Table:** hit rate, Kupiec, Christoffersen independence, DQ, Z2, McNeil–Frey and FZ0 for WC, Gaussian and GARCH-t/HS. **Fig. 3:** daily VaR path with exception markers and a COVID inset. Sub-period table. DM tests. 2×2 ablation (wavelet contribution = none at 1 day). |
| 0.5 | 7. Recommendation | One action, one number, one owner, one review trigger |
| 0.75 | 8. Limitations and robustness | Bands are coefficient bands; H3 effective sample size; exchangeable vs full-matrix copula; the AIC bug found and fixed; transfer to the CSE |
| 0.5 | 9. Reproducibility and AI use | Command, runtime, seeds, pinned versions; the AI paragraph |

**Rewritten recommendation.** Fill the numbers from outputs. Variable names refer to JSON fields.

*Version 1, if the daily-σ and corrected-copula rerun succeeds:*
> **"From tomorrow, set the portfolio's daily 99% Expected Shortfall limit from the t-copula model with daily-updated GARCH volatility, not the Gaussian copula. Over 2020–26 the Gaussian's 99% ES was exceeded on breach days by an average of {g99.ratio−1:.0%} (Acerbi–Szekely p={p}), versus {t99.ratio−1:.0%} for the t-copula. Until the switch is approved, multiply the Gaussian 99% ES by {g99.ratio:.2f}. Owner: market-risk manager. Trigger a review if any 250-day window reaches 5 or more 99% exceptions."**

*Version 2, the text-only fallback, true with today's outputs:*
> **"From tomorrow, recompute GARCH volatility in the VaR engine every day instead of only at each 120-day model refit. Holding it fixed produced 40 breaches of the 99% VaR where 17 were expected over 2020–26 (Kupiec p=1.8×10⁻⁶), with 24% of days breaching in Feb–Jun 2020. Interim: apply a ×1.58 add-on to 99% ES, the observed realised-to-predicted ratio on breach days."**

---

## 9. Likely judge and live-round questions, with model answers

1. **"Your earlier AIC picked Clayton everywhere. Why should we believe any copula selection?"**
   It summed 21 pairwise likelihoods against one full 7-dimensional likelihood, so the comparison was invalid. We showed it by simulation: data from a Gaussian copula with no tail dependence was classified as Clayton with λ=0.6. With correct full likelihoods, the t-copula wins every band (H1 LL 16,197 vs Gaussian 15,296 vs Clayton 8,508). Because parametric λ depends on the family, we headline the nonparametric χ_L with bootstrap CIs (Frahm, Junker & Schmidt 2005).

2. **"Does tail dependence change with horizon? Yes or no."**
   Downside: not significantly. χ_L ≈ 0.47 [0.43, 0.51] at 1 day and 0.40 [0.28, 0.53] at 21 days, both above the Gaussian-implied 0.34. Upside co-movement fades. The risk is asymmetric: diversification that shows up in rallies does not show up in crashes at any horizon.

3. **"Why did Christoffersen reject?"**
   In the submitted version, volatility was held fixed between 120-day refits, so March 2020 was forecast with December 2019 volatility. Breaches cluster when the forecast cannot react. With a daily GARCH recursion the 99% traffic light is green and independence is not rejected. *(Only if fixed; otherwise: "it is a known limitation and here is the fix.")*

4. **"What does the wavelet step add?"**
   Nothing measurable to 1-day VaR: Clayton-H1 vs Clayton-raw, DM p=0.80. Its value is diagnostic, showing dependence scale by scale. Forecasting at a horizon should use dependence matched to that horizon (Berger & Gençay 2018; Rua & Nunes 2009).

5. **"Your 'H1 = D1+D2' bands don't sum back to returns. Why?"**
   They are sums of MODWT wavelet coefficients, not multiresolution-analysis details. We relabelled them. The coefficients are what wavelet-correlation estimators use (Whitcher et al. 2000). Boundary-affected coefficients (L_j−1) should be dropped; that is planned.

6. **"Why is the Gaussian copula the 'simple benchmark'? Why not historical simulation?"**
   Fair challenge. We added HS-250 and a univariate GARCH-t on the portfolio. The univariate GARCH-t is very competitive (99%: 1.24% hits, Kupiec p=0.35). A multivariate copula earns its place in ES, in stress attribution and in the horizon question, not in beating FHS on 1-day 95% VaR.

7. **"How do you test ES, given ES isn't elicitable?"**
   ES is jointly elicitable with VaR (Fissler & Ziegel 2016). We use FZ0 loss with Diebold–Mariano tests for comparison, and Acerbi–Szekely Z2 and McNeil–Frey for absolute calibration.

8. **"Why US sector ETFs and not the CSE?"**
   CSE stocks trade thinly and prices are non-synchronous, which biases dependence estimates towards zero. US ETFs give a clean test with a common history from Dec 1998 that covers the dot-com bust, the GFC, COVID and 2022. Transfer to the CSE with Dimson/Scholes–Williams-style corrections is future work.

9. **"Why include SPY with its own sectors?"**
   It is the market anchor and the natural hedge instrument. It does double-count beta and inflate average correlation, so in Round 2 we treat it as a factor (Oh & Patton 2017) rather than a holding.

10. **"Live task: change the refit window to 500 and the wavelet filter to la8, then rerun."**
    Answer by pointing at `src/config.py` (WINDOW, filter, J). One command, `make reproduce`, about 1 min. Know which outputs change. Rehearse this before Round 2.

11. **AI use: "Which parts did AI write, and how do you know they are right?"**
    AI assisted with code drafting and the literature notes. We independently re-derived the copula likelihoods, which is how we found the composite-likelihood and t-density errors. We replicated Kupiec and Christoffersen by hand, and checked the absence of look-ahead and the MODWT energy preservation. Every number in the PDF is generated from code we can explain line by line.

12. **AI use: "Your notes say 'LOCKED DESIGN, refit every 20 days, copulae MLE'. Why does the code differ?"**
    Be candid. The implementation drifted for speed (FAST mode). We found the drift during review, made the daily update the default (if done), and listed the remaining gaps in Limitations. *The research notes are no longer in the ZIP, but judges may still ask how design decisions were made.*

---

## 10. Appendix: key references

- **Copulas:** Sklar (1959); Nelsen (2006) *An Introduction to Copulas*; Joe (2014) *Dependence Modeling with Copulas*; Genest, Ghoudi & Rivest (1995) *Biometrika*; Demarta & McNeil (2005) *Int. Stat. Rev.*; Embrechts, McNeil & Straumann (2002); Frahm, Junker & Schmidt (2005) *IME* 37; Schmidt & Stadtmüller (2006) *Scand. J. Stat.* 33; Varin & Vidoni (2005) *Biometrika*; Genest, Rémillard & Beaudoin (2009) *IME*.
- **Dependence asymmetry:** Longin & Solnik (2001) *JF* 56; Ang & Chen (2002) *JFE* 63; Hong, Tu & Zhou (2007) *RFS* 20; Patton (2006) *IER* 47; Christoffersen, Errunza, Jacobs & Langlois (2012) *RFS* 25; Oh & Patton (2017) *JBES* 35; Aas, Czado, Frigessi & Bakken (2009) *IME* 44; Creal, Koopman & Lucas (2013) *JAE*.
- **Wavelets:** Percival & Walden (2000); Gençay, Selçuk & Whitcher (2002); Whitcher, Guttorp & Percival (2000) *JGR*; Cornish, Bretherton & Percival (2006); Rua & Nunes (2009) *JEF* 16; Fernández-Macho (2012) *Physica A* 391; Berger & Gençay (2018) *JEDC* 92; Berger (2016) *J. Risk* 18; Conlon, Cotter & Gençay (2016) *EJF* 22; Zhang, Gençay & Yazgan (2017) *Econ. Letters* 158.
- **Applied wavelet-copula studies:** Cai et al. (2020); Jammazi & Reboredo (2016) *Energy*; Shahzad et al. (2016) *Physica A*; Berger & Missong (2014) *IRFA* 33.
- **Risk and backtesting:** McNeil & Frey (2000) *JEF* 7; Barone-Adesi, Giannopoulos & Vosper (1999) *JFM*; Kuester, Mittnik & Paolella (2006) *JFEc*; Kupiec (1995); Christoffersen (1998) *IER*; Engle & Manganelli (2004) *JBES*; Acerbi & Szekely (2014) *Risk*/MSCI; Fissler & Ziegel (2016) *Ann. Stat.*; Patton, Ziegel & Chen (2019) *J. Econometrics* 211; Nolde & Ziegel (2017) *AoAS* 11; Hansen, Lunde & Nason (2011) *Econometrica* (MCS).
- **Horizon and regulation:** Drost & Nijman (1993) *Econometrica* 61; Diebold, Hickman, Inoue & Schuermann (1998) *Risk*; Danielsson & Zigrand (2006) *JBF* 30; BCBS MAR32/MAR33 (FRTB).

**Specialist scratch work** (repo untouched; lift the code from here):
- `/tmp/claude-0/-home-user-quant-edge-round1/9008468d-2c52-5318-9ae3-7346ebbf91aa/scratchpad/`:
  - `val_lib.py`: tests (DQ, Z2, McNeil–Frey, FZ0, DM)
  - `exp_b.py`: daily-σ rerun and 2×2 ablation
  - `exp_c.py`, `exp_d.py`: horizon CIs
  - `exp.py`, `exp2.py`: correct 7-dimensional likelihoods, FHS, MRA check
  - `ana/`: the code critic's ablations and empirical λ
- `/tmp/claude-0/strat/`:
  - `aic_test.py`: the selection-bias demonstration; I re-ran it and confirmed Clayton chosen on Gaussian data
  - `horizon.py`, `boot.py`: χ_L by horizon with CIs

**Repo files that must change:**
- `/home/user/quant-edge-round1/src/copulas_fit.py`
- `/home/user/quant-edge-round1/src/margins.py`
- `/home/user/quant-edge-round1/src/risk.py`
- `/home/user/quant-edge-round1/src/run_all.py`
- `/home/user/quant-edge-round1/src/modwt.py`
- `/home/user/quant-edge-round1/scripts/build_report_pdf.py`
- `/home/user/quant-edge-round1/scripts/make_submission_zip.sh`
- `/home/user/quant-edge-round1/requirements.txt`
- `/home/user/quant-edge-round1/README.md`