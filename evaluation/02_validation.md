# Model validation of the SAIFA Quant Edge Round 1 submission (wavelet–copula VaR/ES)

The current validation does not hold up. Kupiec and Christoffersen are coded correctly, but the headline is an artefact. The "5.24% passes Kupiec" result comes from averaging two failures: the model breaches far too often in the COVID crash and far too rarely afterwards. The model behind it never updates volatility between refits. The "Clayton dominates AIC" and "λ_L ≈ 0.66–0.69" claims come from a likelihood bug. The repo never tests statistically whether tail dependence changes with horizon, which is the research question itself.

None of this is fatal. A fix of about 30 lines, plus 5–6 tests computed from the forecasts the repo already saves, turns it into an honest result that can be defended.

The repo is untouched (`git status` is clean). I replicated the repo's own run within noise: 99% hit rate 2.35% exactly, 95% 5.18% vs 5.24%, so the numbers below are comparable with the report.

## (a) Verdict on current validation

### Validation code that is correct
- **Kupiec test** (`src/backtest.py:25-39`) and **Christoffersen independence and conditional coverage** (`:42-90`): my independent code gives the same LR statistics.

### Validation that is missing or wrong (ordered by severity)

1. **Volatility is frozen for 120 days.**
   - `src/run_all.py:50` defaults `QUANT_EDGE_FAST` to "1", and the `Makefile` reproduce target also sets it to 1, so the model refits every 120 trading days.
   - Between refits, `run_all.py:106-127` caches the whole risk number, and `src/risk.py:139` uses `f.last_vol`. That is σ on the window's last day, not a σ(t+1) forecast.
   - The 1,699 "daily" forecasts have only 15 distinct values. The GARCH model is never actually used as a conditional forecast.
   - This alone explains why the independence test fails everywhere.
2. **The 95% pass hides failure in both directions.** Wavelet–copula 95% hit rates by period:

   | Period | Hit rate (target 5%) |
   |---|---|
   | COVID, Feb–Jun 2020 | 33.7% |
   | 2022 | 8.4% |
   | 2023–26 | 2.0% |

   - 99% hit rates are 24.0% in COVID and 0.64% in 2023–26.
   - The DQ test rejects at p≈0 (DQ=144 at 95%, 565 at 99%).
   - Basel traffic light, full-sample 99%: 40 exceptions in 1,699 days puts **both models in the red zone**. Rolling 250-day windows: wavelet–copula 4.5% red, maximum 27 exceptions; Gaussian 11.4% red, maximum 33.
3. **Expected Shortfall (ES) underestimation is in the outputs but not acknowledged.**
   - `backtest_summary.json` reports realised loss over predicted ES on breach days of 1.52 (WC 95%), 1.58 (WC 99%) and 1.84 (Gaussian 99%). Losses beyond VaR were 52–84% larger than the model's ES.
   - Acerbi–Szekely Z2 for WC 99% is −2.79 against a 5% critical value of about −0.42 (p<0.001). McNeil–Frey has p<0.001.
   - The report never mentions this.
4. **The report misstates its own result.**
   - Section 5 says WC "stays near the 5% target" while WC 99% fails Kupiec (p=1.8e-6), independence and conditional coverage.
   - The "Gaussian understates 99% ES by 28%" claim compares two forecasts that are both rejected.
   - With `QUANT_EDGE_FAST=0`, the denser mode the README advertises, WC 95% hits 3.94% and Kupiec **rejects** (p=0.038). The headline depends on the refit setting.
5. **The AIC model selection is invalid** (`src/copulas_fit.py:170-191`).
   - The Clayton "loglik" sums 21 bivariate pairwise log-likelihoods, each with its own θ, and counts k=1 parameter. It is compared against a full 7-dimensional Gaussian likelihood.
   - On the full 7-dimensional likelihood (≤2019, all bands):
     - Student-t wins everywhere. Raw AIC: t −32,732, Gaussian −30,796, exchangeable Clayton −16,910. H1: −32,350 / −30,551 / −17,013.
     - Clayton is by far the worst.
   - The reported λ_L=0.676 is 2^(−1/mean θ(τ)), i.e. a transform of average Kendall's tau, not a tail measure. The full-MLE Clayton gives λ_L=0.42.
   - The Student-t fit (`:144-152`) evaluates a Gaussian density on t-scores, and `scale` is unused, so ν is pegged at the grid maximum of 30.
6. **The comparison mixes up the wavelet effect and the copula family.**
   - The model is H1-Clayton; the benchmark is raw-Gaussian. My 2×2 ablation (with daily σ) separates the two:
     - Clayton-H1 vs Clayton-raw: DM p=0.80 (99%, FZ0 loss). Gaussian-H1 vs Gaussian-raw: p=0.50.
     - **The wavelet step adds nothing to 1-day risk. Everything comes from the copula family.**
   - The family switches over time: from 2026-03 the H1 AIC selection returns Gaussian (`outputs/point_risk.json`: `wc_family: gaussian`). The recommendation in the report describes a model that is no longer the one running.
7. **The research question is never tested.**
   - No confidence intervals for λ_L per band, no test that λ_L(H1) differs from λ_L(H3), and no h-day backtest.
   - "What does ignoring the horizon do" is answered with Gaussian vs Clayton, which is a question about tail dependence, not about horizon.
   - Band coefficients are strongly autocorrelated (lag-1 AC: H1 −0.43, H2 0.77, H3 0.98). Their iid likelihoods and AICs overstate the information, and H3's effective sample size is tiny.
8. **There is no simple industry benchmark.** No historical simulation and no univariate filtered model.

## (b) Ranked must-add tests, with the numbers I computed

### Results across models

FZ0 is the Fissler–Ziegel joint VaR–ES loss; lower is better.

**99% level, 2020-01-02 to 2026-10-06, n=1,699:**

| Model | Hit % | Kupiec p | Indep. p | DQ p | AS Z2 (p) | McNeil–Frey p | FZ0 | 250-day red % / max |
|---|---|---|---|---|---|---|---|---|
| WC as submitted | 2.35 | 0.000 | 0.000 | 0.000 | −2.79 (0.000) | 0.000 | −1.56 | 4.5 / 27 |
| Gaussian as submitted | 3.18 | 0.000 | 0.000 | 0.000 | −5.30 (0.000) | 0.000 | −0.05 | 11.4 / 33 |
| HS-250 (new simple benchmark) | 1.53 | 0.042 | 0.000 | 0.000 | −0.94 (0.000) | 0.001 | −2.86 | 3.5 / 10 |
| Univariate GARCH-t on portfolio, daily | 1.24 | 0.346 | 0.260 | 0.282 | −0.34 (0.086) | 0.100 | −3.41 | 0 / 7 |
| WC (Clayton-H1), daily σ, refit 20 | 0.71 | 0.199 | 0.071 | 0.011 | +0.25 (0.84) | 0.18 | −3.36 | 0 / 6 |
| Gaussian raw, daily σ, refit 20 | 1.29 | 0.243 | 0.031 | 0.000 | −0.50 (0.018) | 0.002 | −3.32 | 2.5 / 11 |

**95% level:**
- WC with daily σ hits 3.65%: Clayton is too conservative in the body of the distribution (Kupiec p=0.007).
- Gaussian with daily σ hits 5.77% (Kupiec p=0.16), but its ES is too low: Z2 p=0.013, McNeil–Frey p=0.009.

**Diebold–Mariano tests (Newey–West HAC), FZ0 loss:**
- As submitted, WC beats Gaussian: 95% DM=−2.30 (p=0.021); 99% DM=−2.43 (p=0.015).
- As submitted, WC loses to the simple univariate GARCH-t: 99% DM=+2.04 (p=0.041). The as-submitted WC is also worse than HS-250 at 95% (p=0.046).
- Adding daily σ beats the stale version for both models: WC p=0.032, Gaussian p=0.007 (99%, refit 20).
- With daily σ: WC vs Gaussian p=0.58, WC vs univariate GARCH-t p=0.39. They are statistically indistinguishable.

**Model Confidence Set** (Hansen–Lunde–Nason, 90%, FZ0, block size 20): very little power here. COVID dominates the losses, so almost every model, including the as-submitted ones, is retained. Report it as a robustness check, not as evidence.

### Horizon tests (the research question)

In-sample to 2019. Stationary bootstrap of the GARCH residual rows (mean block 20 days), then MODWT re-run on each resample.

| Measure | H1 | H2 | H3 | H3 − H1 |
|---|---|---|---|---|
| Repo λ_L (B=200) | 0.676 [0.653, 0.700] | 0.682 [0.656, 0.708] | 0.656 [0.615, 0.697] | −0.020 [−0.055, 0.009], p≈0.21 |
| Empirical C(q,q)/q at q=0.05 (Schmidt–Stadtmüller), B=200 | 0.387 [0.354, 0.415] | 0.379 | 0.342 [0.280, 0.427] | −0.045 [−0.109, 0.022], p≈0.17 |
| Gaussian-implied C(q,q)/q, same q | 0.337 | 0.338 | 0.317 | – |
| t-copula λ_L, full likelihood (B=100) | 0.172 [0.133, 0.230], ν≈8 | 0.113 [0.074, 0.176], ν≈10 | 0.006 [0.000, 0.031], ν≈40 | −0.166 [−0.225, −0.125], p<0.01 |

- The empirical measure shows excess tail co-movement over Gaussian of about 0.05 at every horizon.
- **Honest answer:** the parametric (t-copula) estimate says tail dependence declines significantly at long horizons. The model-free estimate points the same way but is not significant. The H3 parametric result is weakened by H3's 0.98 autocorrelation.

### Ranked additions

Ranked by value per hour; each has an effort estimate.

1. **Daily σ update in the OOS loop (about 1 hour; most important).** Per refit, store the GARCH parameters and the simulated standardised shocks Z (n_sim×7, built from copula uniforms → t⁻¹/√(ν/(ν−2))). Each day:
   - update `s2 = ω + α·(r−μ)² + β·s2`
   - `rp = (μ + Z*σ_t).mean(1)`, then take the quantile and the tail mean.

   This cuts 99% DQ from 565 to about 8–20 and turns the traffic light green. Make `REFIT_EVERY=20` the reproduce default; it took about 6 minutes per spec on 4 cores.
2. **Basel traffic light (15 minutes).**
   - `pd.Series(hit99).rolling(250).sum()` against green ≤4, yellow 5–9, red ≥10.
   - Report the full-sample binomial CDF (`scipy.stats.binom.cdf(x, n, .01)`) as well.
3. **Engle–Manganelli DQ test (20 minutes).** Regress Hit_t=I_t−α on [1, Hit(t−1…t−4), VaR_t]. DQ = β̂′X′Xβ̂/(α(1−α)), compared with χ²(6).
4. **ES backtests (45 minutes).**
   - Acerbi–Szekely Z2 = Σ r_t·I_t/(T·α·ES_t) + 1. Get critical values by simulating under H0 (location-scale t matched to each VaR_t), or use the fixed approximation of about −0.7 at 97.5%.
   - McNeil–Frey: exceedance residuals (L_t−ES_t)/σ_t, one-sided bootstrap test of zero mean.
   - Du–Escanciano needs the PIT u_t=F̂_t(r_t). It is cheap if the per-day simulated portfolio returns are saved (`np.mean(rp_sims <= r_t)`). Cumulative violation H_t = (α − u_t)·1{u_t≤α}/α. Mean test: √T(H̄−α/2)/√(α(1/3−α/4)). Ljung–Box on H_t for independence.
5. **Comparative backtest with consistent scoring functions (30 minutes).**
   - Quantile (pinball) loss: (α−1{r<v})(r−v), with v = −VaR.
   - FZ0 loss: −1/(αe)·1{r≤v}(v−r) + v/e + log(−e) − 1, with e = −ES (Patton, Ziegel & Chen 2019; Fissler & Ziegel 2016).
   - Diebold–Mariano with Newey–West HAC on the loss differences; MCS from `arch.bootstrap.MCS`.
   - This is the formally correct way to say one model beats another; hit rates alone are not.
6. **Add HS-250 and a univariate GARCH-t on the portfolio as simple benchmarks (30 minutes).** The brief asks for a "simple benchmark", and the univariate GARCH-t is the honest one to beat.
7. **2×2 ablation (Gaussian/Clayton × raw/H1) (30 minutes on top of item 1).** This is the only way to show what the wavelet step contributes. Answer: nothing at the 1-day horizon. The family matters for 99% ES.
8. **Fix the AIC selection and add bootstrap CIs and a horizon test for λ_L (1–1.5 hours).**
   - Use full-dimension likelihoods. Use a proper t-copula: `scipy.stats.multivariate_t(shape=R, df=ν).logpdf(t⁻¹(u)) − Σ t.logpdf`, profiling ν.
   - Report λ_L per band with stationary-bootstrap 95% CIs (resample z rows → MODWT → re-estimate).
   - Test the H3−H1 difference with the bootstrap percentile CI. Add empirical C(q,q)/q at q ∈ {0.10, 0.05, 0.02} against the Gaussian-implied value as a model-free robustness check.
9. **Sub-period table (15 minutes).** COVID, 2020H2–21, 2022, 2023–26, as computed above.
10. **h-day horizon backtest (2–3 hours; this is what actually answers "what does ignoring horizon do").**
    - 10-day VaR/ES from an H2-band copula, compared with the daily model scaled by √10 or simulated as a path.
    - Backtest on non-overlapping 10-day returns (n≈170, so low power; state it) or overlapping returns with a HAC-adjusted Kupiec.
11. **Copula goodness of fit (1–2 hours, optional).** Rosenblatt-transform PIT test, or the Genest–Rémillard–Beaudoin (2009) Cramér–von Mises S_n with parametric bootstrap. Cost is about B×refit, so run it only on pairs or the raw band.
12. **Sensitivity (optional).** Wavelet filter (db2/la8/haar), J=4/6, window 500/750/1000, refit 20/60/120. Report hit rates and FZ0 in one table.

## (c) Wording for presenting validation honestly in the report

**Section 4 lead-in:**
> "We validate in three layers: (i) absolute calibration of VaR (Kupiec unconditional coverage, Christoffersen independence, Engle–Manganelli DQ, Basel traffic light); (ii) absolute calibration of ES (Acerbi–Szekely Z2, McNeil–Frey exceedance residuals); (iii) relative accuracy using strictly consistent scoring functions (quantile loss for VaR, FZ0 loss for joint VaR–ES; Fissler & Ziegel 2016; Patton, Ziegel & Chen 2019) with Diebold–Mariano HAC tests. Tests are run at 95% and 99% on 1,699 out-of-sample days, 2020-01-02 to 2026-10-06."

**Findings paragraph** (rerun after the daily-σ fix and substitute the numbers):
> "With daily-updated GARCH volatilities, the tail-dependent copula passes unconditional coverage and the 99% ES tests (Z2 = 0.25, p = 0.84) and spends no 250-day window in the Basel red zone, whereas the Gaussian copula under-predicts 99% ES (Z2 p = 0.02; McNeil–Frey p = 0.002). The two are statistically indistinguishable on FZ0 loss (DM p = 0.58), and neither beats a univariate GARCH-t on the portfolio (DM p = 0.39). An ablation shows that the wavelet band contributes nothing measurable to 1-day risk (Clayton-H1 vs Clayton-raw, DM p = 0.80); the gain comes from the copula family. All models fail the DQ test at 95%, and all over-breach during February–June 2020; we report this rather than average it away."

**Horizon paragraph:**
> "Parametric (t-copula) lower-tail dependence falls from 0.17 [0.13, 0.23] at 2–8 days to 0.01 [0.00, 0.03] at 32–128 days (bootstrap difference p < 0.01). A model-free estimator shows the same direction but the difference is not significant (−0.045 [−0.109, 0.022]). Coarse-band coefficients are highly autocorrelated (lag-1 ρ ≈ 0.98), so long-horizon estimates rest on few effective observations. We therefore state the horizon effect as suggestive, not established."

**Limitations line:**
> "The AIC comparison in an earlier draft used a pairwise composite likelihood for Clayton against a full likelihood for the Gaussian and t copulas; corrected full-likelihood AIC prefers the t-copula on every band."

Use this only if the bug is fixed. Otherwise remove the "Clayton dominates AIC" and "λ_L ≈ 0.66–0.69" claims entirely.

**Phrases to delete:** "stays near the 5% target", "Gaussian understates crash ES by 28%" (both forecasts it compares are rejected), and "reflecting COVID and 2022 stress" as the excuse for clustering. The clustering comes from the stale volatility in `risk.py:139` / `run_all.py:106-127`.

Everything is in /tmp/claude-0/-home-user-quant-edge-round1/9008468d-2c52-5318-9ae3-7346ebbf91aa/scratchpad/. The test library is `val_lib.py` and the tests can be lifted into `src/backtest.py`. To rerun:
- `exp_a.py`: tests on the existing forecasts.
- `exp_b.py`: the daily-σ rerun and 2×2 ablation (`STEP=20` or `STEP=120`).
- `exp_c.py` and `exp_d.py`: the horizon tests (λ_L CIs, t-copula bootstrap).
- `dm_b.py` and `mcs.py`: DM tests and the Model Confidence Set.

Files in that folder:
- val_lib.py
- exp_a.py
- exp_b.py
- exp_c.py
- exp_d.py
- dm_b.py
- mcs.py
- oos_aug.csv
- oos_b_step20.csv
- oos_b_step120.csv