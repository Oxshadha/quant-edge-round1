**Time check, read first.** At the time of this review it was 16:37 UTC, which is 22:07 Sri Lanka time (UTC+5:30). The deadline is 23:59 SLT, so about 1 h 50 min is left, and the Drive upload and link test need around 20 minutes of that. A full 10-page rewrite won't fit. Section 6 is a triage plan for this window. The 10-page outline in section 5 is the target if there's any slack, and the backbone for the Round-2 live defence.

---

## 0. Problems that change the story (I checked these myself, nothing in the repo was modified)

1. **The headline result comes from a bug in model selection.** `src/copulas_fit.py:104-125` scores Clayton by adding up 21 separate pairwise log-likelihoods, using 21 pair-specific θ values, but charges it only k=1 parameter (`:125`). Gaussian is scored on the joint 7-asset likelihood with k=21 (`:61-67`). These AIC numbers can't be compared.
   - I fed the repo's own `fit_all_families` data with no tail dependence at all (a Gaussian copula, ρ=0.6, n=5000). It picked **Clayton with λL=0.595**: AIC −28,621 against −19,269 for Gaussian.
   - On data with real tail dependence (a t(ν=4) copula, true λL=0.314), it again picked Clayton with λL=0.588, and estimated Student-t ν=30.
   - Clayton θ is just 2τ/(1−τ) averaged over pairs (`:110-124`), so its λL is a relabelled Kendall's τ. It can't detect tail dependence.
   - So report §3 ("Clayton dominates AIC on every band", λL 0.66–0.69) and Appendix B don't hold up.
2. **The Student-t fit isn't a t-copula likelihood.** `copulas_fit.py:85-86` scores t-quantiles under a Gaussian density. ν hits the top of the grid (30) in every band, giving λ≈0.0099. The t family can never win.
3. **VaR is frozen for 120 days at a time.**
   - `margins.py:414` uses `last_vol = vol.iloc[-1]`, the in-sample conditional volatility, not a one-step-ahead forecast.
   - `run_all.py:132-153` then holds that VaR for `step` days.
   - `run_all.py:76` defaults to fast mode (`"1"`), so the 20-day refit in `config.py:20` never runs. The design spec in `reports/...md:39` said refit every 20.
   - The March 2020 forecasts therefore use December 2019 volatility. Figure 2 shows step-shaped VaR lines that a crash cuts straight through. That's why Christoffersen rejects both models (p_ind of about 1e-5 or smaller).
4. **The report leaves out results that hurt it.**
   - The wavelet-copula 99% VaR has 2.35% hits, Kupiec p=1.8e-6, so it is rejected (`outputs/backtest_summary.json`). §5 mentions only the 95% result.
   - Realised loss on breach days is **1.52× (95%) and 1.58× (99%) the wavelet-copula's own predicted ES** (`ratio` field, `backtest.py:374-386`). Both models understate ES, so "Gaussian understates ES by 28%" isn't the whole picture.
   - The 28% is simply the mean-ES ratio 0.0340/0.0266. A model that is uniformly higher passes Kupiec at 95% just by being more conservative. There's no loss-function comparison (tick loss, FZ0) and no Diebold–Mariano test.
   - `outputs/point_risk.json` says the AIC winner on the latest window is **gaussian, λL=0**, which contradicts "H1 Clayton". It isn't reported.
5. **The horizon half of the question is never answered.** All risk is 1-day. The OOS test compares Gaussian-on-raw-residuals with Clayton-on-the-H1-band (`run_all.py:140-147`), which changes family and wavelet at the same time. Nothing measures what ignoring horizon does to risk (for example 10-day ES versus √10 × 1-day). §5 quietly swaps "ignoring horizon" for "ignoring tail dependence".
6. **Wavelet band mismatch.** The copula is fitted on summed MODWT *coefficients* (`modwt.py:275-281`, which are not MRA details despite the name) and then used to simulate *1-day return* shocks (`risk.py:191-196`). It's hard to defend live.
7. **Hard-coded numbers.** `build_report_pdf.py:126` (the λL values), `:141` ("1,699", "H1 Clayton") and `:214-217` ("0.66–0.69", "~28%") are typed in, not read from outputs. That weakens the "one command reproduces every number" claim.
8. **What the empirical data actually says** (my scratch check on `data/etf_prices.csv`, non-overlapping returns, block bootstrap):

   | Horizon | χ_L at 5% | 95% CI |
   |---|---|---|
   | 1d | 0.467 | 0.425–0.510 |
   | 5d | 0.479 | 0.402–0.556 |
   | 10d | 0.441 | 0.324–0.593 |
   | 21d | 0.398 | 0.283–0.532 |

   Tail co-movement is substantial and **statistically flat across horizons**. It sits well below the Clayton figure of 0.68. For the portfolio's 97.5% ES, √10 scaling matches overlapping 10-day historical simulation over the full sample (ratio 0.98) but **understates it by 6% over 2020–26** (ratio 1.06). This is an honest, defensible answer.

---

## 1. Compliance matrix

| # | Requirement (brief, CB p1–4) | Where met now | Gap | Fix (priority) |
|---|---|---|---|---|
| R1 | Copula + wavelet framework | MODWT db2 J=6 + 3 copulas (`src/`) | AIC bug; fake t fit; band/return mismatch | P0: stop claiming AIC dominance; P1: compare families on the same composite likelihood, or report nonparametric χ_L by band |
| R2 | Answer part (a): does tail dependence change with horizon? | §3, 3 bars, no uncertainty | Artifact numbers, no CIs, no test | P0: χ_L-by-horizon table with bootstrap CIs (numbers in item 8) and an explicit "no significant change" verdict |
| R3 | Answer part (b): what ignoring horizon does to measured risk | Not answered; swapped for Gaussian vs Clayton | Core question half-missing | P0: 10-day ES: √10 × 1-day versus direct 10-day/H-band; report the % gap, especially in stress |
| R4 | Justify the portfolio | One line, "equal-weight basket" | No reason for these 6 of 9 original SPDR sectors (XLB, XLY, XLP dropped) or for SPY, which overlaps them | P0: two sentences: liquid, inception Dec 1998 gives the longest common history; a sector-rotation proxy; SPY as the market anchor (or drop it) |
| R5 | Justify assets/markets | Implicit | No reason for US over Sri Lanka's market (CSE/ASPI) | P0: one sentence: CSE thin trading and non-synchronous prices bias tail estimates; US ETFs give a clean test; transfer listed as future work |
| R6 | Justify the period | Dates only | No reason 1999 start / 2020 OOS split | P0: ETF inception; in-sample covers dot-com and GFC; OOS covers COVID, 2022 rates, 2025 |
| R7 | Justify the risk measure | None | Cites MAR33 but doesn't use 97.5% ES | P1: add ES 97.5%, the regulatory standard |
| R8 | Justify the models | One line each | No GARCH-t choice, wavelet, J or band rationale | P1: half-page design rationale table |
| R9 | OOS test vs a simple benchmark | Table 1, Kupiec + Christoffersen | Frozen VaR; confounded benchmark; no ES test; no loss comparison; 99% failure buried | P0: report the 99% failure and ES ratio honestly. P1: daily GARCH update, add FHS/HS benchmark, FZ0 loss + DM test, McNeil–Frey ES test (already planned in research notes) |
| R10 | ONE concrete recommendation actionable tomorrow | `manager_recommendation.txt`, canned by `run_all.py:253-267` | Jargon ("SA high-ρ review", "LH anchor"), no number, no trigger, no owner | P0: rewrite (section 5 below) |
| R11 | One command reproduces every number and figure | `make reproduce` | Hard-coded prose numbers; unpinned `>=` deps; needs internet for pip; no Python version; runtime not stated | P0: state runtime and Python version in README; P1: drive prose from JSON, pin versions |
| R12 | README | Present | Results quoted are wrong (item 1); no runtime; no file map to figures | P0: update the key-result lines |
| R13 | Dependency list | `requirements.txt` | Unpinned; `copulae`, `pyyaml`, `statsmodels` unused | P1: `pip freeze`, pinned |
| R14 | Data or a download script | `data/etf_prices.csv` + yfinance | OK. The yfinance fallback (`data.py:16-23`) isn't tested and may drift | Keep the cached CSV; note the snapshot date |
| R15 | AI disclosure (README or appendix) | Appendix A + README | "Follow the team's locked design" is vague; `research_notes/` contains AI working files ("LOCKED DESIGN … do not reopen") | P0: one honest paragraph naming the tools, what they generated, and what the team verified (and the bugs found) |
| R16 | ≤10 pages excl. cover/refs | 4 pages: cover, about 2 pages of content, refs/appendix | Under-uses 8 pages; looks thin | P1: grow to 7–9 pages if time allows |
| R17 | Explain every line/claim (live Round 2) | — | Bugs 1–3 would be exposed in minutes | P0: fix the claims; prepare a "known limitations" sheet |
| R18 | ZIP ≤25 MB with report, code, README, deps, data | 0.84 MB, contents OK | Bundles the brief itself (`20261002114514_36365bae65d9.pdf`, identical to Challenge_Book.pdf; `make_submission_zip.sh:29`, and that file isn't in the repo, so the script fails under `set -e`); ships ~100 KB of AI research notes (`:27-28`) | P0: drop the brief and `research_notes/`; regenerate the ZIP |
| R19 | Team identity, Drive link, tracking code | Author is "SAIFA Quant Edge Team" (`build_report_pdf.py:66`), no team name | Judges can't attribute it | P0: real team name and tracking code on the cover |
| R20 | Drive folder, open access, link tested | Checklist only | Outside the repo | P0: do it by 23:30 SLT |

---

## 2. Who SAIFA and the judges probably are

- **SAIFA is the Students' Association for Industrial & Financial Analysis**, at the University of Colombo Faculty of Science. It was founded in 2013 by undergraduates on the **Industrial Statistics & Mathematical Finance (ISMF)** degree, which is run largely by the Department of Statistics. Its stated aims are industry links and preparing students for the corporate world. Past events include an inter-school quiz, the SAIFA Business Challenge (case study, open to all undergraduates, 2016–17) and SAIFA Day. Sources: [UoC Science, SAIFA page](https://science.cmb.ac.lk/?p=14296) (search snippet; the page itself is blocked by the proxy), [saifa.lk](https://saifa.lk/) (blocked; seen as a search title), [UoC Dept of Statistics](https://science.cmb.ac.lk/?p=2157), [FT.lk on the Statistics Department](https://ft.lk/hr/A-golden-milestone-for-the-UOC-Stat-Department/47-716744).
- **I found no public material on "Quant Edge 1.0"**, which fits its being new in 2026. Judges and sponsors are inferred.
- **Likely judging panel:** University of Colombo Statistics and Mathematics faculty (statisticians, time-series and econometrics people), plus ISMF alumni in industry. That means bank market-risk and ALM teams, actuarial and insurance analytics, CSE brokers and asset managers, and data-science firms in Colombo. Expect CFA/FRM-literate practitioners and PhD statisticians. They are not wavelet specialists, so they will reward clear logic and correct statistics over jargon.
- **What this audience values:**
  - Correct inference: tests, p-values used properly, confidence intervals.
  - Honest reporting of failures.
  - Code they can rerun.
  - A local angle (Sri Lanka or emerging markets) as a creativity bonus.
  - Practitioner realism: Basel/FRTB 97.5% ES and the 10-day horizon. The Central Bank of Sri Lanka follows Basel.
  - FRTB references are fine, but "MAR33", "PLA" and "SA high-ρ" without explanation will read as name-dropping.

---

## 3. What the judges are hoping to see

Formal statistical validation is clearly expected. The brief's wording supports this at every turn:

- **"Tested out-of-sample against a simple benchmark"** is a hypothesis-testing requirement. Coverage tests (Kupiec/Christoffersen) are the minimum. Statisticians will also expect a test that one model beats the other: a Diebold–Mariano test on quantile (tick) loss or FZ0 joint VaR–ES loss.
- **"Justify each one"** means each design choice needs evidence: GARCH-t diagnostics (Ljung–Box on squared residuals), the reasoning behind the wavelet bands, and sensible model selection.
- **"Every claim … may be tested"** means each number needs a confidence interval or a test. "λL changes with horizon" is a claim about a difference, so it needs a bootstrap CI on λL(H3)−λL(H1).
- **"Results that cannot be reproduced … unsupported"** means fixed seeds, pinned dependencies, and prose driven from the output files.
- **"A simple idea explained well beats a complex one explained badly"** means judges will prefer an honest, CI-backed "tail dependence is high but horizon-stable; what matters is volatility scaling" over a confident claim built on an AIC artifact.

**Will 4 pages read as thin?** Yes. There are only about 2 pages of substance in a 10-page budget. There are no diagnostics, no uncertainty, no robustness checks and no pairwise heterogeneity, and Appendix B is a raw CSV dump. It reads as a screening-level effort.

**What would make it stand out:**
- (a) A pair-level tail-dependence heatmap by horizon. Which sectors stop diversifying in a crash: does XLU/XLV hedging fail at 1 month?
- (b) A plain "risk-horizon mismatch" number, such as "√10 scaling understates 10-day ES by X% in stress".
- (c) A lead–lag or regime view: COVID versus the 2022 rate shock.
- (d) One paragraph on transfer to the CSE.
- (e) Openly listing the AIC pitfall the team found. This shows exactly the understanding Round 2 will probe.

---

## 4. Report critique

- **Structure.**
  - There is no executive summary or answer-first opening.
  - Sections 1–2 cram every design choice into two paragraphs with no reasons given.
  - The "Recommendation" is a pasted 6-line text file in 8pt font full of jargon.
  - Appendix B is a raw CSV with 15-digit numbers and a broken line wrap.
- **Figure 1** shows three near-identical bars with no error bars and no comparison to Gaussian/t or to empirical estimates. It conveys nothing and is based on the flawed estimator. Replace it with χ_L by horizon with CIs, plus a pairwise heatmap.
- **Figure 2** has a dense grey line and flat VaR lines; exceptions aren't marked and COVID isn't zoomed. It visually advertises the frozen-VaR bug. Fix: a daily-updating VaR, exception markers, and a 2020 inset.
- **Table 1** has inconsistent p-value formats (0.655 versus 1.80e-06), no Christoffersen/CC columns, no ES test, and no loss column.
- **Narrative honesty.**
  - The §4 title "ignoring tails understates risk" is backed only by higher average ES.
  - §5 omits the 99% rejection.
  - The section titled "Limitations" skips the biggest limitations.
  - There's a "fast reproduction mode" caveat inside the method description.
- **References** lack titles, volumes and DOIs. Reboredo/Ugolini, Patton (2006), Joe (2014), McNeil–Frey (2000), Fissler–Ziegel (2016) and Percival–Walden (2000) are missing.
- **Current recommendation:** "Tighten tactical limits and size hedges to the H1 band; … flags strategic diversification for SA high-ρ review." This is **not concrete**. It has no number, no limit, no trigger, no owner and no deadline. "The H1 band" means nothing to a desk, and it's chosen by an if/else template (`run_all.py:253`). A judge will spot the canned text.

---

## 5. Proposed 10-page outline (cover and references excluded)

| Page | Section | Content |
|---|---|---|
| 0.75 | 1. Executive summary | The question, a two-line answer, three key numbers with CIs, the recommendation in bold |
| 1.25 | 2. Design and justification | Table: choice → alternative considered → reason. Portfolio, assets, period, measure (VaR 99 / ES 97.5), margins, wavelet, bands, copulas, benchmark |
| 1.0 | 3. Data and margins | Summary statistics, GARCH-t fit, residual diagnostics (Ljung–Box, PIT uniformity) |
| 2.0 | 4. Tail dependence across horizons | Empirical χ_L plus a correctly fitted t/Clayton/survival-Clayton by band with block-bootstrap CIs; a test of H3−H1; pairwise heatmap; crisis versus calm |
| 2.0 | 5. Does ignoring horizon change measured risk? | 1d and 10d ES 97.5: √10-scaled versus direct/band-aware versus HS; % gap, full sample and stress |
| 1.5 | 6. Out-of-sample backtest | Daily VaR/ES 2020–26 against the Gaussian and FHS benchmarks; Kupiec/Christoffersen/CC, McNeil–Frey or Acerbi–Székely ES test, FZ0 loss with DM test; exception plot with COVID inset |
| 0.5 | 7. Recommendation | See below |
| 0.5 | 8. Limitations and robustness | Window length, J, wavelet filter, the composite-likelihood caveat, transfer to the CSE |
| 0.5 | 9. Reproducibility and AI use | The command, runtime, seeds, versions, honest AI-use paragraph |

**Rewritten recommendation.** Fill X, Y and Z from the corrected run. The numbers in brackets are my quick estimates.

> "From tomorrow, compute the portfolio's 10-day 97.5% ES directly from overlapping 10-day scenarios instead of scaling 1-day ES by √10. In 2020–26 the √10 rule understated it by **[~6%] (X%)**. As an immediate interim step, apply a ×**(1+X%)** add-on to the 10-day limit utilisation. Separately, because sector crash co-movement stays at **χ_L ≈ 0.40–0.48 at every horizon** (95% CI [Y]), stop counting cross-sector diversification as a hedge in stress. Size the stress loss limit assuming the [XLK–XLF–XLI] pairs fall together. Owner: market-risk manager. Review it monthly as part of the λL/χ_L monitor."

If they want strictly one action, keep only the first sentence and its add-on.

---

## 6. Triage plan for the remaining ~1 h 50 min (impact per hour)

1. **(20 min, highest impact) Make the text honest.**
   - Remove "Clayton dominates AIC". Say instead that family-selection AIC isn't comparable under composite likelihood, so we report nonparametric χ_L with bootstrap CIs (the item 8 numbers, or a 20-line script added to `run_all`).
   - Report the 99% Kupiec rejection and the 1.5× ES breach ratio.
   - Reword §5's answer: tail dependence is high and doesn't change significantly with horizon; what goes wrong is scaling and family (Gaussian has λ=0).
2. **(10 min)** Replace the recommendation (section 5) and add an executive-summary paragraph.
3. **(10 min)** Put the team name and tracking code on the cover. Rewrite the AI disclosure honestly. Add the justification sentences for R4–R7.
4. **(30 min, only if the run is quick)**
   - Daily volatility update: recursive σ²ₜ = ω + αε²ₜ₋₁ + βσ²ₜ₋₁ with frozen parameters, plus a one-step forecast instead of `vol.iloc[-1]`.
   - Re-run and rebuild the PDF.
   - If the run is slow, skip this and list it as a limitation.
5. **(10 min)** Fix `make_submission_zip.sh`: drop line 29 (the brief PDF) and `research_notes`. Regenerate the ZIP and check it with `unzip -l`.
6. **By 23:30 SLT:** upload the ZIP and Drive folder, set "anyone with link", test the link in a private window, submit as "General / Final Submission".

**What the bug triage means for Round 2:** the AIC test (Gaussian data in, Clayton λL=0.595 out) is the first thing a statistician judge would try live. Owning it in the report turns a fatal flaw into a credibility point.

Scratch scripts (repo untouched) are in `/tmp/claude-0/strat/`: `aic_test.py` and `t_test.py` (selection-bias tests), `horizon.py` (χ_L by horizon and the √10 check), `boot.py` (bootstrap CIs). One caveat: running the AIC tests may have created an empty, gitignored `src/__pycache__` directory, because the `-I` flag overrode my no-bytecode setting. `git status` is clean.

Sources:
- [UoC Faculty of Science, SAIFA](https://science.cmb.ac.lk/?p=14296)
- [saifa.lk](https://saifa.lk/)
- [saifa.lk members](https://saifa.lk/members.php)
- [UoC Dept of Statistics](https://science.cmb.ac.lk/?p=2157)
- [FT.lk: UoC Stat Department milestone](https://ft.lk/hr/A-golden-milestone-for-the-UOC-Stat-Department/47-716744)
- [BIS MAR33](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/33/inforce/2023-01-01/published/2020-03-27) (cited in the repo's research notes)