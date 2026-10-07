# SAIFA Quant Edge 1.0, Round 1: Risk Across Tails and Timescales

**Team Gmora** · Tracking code **SAIFA-2026-75323CFE**

**Question.** Does tail dependence change with the investment horizon, and what does ignoring this do to a
portfolio's measured risk?

**Portfolio.** Equal-weight US sector ETFs XLE, XLF, XLK, XLV, XLI, XLU (daily, 1999 → 2026).
**Framework.** GARCH(1,1)-t margins → MODWT wavelet multiresolution bands (2–8 d, 8–32 d, > 32 d) →
copulas (Gaussian / Student-t / Clayton by full-likelihood AIC, plus empirical) and model-free tail
co-movement with bootstrap inference → 1-day and 10-day VaR / ES, backtested out-of-sample 2020 → 2026
against a Gaussian-copula benchmark and historical simulation.

The report is `report/QuantEdge_Round1_Report.pdf`. The desk recommendation is `outputs/manager_recommendation.txt`.

## The analysis notebook

`QuantEdge_Gmora.ipynb` is the complete analysis as one connected chain, with our commentary on what each step shows
and the decision it leads to:

1. Data loading and quality checks
2. Visualisation (performance, volatility, correlation over time)
3. Exploratory data analysis (moments, normality, volatility clustering, joint extremes)
4. Preprocessing (GARCH-t filtering and its diagnostics, MODWT wavelet bands, pseudo-observations)
5. Model building: tail dependence across horizons, with bootstrap tests (part a)
6. Out-of-sample forecasting engine
7. Evaluation: one-day and ten-day backtests, model comparison, seed robustness (part b)
8. Answer, recommendation and the PDF report

The notebook is saved with all outputs, so it can be read without running it.

## Reproduce

Everything runs locally. The price data is bundled (`data/etf_prices.csv`), so nothing is downloaded except the
Python packages. Requires Python 3.10 to 3.13 on Windows, macOS or Linux.

**Option 1: notebook.** Open `QuantEdge_Gmora.ipynb` in Jupyter or VS Code from the unzipped folder and choose
*Run All*. It installs the requirements, runs the unit tests and the full analysis, and rebuilds the PDF
(about 10 minutes).

**Option 2: one command.**

```bash
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` runs the 16 unit tests, regenerates every number, table and figure (`outputs/`,
`report/assets/`) and rebuilds `report/QuantEdge_Round1_Report.pdf`. On macOS/Linux, `make reproduce` does the
same inside a fresh virtual environment.

**Verified.** The full pipeline was run end to end on Python 3.13 and 3.12 (latest packages) and on Python 3.11
with older packages (numpy 1.26, pandas 2.1, scipy 1.11, arch 7.0). `requirements-lock.txt` lists the exact
versions used for the submitted report.

To refresh prices from Yahoo Finance (optional, needs internet):
`python -c "from src.data import download_prices; download_prices(force=True)"`.

## Code map

| File | What it does |
|------|--------------|
| `src/config.py` | Every design constant (tickers, windows, horizons, bands, seeds) |
| `src/data.py` | Load the bundled prices (optional re-download), log returns |
| `src/eda.py` | Data-quality checks, return statistics, normality and volatility-clustering tests, joint extremes, GARCH diagnostics |
| `src/margins.py` | GARCH(1,1)-t fit, daily variance update, pseudo-observations, FHS shocks |
| `src/modwt.py`, `src/wavelets.py` | MODWT multiresolution analysis (additive, LA(8)), horizon bands |
| `src/copulas_fit.py` | Gaussian, Student-t, Clayton (full likelihoods), empirical copula; simulation |
| `src/tail_dependence.py` | χ_L / χ_U tail co-movement, Gaussian-implied values, stationary bootstrap |
| `src/horizon_analysis.py` | Part (a): copula fits and tail co-movement by scale and by horizon, with tests |
| `src/risk.py` | Monte Carlo 1-day and h-day VaR / ES |
| `src/oos.py` | Rolling out-of-sample engine (refit every 20 days, volatility updated daily) |
| `src/backtest.py` | Kupiec, Christoffersen, DQ, traffic light, Acerbi–Szekely Z2, McNeil–Frey, FZ0, Diebold–Mariano |
| `src/evaluate.py` | Backtest tables |
| `src/figures.py` | Figures |
| `src/run_all.py` | Orchestration, `outputs/results.json`, recommendation |
| `scripts/build_report_pdf.py` | Builds the PDF; every number is read from `outputs/` |
| `tests/test_core.py` | 16 unit tests (copula selection, MRA additivity, test statistics, risk maths) |

## Main outputs

| File | Content |
|------|---------|
| `outputs/results.json` | Every headline number used in the report |
| `outputs/eda_*.csv` | Data quality, return statistics, correlations, joint extremes, GARCH diagnostics, wavelet variance shares |
| `outputs/insample_copula_fits.csv` | Copula AIC by scale band |
| `outputs/tail_comovement_by_band.csv`, `tail_tests_by_band.csv` | χ_L, χ_U by wavelet band, CIs and tests |
| `outputs/tail_comovement_by_horizon.csv`, `tail_tests_by_horizon.csv` | Same for 1/5/10/21-day returns |
| `outputs/model_implied_tail_by_horizon.csv` | What daily models imply for h-day tail co-movement |
| `outputs/oos_1d_forecasts.csv`, `oos_10d_forecasts.csv` | Every out-of-sample forecast |
| `outputs/subperiod_hit_rates.csv`, `seed_robustness.csv`, `refit_log.csv` | Robustness |

## AI disclosure

All cognitive and statistical work was done by Team Gmora: the research question, portfolio, data and sample
split, the choice of margin, wavelet and copula models, the design of the out-of-sample experiments and
validation tests, the interpretation of results and the recommendation. AI assistants (Anthropic Claude) were used
as tools under the team's direction for literature search, writing and debugging code, reviewing code for errors,
and editing text. The team checked every result and can explain every line of code and every claim.
See Appendix A of the report.
