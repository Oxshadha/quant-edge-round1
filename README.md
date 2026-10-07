# SAIFA Quant Edge 1.0 — Round 1: Risk Across Tails and Timescales

**Question.** Does tail dependence change with the investment horizon, and what does ignoring this do to a
portfolio's measured risk?

**Portfolio.** Equal-weight US sector ETFs XLE, XLF, XLK, XLV, XLI, XLU (daily, 1999 → 2026).
**Framework.** GARCH(1,1)-t margins → MODWT wavelet multiresolution bands (2–8 d, 8–32 d, > 32 d) →
copulas (Gaussian / Student-t / Clayton by full-likelihood AIC, plus empirical) and model-free tail
co-movement with bootstrap inference → 1-day and 10-day VaR / ES, backtested out-of-sample 2020 → 2026
against a Gaussian-copula benchmark and historical simulation.

The report is `report/QuantEdge_Round1_Report.pdf`. The desk recommendation is `outputs/manager_recommendation.txt`.

## Reproduce (one command)

```bash
make reproduce
```

This creates `.venv` with pinned dependencies, runs the unit tests, regenerates every number, table and
figure (`outputs/`, `report/assets/`) and rebuilds the PDF. Runtime is about 10 minutes on a laptop
(most of it is the bootstrap and the Monte Carlo seed-robustness re-runs).

Manual equivalent:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                 # 15 unit tests
.venv/bin/python -m src.run_all               # pipeline -> outputs/
.venv/bin/python scripts/build_report_pdf.py  # report -> report/QuantEdge_Round1_Report.pdf
```

Data: `data/etf_prices.csv` (Yahoo Finance adjusted closes, cached so results are exactly reproducible).
To refresh from Yahoo: `python -c "from src.data import download_prices; download_prices(force=True)"`.

## Code map

| File | What it does |
|------|--------------|
| `src/config.py` | Every design constant (tickers, windows, horizons, bands, seeds) |
| `src/data.py` | Download / load prices, log returns |
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
| `tests/test_core.py` | Unit tests (copula selection, MRA additivity, test statistics, risk maths) |

## Main outputs

| File | Content |
|------|---------|
| `outputs/results.json` | Every headline number used in the report |
| `outputs/insample_copula_fits.csv` | Copula AIC by scale band |
| `outputs/tail_comovement_by_band.csv`, `tail_tests_by_band.csv` | χ_L, χ_U by wavelet band, CIs and tests |
| `outputs/tail_comovement_by_horizon.csv`, `tail_tests_by_horizon.csv` | Same for 1/5/10/21-day returns |
| `outputs/model_implied_tail_by_horizon.csv` | What daily models imply for h-day tail co-movement |
| `outputs/oos_1d_forecasts.csv`, `oos_10d_forecasts.csv` | Every out-of-sample forecast |
| `outputs/subperiod_hit_rates.csv`, `seed_robustness.csv`, `refit_log.csv` | Robustness |

## AI disclosure

AI assistants (Anthropic Claude) were used for literature search, code drafting and debugging, an independent
code review, and drafting text. The team chose the question, data, models and tests, verified every result, and
is responsible for every line of code and every claim. See Appendix A of the report.
