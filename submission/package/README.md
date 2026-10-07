# SAIFA Quant Edge 1.0, Round 1: Risk Across Tails and Timescales

**Team Gmora** · Tracking code **SAIFA-2026-75323CFE**

**Report:** `QuantEdge_Round1_Report.pdf`

**Question.** Does tail dependence change with the investment horizon, and what does ignoring this do to a
portfolio's measured risk?

**Data.** Daily adjusted closing prices of six US sector ETFs (XLE, XLF, XLK, XLV, XLI, XLU) from Yahoo Finance,
1999 to 2026, bundled in `data/etf_prices.csv`. Equal-weight portfolio; in-sample to 2019, out-of-sample 2020 to 2026.

## Reproduce (one command)

Python 3.10 to 3.13 on Windows, macOS or Linux.

```bash
pip install -r requirements.txt
python -m src.run_all
```

This regenerates every number, table and figure in the report (about 8 to 10 minutes) into `outputs/` and
`report/assets/`. Every headline number is written to `outputs/results.json`. All randomness is seeded, and the
bundled price file means nothing is downloaded. To refresh prices from Yahoo Finance instead (needs internet):
`python -c "from src.data import download_prices; download_prices(force=True)"`.

## Code

| File | What it does |
|------|--------------|
| `src/config.py` | Every design constant (tickers, windows, horizons, bands, seeds) |
| `src/data.py` | Load the bundled prices (or download them), log returns |
| `src/eda.py` | Data-quality checks, return statistics, normality and volatility-clustering tests, joint extremes |
| `src/margins.py` | GARCH(1,1)-t fit, daily variance update, pseudo-observations, FHS shocks |
| `src/modwt.py`, `src/wavelets.py` | MODWT multiresolution analysis (LA(8)), horizon bands |
| `src/copulas_fit.py` | Gaussian, Student-t, Clayton and empirical copulas; simulation |
| `src/tail_dependence.py` | Tail co-movement, Gaussian-implied values, stationary bootstrap |
| `src/horizon_analysis.py` | Part (a): tail dependence by scale and by horizon, with tests |
| `src/risk.py` | Monte Carlo 1-day and h-day VaR / ES |
| `src/oos.py` | Rolling out-of-sample engine |
| `src/backtest.py`, `src/evaluate.py` | VaR and ES backtests, model comparison tables |
| `src/figures.py` | Figures |
| `src/run_all.py` | Runs everything end to end |

## AI disclosure

All cognitive and statistical work was done by Team Gmora. AI assistants (Anthropic Claude) were used as tools under
the team's direction for writing and debugging code, and reviewing code. See
Appendix A of the report.
