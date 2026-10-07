#!/usr/bin/env bash
# Package the submission: report, runnable code, data, outputs. Run after `make reproduce`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="submission/QuantEdge_Round1_submission.zip"
mkdir -p submission
rm -f "$OUT"

zip -r "$OUT" \
  README.md requirements.txt requirements-lock.txt Makefile pytest.ini reproduce.py QuantEdge_Gmora.ipynb \
  src tests scripts/build_report_pdf.py scripts/make_submission_zip.sh \
  report/QuantEdge_Round1_Report.pdf report/assets/fig*.png \
  outputs/results.json outputs/manager_recommendation.txt outputs/eda_*.csv \
  outputs/insample_copula_fits.csv \
  outputs/tail_comovement_by_band.csv outputs/tail_tests_by_band.csv \
  outputs/tail_comovement_by_horizon.csv outputs/tail_tests_by_horizon.csv \
  outputs/oos_1d_forecasts.csv outputs/oos_10d_forecasts.csv outputs/refit_log.csv \
  outputs/subperiod_hit_rates.csv outputs/seed_robustness.csv outputs/model_implied_tail_by_horizon.csv \
  data/etf_prices.csv data/download_meta.txt \
  -x '*/__pycache__/*' '*.pyc'

ls -lh "$OUT"
unzip -l "$OUT" | tail -1
