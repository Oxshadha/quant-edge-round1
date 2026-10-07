#!/usr/bin/env bash
# Package the submission into one ZIP that unzips to a single folder. Run after the pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NAME="QuantEdge_Gmora_SAIFA-2026-75323CFE"
OUT="submission/QuantEdge_Round1_submission.zip"
STAGE="submission/.stage"
mkdir -p submission
rm -rf "${STAGE:?}" "$OUT"
mkdir -p "$STAGE/$NAME"

FILES=(
  README.md requirements.txt requirements-lock.txt Makefile pytest.ini reproduce.py QuantEdge_Gmora.ipynb
  src/*.py tests/*.py scripts/build_report_pdf.py scripts/make_submission_zip.sh
  report/QuantEdge_Round1_Report.pdf report/assets/fig*.png
  outputs/results.json outputs/manager_recommendation.txt outputs/eda_*.csv
  outputs/insample_copula_fits.csv
  outputs/tail_comovement_by_band.csv outputs/tail_tests_by_band.csv
  outputs/tail_comovement_by_horizon.csv outputs/tail_tests_by_horizon.csv
  outputs/model_implied_tail_by_horizon.csv
  outputs/oos_1d_forecasts.csv outputs/oos_10d_forecasts.csv outputs/refit_log.csv
  outputs/subperiod_hit_rates.csv outputs/seed_robustness.csv
  outputs/fig*.png
  data/etf_prices.csv data/download_meta.txt
)
for f in "${FILES[@]}"; do
  mkdir -p "$STAGE/$NAME/$(dirname "$f")"
  cp "$f" "$STAGE/$NAME/$f"
done
(cd "$STAGE" && zip -qr "../$(basename "$OUT")" "$NAME")
rm -rf "${STAGE:?}"
ls -lh "$OUT"
unzip -l "$OUT" | tail -1
