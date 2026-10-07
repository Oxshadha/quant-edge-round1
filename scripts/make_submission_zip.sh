#!/usr/bin/env bash
# Package only what the Challenge Book asks for: report PDF (at the ZIP root), runnable code, README,
# dependency list and the data used. Run after the pipeline and the report build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="submission/QuantEdge_Gmora_SAIFA-2026-75323CFE.zip"
STAGE="submission/.stage"
rm -rf "${STAGE:?}" submission/*.zip
mkdir -p "$STAGE/src" "$STAGE/data"
cp report/QuantEdge_Round1_Report.pdf "$STAGE/"
cp submission/package/README.md submission/package/requirements.txt "$STAGE/"
cp src/*.py "$STAGE/src/"
cp data/etf_prices.csv "$STAGE/data/"
(cd "$STAGE" && zip -qrX "../$(basename "$OUT")" .)
rm -rf "${STAGE:?}"
ls -lh "$OUT"
unzip -l "$OUT"
