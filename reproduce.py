"""Reproduce every number, table and figure in the report, then rebuild the PDF.

    python reproduce.py            (Windows, macOS, Linux; no internet needed after installing requirements)

Steps: unit tests -> pipeline (src/run_all.py) -> report (scripts/build_report_pdf.py).
The price data is bundled in data/etf_prices.csv, so nothing is downloaded.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(title: str, args: list[str]) -> None:
    print(f"\n=== {title} ===", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"--- {title}: done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    step("1/3 Unit tests", ["-m", "pytest", "-q"])
    step("2/3 Pipeline (about 8-10 minutes)", ["-m", "src.run_all"])
    step("3/3 Report PDF", ["scripts/build_report_pdf.py"])
    print("\nAll outputs regenerated:\n  outputs/                          numbers, tables, forecasts"
          "\n  report/assets/                    figures"
          "\n  report/QuantEdge_Round1_Report.pdf")


if __name__ == "__main__":
    main()
