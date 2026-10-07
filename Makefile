.PHONY: reproduce install test pipeline report zip clean

PY := .venv/bin/python

# One command: install pinned deps, run tests, regenerate every number/figure, rebuild the PDF.
reproduce: install
	$(PY) reproduce.py

install:
	python3 -m venv .venv
	.venv/bin/pip install -q -U pip
	.venv/bin/pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q

pipeline:
	$(PY) -m src.run_all

report:
	$(PY) scripts/build_report_pdf.py

zip:
	bash scripts/make_submission_zip.sh

clean:
	rm -rf outputs/* report/assets/*
