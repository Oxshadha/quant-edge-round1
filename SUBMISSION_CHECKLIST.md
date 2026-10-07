# SAIFA Quant Edge 1.0, Round 1: submission checklist (Team Gmora)

- [ ] Tracking code: **SAIFA-2026-75323CFE**
- [ ] Team Leader email entered correctly
- [ ] Submission type: **General / Final Submission**
- [ ] Upload `submission/QuantEdge_Gmora_SAIFA-2026-75323CFE.zip` (about 1.3 MB, limit 25 MB)
- [ ] Upload the same ZIP (or its unzipped folder) to Google Drive, share as "Anyone with the link can view"
- [ ] Paste the Drive link into "External Project Link" and test it in a private browser window
- [ ] Submit before the deadline

## What the ZIP contains (brief requirement -> file)

Only what the Challenge Book asks for. Built by `bash scripts/make_submission_zip.sh`.

| Requirement in the Challenge Book | In the ZIP |
|---|---|
| Report, PDF, max 10 pages excl. cover and references | `QuantEdge_Round1_Report.pdf` (ZIP root) |
| Code, Python, complete and ready to run | `src/` |
| One command reproduces every number and figure | `python -m src.run_all` |
| README | `README.md` |
| Dependency list | `requirements.txt` |
| Data used, or a script that downloads it | `data/etf_prices.csv` (and `src/data.py` re-downloads it) |
| AI-use disclosure | Report Appendix A and README |
