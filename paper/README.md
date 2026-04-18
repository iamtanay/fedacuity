# FedAcuity — Paper

**Target:** IEEE Journal of Biomedical and Health Informatics (JBHI)
**Format:** IEEE Transactions (`IEEEtran` document class)
**Page limit:** 13 pages (JBHI standard for full papers)

---

## Files

| File | Purpose |
|---|---|
| `main.tex` | Full paper — all sections with placeholders for results |
| `references.bib` | BibTeX entries for all cited works |
| `figures/` | Place all paper figures here (PNG + PDF at 300 DPI) |

---

## How to Compile

```bash
# One-shot compile with bibliography
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or use latexmk (recommended)
latexmk -pdf main.tex

# Clean build artefacts
latexmk -c
```

**Recommended editor:** Overleaf (free, no local LaTeX install needed)
Upload `main.tex` + `references.bib` + `figures/` folder.

---

## Figure Checklist

| Figure | Script | Status |
|---|---|---|
| Fig 1 — System Architecture | Manual / draw.io | ⬜ Week 2 |
| Fig 2 — MIMIC-IV Fidelity Distributions | `src/data/fidelity.py` | ⬜ Week 5 |
| Fig 3 — FL Convergence Curves | `src/evaluation/figures.py` | ⬜ Week 8 |
| Fig 4 — Five-Model Bar Chart | `src/evaluation/figures.py` | ⬜ Week 8 |
| Fig 5 — Privacy-Utility Tradeoff | `src/dp/epsilon_sweep.py` | ⬜ Week 9 |
| Fig 6 — XAI Radar Chart | `src/xai/scorecard.py` | ⬜ Week 12 |

---

## Sections Still Needing Results (fill as experiments complete)

- Abstract: AUC-ROC numbers, DP degradation %, XAI plausibility %
- Table II: Five-model comparison results
- Section VII: All experimental results subsections
- Section VIII: Key findings + discussion
- Hardware/compute specs for reproducibility statement

---

## Secondary Venues (if JBHI rejects)

1. JAMIA — Journal of the American Medical Informatics Association
2. MLHC 2026 — Machine Learning for Healthcare
3. ACM FAccT — Fairness, Accountability, Transparency
