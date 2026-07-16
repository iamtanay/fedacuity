# FedAcuity — Paper

**Status:** complete — under submission to arXiv (cs.LG, cross-list cs.CY + stat.ML)
**Format:** IEEE Transactions (`IEEEtran` document class), 8 pages
**Journal target:** IEEE Journal of Biomedical and Health Informatics (JBHI)

---

## Files

| File | Purpose |
|---|---|
| `main.tex` | Full paper — all sections filled with final results |
| `references.bib` | BibTeX entries (20 references) |
| `main.bbl` | Pre-built bibliography (required by arXiv, which does not run bibtex) |
| `figures/` | All 6 paper figures (Fig 1 PNG + Figs 2–6 PDF, 300 DPI) |
| `fedacuity_arxiv.zip` | Ready-to-upload arXiv bundle: `main.tex` + `main.bbl` + `figures/` |

---

## How to Compile

```bash
latexmk -pdf main.tex        # recommended
latexmk -c                   # clean build artefacts
```

The arXiv bundle is verified against arXiv's build path (pdflatex only, no bibtex):

```bash
pdflatex main.tex && pdflatex main.tex   # uses main.bbl
```

Or upload `main.tex` + `references.bib` + `figures/` to Overleaf.

---

## Figures (all generated programmatically from the pipeline)

| Figure | Script |
|---|---|
| Fig 1 — System Architecture | polished export (programmatic base: `src/evaluation/architecture_figure.py`) |
| Fig 2 — MIMIC-IV Fidelity | `src/data/fidelity.py` |
| Fig 3 — AUC Across Rounds | `src/evaluation/figures.py` |
| Fig 4 — Five-Model Comparison | `src/evaluation/figures.py` |
| Fig 5 — Privacy-Utility Tradeoff | `src/dp/epsilon_sweep.py` |
| Fig 6 — XAI Radar Chart | `src/xai/run_xai_audit.py` |

Statistics: paired instance-level bootstrap CIs from `src/evaluation/bootstrap_ci.py`.

---

## Venues

1. arXiv preprint (current)
2. IEEE JBHI — primary journal target
3. JAMIA · MLHC 2026 · ACM FAccT — alternatives
