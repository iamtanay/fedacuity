# CONTEXT.md — FedAcuity

> **Read this at the start of every coding session.**
> This file tells you (and any AI assistant) exactly where the project stands, what was built, what's next, and what decisions were made and why.

---

## What This Project Is

**FedAcuity** is an M.Tech AI/ML dissertation by Tanay Kashyap building a privacy-preserving Federated Learning (FL) framework for predicting staffing-acuity mismatch in Long-Term Care (LTC) facilities. The core constraint: HIPAA means facilities cannot share resident records, so FL enables cross-facility learning via model weights only.

**Three contributions:**
- **C1** — Domain-driven Clustered FL system (care-type-aware aggregation + differential privacy)
- **C2** — CTGAN synthetic LTC dataset + MIMIC-IV fidelity validation (kills the "synthetic data" objection)
- **C3** — 4-dimension XAI Audit Scorecard (fidelity, stability, fairness, plausibility via SHAP)

**Five model variants to compare:** Local (no federation) · Centralised Oracle · FedAvg · FedProx · Clustered FL

---

## Current Status — Session 2

### What was done in Session 1 (Scaffolding)
Full project scaffold created from scratch. All source files, config, and directory structure built. See Session 1 section below for full file list.

### What was done in Session 2

| Item | Status |
|---|---|
| `requirements.txt` updated to latest versions (Python 3.12 compatible) | ✅ Done |
| All packages installed and verified (`import flwr, sdv, xgboost, shap, opacus` passes) | ✅ Done |
| `python -m src.data.schema` smoke test passes | ✅ Done |
| `README.md` rewritten — more engaging, sectioned, emoji-anchored | ✅ Done |
| GitHub repo topics and short description written | ✅ Done |
| `paper/main.tex` — full IEEE JBHI paper scaffold in IEEEtran format | ✅ Done |
| `paper/references.bib` — 17 BibTeX entries for all cited works | ✅ Done |
| `paper/README.md` — compile instructions, figure checklist, venue list | ✅ Done |
| MIMIC-IV PhysioNet application — **in progress** | ⏳ Pending |

---

## File Map — Complete Current State

### Source Code (all ✅ from Session 1)

| File | Module | Status |
|---|---|---|
| `requirements.txt` | — | ✅ Updated to latest versions, Python 3.12 |
| `config.yaml` | — | ✅ Done — all hyperparams centralised here |
| `src/config.py` | — | ✅ Done — config loader |
| `src/data/schema.py` | M1.1 | ✅ Done — feature specs, care-type non-IID distributions, label definition |
| `src/data/generator.py` | M1.2 | ✅ Done — CTGAN generation pipeline, 10 facilities × 3 years |
| `src/data/fidelity.py` | M1.4 | ✅ Done — KS-test, Frobenius norm, TSTR experiment |
| `src/data/loaders.py` | — | ✅ Done — per-facility train/val/test splits |
| `src/fl/client.py` | M2.1/M2.2 | ✅ Done — Flower client with XGBoost |
| `src/fl/clustered_fl.py` | M2.5 | ✅ Done — C1 core: care-type cluster aggregation |
| `src/fl/simulation.py` | M2.6 | ✅ Done — all 5 strategy runner (CLI) |
| `src/dp/epsilon_sweep.py` | M3 | ✅ Done — Opacus DP + ε sweep + Figure 5 |
| `src/xai/scorecard.py` | M4.6 | ✅ Done — XAI Audit Scorecard + radar chart (Figure 6) |
| `src/evaluation/logger.py` | M5.1 | ✅ Done — centralised results logger |
| `src/__init__.py` + all subdir `__init__.py` | — | ✅ Done |

### Paper (added Session 2)

| File | Status |
|---|---|
| `paper/main.tex` | ✅ Done — full IEEE JBHI scaffold, IEEEtran format, 9 sections, 2 equations, Table I filled, Table II stubbed, 6 figure placeholders |
| `paper/references.bib` | ✅ Done — 17 BibTeX entries |
| `paper/README.md` | ✅ Done — compile instructions, figure checklist |
| `paper/figures/` | ⬜ Empty — populated as experiments complete (Weeks 5–12) |

### Docs / Notebooks (not yet created)

| File | When Needed |
|---|---|
| `docs/architecture.md` | Week 2 |
| `notebooks/01_literature_map.ipynb` | Week 1 |
| All other notebooks (02–10) | Weeks 3–12 |

### Not Yet Created — Future Modules

| File | Module | When Needed |
|---|---|---|
| `src/fl/fedavg.py` | M2.3 | Week 6 if needed standalone |
| `src/fl/fedprox.py` | M2.4 | Week 6 if needed standalone |
| `src/xai/shap_pipeline.py` | M4.1 | Week 10 |
| `src/xai/d1_fidelity.py` | M4.2 | Week 10 |
| `src/xai/d2_stability.py` | M4.3 | Week 11 |
| `src/xai/d3_fairness.py` | M4.4 | Week 11 |
| `src/xai/d4_plausibility.py` | M4.5 | Week 12 |
| `src/evaluation/metrics.py` | M5.2 | Week 6 |
| `src/evaluation/figures.py` | M5.2 | Week 8+ |
| `src/dp/opacus_wrapper.py` | M3.1 | Week 9 |
| `tests/` | — | Ongoing |

---

## Key Design Decisions (Don't Change Without Reason)

### Environment
- **Python 3.12 + venv** (installed on Windows, plain virtualenv — NOT conda).
- Activate with `.venv\Scripts\Activate.ps1` (Windows PowerShell).
- All deps in `requirements.txt` are **latest versions as of April 2026** — not pinned to old versions from the original proposal.
- Notable version changes from proposal: `torch==2.11.0`, `sdv==1.36.0`, `ctgan==0.12.1`, `xgboost==3.2.0`, `pandas==2.3.3`, `numpy==2.4.4`.

### Data
- **10 facilities**: 3 MC (Memory Care), 4 SNF (Skilled Nursing), 3 IL (Independent Living)
- **Facility IDs 8 and 9 are HELD-OUT** — never used during FL training, only for final evaluation
- **Cluster assignment** is in `config.yaml → fl.clustered.clusters` and hardcoded in `src/data/schema.py → CLUSTER_ASSIGNMENTS`
- **Label definition**: mismatch = 1 when (ADL demand × census) / total nursing hours > threshold. Threshold calibrated per care type to hit ~30% mismatch rate. See `src/data/schema.py → compute_mismatch_label()`
- **MIMIC-IV** is used only as a statistical anchor for fidelity validation — NOT as training data. No clinical equivalence claimed. Place the elderly subset (age≥65, ≥3 comorbidities) at `data/mimic_iv/mimic_elderly_subset.parquet`. If missing, `fidelity.py` uses a synthetic holdout as proxy. **PhysioNet application in progress.**

### FL Architecture
- **XGBoost** is the primary local model (better for tabular data, interpretable with SHAP)
- **PyTorch NN** is secondary, used exclusively for Opacus DP (Opacus doesn't support XGBoost)
- XGBoost federation uses model serialisation + weight exchange via `serialize_xgb_model()` / `deserialize_xgb_model()` in `src/fl/client.py`
- The Clustered FL `_weighted_average_xgb()` in `src/fl/clustered_fl.py` is currently a **placeholder** — returns largest client's model as proxy. True XGBoost tree merging to implement in Week 8.

### Paper
- **Target journal**: IEEE JBHI (Journal of Biomedical and Health Informatics)
- **Format**: `IEEEtran` document class, journal mode
- **Compile on Overleaf**: Upload `paper/main.tex` + `paper/references.bib` together
- **Secondary venues**: JAMIA → MLHC 2026 → ACM FAccT
- All `% TBD` and `% TODO` comments in `main.tex` mark sections that need real experimental results — fill these in Weeks 8–12

### Config
- **Single source of truth**: `config.yaml`. Never hardcode hyperparams in source files.
- `SEED = 42` everywhere for reproducibility.

---

## 16-Week Plan — Current Week

**Current week: Week 1**

### Immediate Next Actions

| Priority | Task |
|---|---|
| 🔴 High | Apply for MIMIC-IV PhysioNet access — takes ~2 weeks, do NOW: https://physionet.org/content/mimiciv/ |
| 🔴 High | Add `paper/` folder to GitHub repo |
| 🟡 Medium | Upload `paper/main.tex` + `references.bib` to Overleaf and verify it compiles |
| 🟡 Medium | Build `notebooks/01_literature_map.ipynb` — annotated bibliography of 4 core papers |
| 🟡 Medium | Write `docs/architecture.md` — system architecture description (Week 2) |
| 🟢 Low | Run `python -m src.data.generator` — generates 10 facility CSVs (Week 4 goal, can start now) |
| 🟢 Low | Write unit tests in `tests/` for `schema.py` and `loaders.py` |

### Week 4 Goal
`python -m src.data.generator` completes without error and produces 10 facility CSVs in `data/synthetic/`.

---

## Known Issues / TODOs

- `src/fl/clustered_fl.py → _weighted_average_xgb()` is a placeholder. Replace with proper XGBoost tree-merging in Week 8.
- `src/xai/scorecard.py` loads placeholder XAI scores from `results/tables/xai_audit_raw.json` — this file doesn't exist yet. Generated by `d1_fidelity.py` through `d4_plausibility.py` in Weeks 10–12.
- `paper/main.tex` has `% TBD` placeholders in Abstract, Table II, and Section VII/VIII — fill after experiments complete.
- `paper/references.bib` has two `% TODO` citation stubs for LTC staffing ML literature — add real refs during Week 1 literature review.
- `data/mimic_iv/` is empty and gitignored — MIMIC-IV access pending.
- `.gitignore` is in place and reviewed ✅

---

## Quick Commands Reference

```bash
# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install / update deps
pip install -r requirements.txt

# Verify imports
python -c "import flwr, sdv, xgboost, shap, opacus; print('All good')"

# Verify schema
python -m src.data.schema

# Generate synthetic data (Week 4)
python -m src.data.generator

# Validate fidelity (Week 5)
python -m src.data.fidelity

# FL Simulation (Week 6+)
python -m src.fl.simulation --strategy fedavg --rounds 50
python -m src.fl.simulation --strategy clustered --rounds 50
python -m src.fl.simulation --all

# DP Sweep (Week 9)
python -m src.dp.epsilon_sweep

# XAI Scorecard (Week 12)
python -m src.xai.scorecard

# Jupyter
jupyter notebook

# Compile paper (local LaTeX)
cd paper
latexmk -pdf main.tex
```

---

## Paper Figure Checklist

| Figure | Script | Status |
|---|---|---|
| Fig 1 — System Architecture | Manual / draw.io | ⬜ Week 2 |
| Fig 2 — MIMIC-IV Fidelity Distributions | `src/data/fidelity.py` | ⬜ Week 5 |
| Fig 3 — FL Convergence Curves | `src/evaluation/figures.py` | ⬜ Week 8 |
| Fig 4 — Five-Model Bar Chart | `src/evaluation/figures.py` | ⬜ Week 8 |
| Fig 5 — Privacy-Utility Tradeoff | `src/dp/epsilon_sweep.py` | ⬜ Week 9 |
| Fig 6 — XAI Radar Chart | `src/xai/scorecard.py` | ⬜ Week 12 |

---

## Paper Target

**Primary:** IEEE JBHI (Journal of Biomedical and Health Informatics)
**Secondary:** JAMIA · MLHC 2026 · ACM FAccT

---

*Last updated: Session 2 — Requirements fixed, verified working, paper scaffold built*
*Next session: Week 1 literature notebook + docs/architecture.md*
