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

## Current Status — Session 1 (Scaffolding Complete)

**What was done in this session:**
Full project scaffold was created from scratch based on the dissertation proposal (FedAcuity_PPT.pptx) and 16-week execution plan (FedAcuity_16Week_Plan.docx).

### Files Created

| File | Module | Status |
|---|---|---|
| `requirements.txt` | — | ✅ Done — venv-based, Python 3.11 |
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
| `README.md` | — | ✅ Done |
| `CONTEXT.md` | — | ✅ Done (this file) |

### Directories Created (Empty — populate as work progresses)

```
data/raw/          ← schema seed files go here
data/synthetic/    ← CTGAN output goes here (auto-created by generator.py)
data/mimic_iv/     ← place mimic_elderly_subset.parquet here
data/processed/    ← future use
notebooks/         ← Jupyter notebooks (create as you go per week)
results/figures/   ← auto-created by figure scripts
results/tables/    ← auto-created by experiment scripts
results/logs/      ← auto-created by logger
docs/              ← architecture.md to write in Week 2
tests/             ← unit tests to write alongside modules
```

### Not Yet Created (Next Sessions)

| File | Module | When Needed |
|---|---|---|
| `src/fl/fedavg.py` | M2.3 | Extracted from simulation.py if needed standalone |
| `src/fl/fedprox.py` | M2.4 | Same |
| `src/xai/shap_pipeline.py` | M4.1 | Week 10 |
| `src/xai/d1_fidelity.py` | M4.2 | Week 10 |
| `src/xai/d2_stability.py` | M4.3 | Week 11 |
| `src/xai/d3_fairness.py` | M4.4 | Week 11 |
| `src/xai/d4_plausibility.py` | M4.5 | Week 12 |
| `src/evaluation/metrics.py` | M5.2 | Week 6 |
| `src/evaluation/figures.py` | M5.2 | Week 8+ |
| `src/dp/opacus_wrapper.py` | M3.1 | Week 9 |
| `notebooks/01_literature_map.ipynb` | — | Week 1 |
| `docs/architecture.md` | — | Week 2 |
| `tests/` | — | Ongoing |

---

## Key Design Decisions (Don't Change Without Reason)

### Environment
- **Python 3.11 + venv** (not conda). Activate with `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\Activate.ps1` (Windows).
- All deps pinned in `requirements.txt`.

### Data
- **10 facilities**: 3 MC (Memory Care), 4 SNF (Skilled Nursing), 3 IL (Independent Living)
- **Facility IDs 8 and 9 are HELD-OUT** — never used during FL training, only for final evaluation
- **Cluster assignment** is in `config.yaml → fl.clustered.clusters` and also hardcoded in `src/data/schema.py → CLUSTER_ASSIGNMENTS`
- **Label definition**: mismatch = 1 when (ADL demand × census) / total nursing hours > threshold. Threshold is calibrated per care type to hit target mismatch rate (~30% overall). See `src/data/schema.py → compute_mismatch_label()`
- **MIMIC-IV** is used only as a statistical anchor for fidelity validation — NOT as training data. No clinical equivalence claimed. Put the elderly subset (age≥65, ≥3 comorbidities) at `data/mimic_iv/mimic_elderly_subset.parquet`. If missing, `fidelity.py` uses a synthetic holdout as a proxy.

### FL Architecture
- **XGBoost** is the primary local model (better for tabular data, interpretable with SHAP)
- **PyTorch NN** is secondary, used exclusively for Opacus DP (Opacus doesn't support XGBoost)
- XGBoost doesn't natively support gradient-based federation — we use model serialisation + weight exchange via `serialize_xgb_model()` / `deserialize_xgb_model()` in `src/fl/client.py`
- The Clustered FL strategy in `src/fl/clustered_fl.py` uses `_weighted_average_xgb()` which is currently a **placeholder** — it returns the largest client's model as a proxy. True XGBoost FL averaging (tree merging) needs to be implemented properly in Week 8.

### Config
- **Single source of truth**: `config.yaml`. Never hardcode hyperparams in source files. Always `from src.config import cfg`.
- `SEED = 42` everywhere for reproducibility.

---

## 16-Week Plan — Current Week

**Current week: Pre-Week 1 (Scaffolding)**

Next immediate actions (Week 1):
1. `source .venv/bin/activate` and `pip install -r requirements.txt`
2. Apply for MIMIC-IV PhysioNet access at https://physionet.org/content/mimiciv/ (takes ~2 weeks — do this NOW)
3. Read core papers: McMahan 2017 (FedAvg), Li 2020 (FedProx), Lundberg 2017 (SHAP), Rieke 2020 (FL clinical AI)
4. Build `notebooks/01_literature_map.ipynb` — annotated bibliography
5. Run `python -m src.data.schema` to verify schema prints correctly

Week 4 goal: `python -m src.data.generator` completes without error and produces 10 facility CSVs in `data/synthetic/`.

---

## Known Issues / TODOs

- `src/fl/clustered_fl.py → _weighted_average_xgb()` is a placeholder. Replace with proper XGBoost tree-merging in Week 8.
- `src/xai/scorecard.py` currently loads **placeholder XAI scores** from `results/tables/xai_audit_raw.json`. This file doesn't exist yet — it will be generated by `d1_fidelity.py` through `d4_plausibility.py` in Weeks 10–12.
- `__init__.py` files are missing from `src/` subdirs — add them if you get module import errors: `touch src/__init__.py src/data/__init__.py src/fl/__init__.py src/dp/__init__.py src/xai/__init__.py src/evaluation/__init__.py`
- `src/fl/simulation.py` imports `src/evaluation/logger.py` — make sure to run `python -m src.fl.simulation` (not `python src/fl/simulation.py`) to preserve module paths.
- `.gitignore` not yet created — add one before first `git push`.

---

## Quick Commands Reference

```bash
# Activate environment
source .venv/bin/activate           # Mac/Linux
.venv\Scripts\Activate.ps1          # Windows

# Install / update deps
pip install -r requirements.txt

# Fix module imports (run once if you see ModuleNotFoundError)
touch src/__init__.py src/data/__init__.py src/fl/__init__.py \
      src/dp/__init__.py src/xai/__init__.py src/evaluation/__init__.py

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
```

---

## Paper Target

**Primary:** IEEE JBHI (Journal of Biomedical and Health Informatics)
**Secondary:** JAMIA · MLHC 2026 · ACM FAccT

**Six required paper figures:**
- Fig 1: System Architecture diagram (draw in Week 2, make in Matplotlib/draw.io)
- Fig 2: MIMIC-IV fidelity distributions — `src/data/fidelity.py → plot_ks_distributions()`
- Fig 3: FL convergence curves — `src/evaluation/figures.py` (to build Week 8)
- Fig 4: Five-model comparison bar chart — `src/evaluation/figures.py` (to build Week 8)
- Fig 5: Privacy-utility tradeoff — `src/dp/epsilon_sweep.py → _plot_privacy_utility()`
- Fig 6: XAI Audit radar chart — `src/xai/scorecard.py → plot_radar_chart()`

---

*Last updated: Session 1 — Initial scaffolding*
*Next session: Start Week 1 literature review + MIMIC-IV application*
