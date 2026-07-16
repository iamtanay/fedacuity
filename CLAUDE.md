# CLAUDE.md — FedAcuity

This file is the authoritative context file for AI assistants working on this codebase.
Read it before doing anything. Then read **PLAN.md** for the active week-by-week execution plan and task checklist.

---

## What This Project Is

**FedAcuity** is Tanay Kashyap's M.Tech AI/ML dissertation at BITS Pilani.
It is a privacy-preserving Federated Learning (FL) framework for predicting **staffing-acuity mismatch**
in Long-Term Care (LTC) facilities. The HIPAA constraint means resident records cannot leave any facility —
only model weights travel over the network.

**Three research contributions:**
- **C1** — Domain-driven Clustered FL (care-type-aware aggregation + differential privacy)
- **C2** — CTGAN synthetic LTC dataset with MIMIC-IV fidelity validation
- **C3** — 4-dimension XAI Audit Scorecard (fidelity, stability, fairness, plausibility via SHAP)

**Five model variants compared:**
Local (no federation) · Centralised Oracle · FedAvg · FedProx · Clustered FL (C1)

**Target publication:** IEEE JBHI (Journal of Biomedical and Health Informatics)

---

## Environment

- **Python 3.12** + plain `venv` (NOT conda)
- **Platform:** Windows 11, PowerShell
- Activate: `venv\Scripts\Activate.ps1`  (venv name is `venv`, not `.venv`)
- All deps in `requirements.txt` — latest versions as of April 2026, Python 3.12 compatible
- Core packages: `flwr` (Flower FL), `sdv` (CTGAN), `xgboost`, `shap`, `opacus`, `torch`, `sklearn`, `scipy`, `matplotlib`
- Verify imports: `python -c "import flwr, sdv, xgboost, shap, opacus; print('All good')"`
- `SEED = 42` everywhere — do not change this; it controls CTGAN, all train/val/test splits, and perturbation tests

---

## Directory Structure

```
fedacuity/
├── CLAUDE.md                      ← this file
├── CONTEXT.md                     ← session diary (read for history)
├── config.yaml                    ← single source of truth for all hyperparams
├── requirements.txt
│
├── src/
│   ├── config.py                  ← loads config.yaml; use `from src.config import cfg`
│   ├── data/
│   │   ├── schema.py              ← feature specs, non-IID dists, cluster assignments, label fn
│   │   ├── generator.py           ← CTGAN pipeline → 10 facility CSVs
│   │   ├── fidelity.py            ← KS-test, Frobenius norm, TSTR vs MIMIC-IV
│   │   └── loaders.py             ← per-facility 60/20/20 stratified splits
│   ├── fl/
│   │   ├── client.py              ← FedAcuityClient (Flower NumPyClient, XGBoost local model)
│   │   ├── clustered_fl.py        ← ClusteredFLServer — C1 core contribution
│   │   └── simulation.py          ← CLI runner for all 5 strategies
│   ├── dp/
│   │   └── epsilon_sweep.py       ← Opacus DP-SGD on PyTorch NN, ε ∈ {1,2,5,10,∞}
│   ├── xai/
│   │   └── scorecard.py           ← XAI Audit Scorecard + radar chart (Fig 6)
│   └── evaluation/
│       └── logger.py              ← ResultsLogger (JSON + CSV, append-only)
│
├── data/
│   ├── synthetic/                 ← generated CSVs: facility_NN_TYPE.csv + all_facilities.parquet
│   ├── mimic_iv/                  ← MIMIC-IV elderly subset (credentialed access pending)
│   └── processed/
│
├── results/
│   ├── figures/                   ← Figs 1–6 (PNG + PDF at 300 dpi)
│   ├── tables/                    ← CSV/JSON results, XAI scorecard
│   └── logs/                      ← per-round FL metrics (ResultsLogger output)
│
├── notebooks/
│   ├── 01_literature_map.ipynb    ← 18-paper annotated bibliography, gap statement
│   ├── 02_eda_schema.ipynb        ← EDA + schema validation + non-IID visualisation
│   └── 03_mimic_exploration.ipynb ← (pending MIMIC-IV access)
│
├── paper/
│   ├── main.tex                   ← IEEE JBHI scaffold (IEEEtran, 9 sections)
│   ├── references.bib             ← 17 BibTeX entries (2 stubs remain)
│   └── figures/                   ← paper figures go here (populated by experiment scripts)
│
├── docs/
│   └── architecture.md            ← full architecture + ASCII diagram + design decisions table
│
└── tests/
    ├── test_schema.py             ← 35 unit tests (no data needed)
    └── test_loaders.py            ← 25 unit tests (data-dependent tests skip gracefully)
```

---

## Key Invariants — Do Not Change Without Understanding

### Data / Facility Setup
- **10 facilities**: IDs 0–9. Care types: `{0,1,2}→MC`, `{3,4,5,6}→SNF`, `{7,8,9}→IL`
- **Held-out facilities: 8 and 9 (both IL)** — never seen during any FL training round, only used in final evaluation
- **Cluster assignments** are defined in two places that must stay in sync: `config.yaml → fl.clustered.clusters` and `src/data/schema.py → CLUSTER_ASSIGNMENTS`
- **Label definition**: `staffing_mismatch = 1` when `(ADL_demand × census) / total_nursing_hours > threshold`. Threshold is calibrated per care type to hit the target mismatch rates (MC≈40%, SNF≈28%, IL≈12%). See `src/data/schema.py:compute_mismatch_label()` and `src/data/generator.py:_calibrate_threshold()`
- **Splits**: 60% train / 20% val / 20% test, stratified by label, per facility. Seeded with `SEED + facility_id` so each facility gets a deterministic but distinct shuffle
- **3 years × 365 days = ~1095 rows per facility** (CTGAN generates then adds facility noise)

### FL Architecture
- **XGBoost** is the primary local model — better for tabular data, native SHAP support
- **PyTorch NN** (`StaffingNN` in `src/dp/epsilon_sweep.py`) is secondary and used **only** for Opacus DP (Opacus doesn't support XGBoost gradient clipping)
- XGBoost federation works via byte serialization: `serialize_xgb_model()` / `deserialize_xgb_model()` in `src/fl/client.py`
- FedProx proximal term is noted in config and metrics but **not implemented as a true gradient penalty for XGBoost** — it's applied correctly only in the PyTorch NN variant. This is a known limitation documented in the code.

### Config
- `config.yaml` is the single source of truth. **Never hardcode hyperparams in source files.** Read everything through `from src.config import cfg`

---

## Known Stubs / Placeholders (Not Bugs)

These are intentional scaffolds to be filled in later phases:

| Location | What it is | When to fix |
|---|---|---|
| `src/xai/scorecard.py:load_xai_results()` | Uses hardcoded placeholder scores when `results/tables/xai_audit_raw.json` is missing | Weeks 10–12 after d1–d4 modules run |
| `paper/main.tex` | Sections VII/VIII still have `% TBD` placeholders pending XAI results | After C3 modules run |
| `paper/references.bib` | Two `% TODO` stubs: `bates2021ml_ltc` and `dellefield2015staffing` | Any time now — entries identified in notebook 01 |
| ~~`src/xai/shap_pipeline.py`~~ | DONE (Session 8) — real SHAP over all 5 deployed models | — |
| ~~`src/xai/d1_fidelity.py` … `d4_plausibility.py`~~ | DONE (Session 8); `run_xai_audit.py` orchestrates + `scorecard.py` reads real `xai_audit_raw.json` | — |
| `src/dp/opacus_wrapper.py` | Does not exist yet | Week 9 |

---

## Common Commands

All commands assume the venv is active (`venv\Scripts\Activate.ps1`).

```powershell
# Verify environment
python -c "import flwr, sdv, xgboost, shap, opacus; print('All good')"

# Verify schema (no data needed)
python -m src.data.schema

# Run schema tests (no data needed)
pytest tests/test_schema.py -v

# Run all tests (loader tests auto-skip if data not generated)
pytest tests/ -v

# Skip data-dependent tests explicitly
pytest tests/ -v -m "not requires_data"

# Generate synthetic data (~30min on CPU due to CTGAN training)
python -m src.data.generator

# Fidelity validation (requires synthetic data + ideally MIMIC-IV)
python -m src.data.fidelity

# FL simulation — individual strategies
python -m src.fl.simulation --strategy local
python -m src.fl.simulation --strategy centralised
python -m src.fl.simulation --strategy fedavg --rounds 50
python -m src.fl.simulation --strategy fedprox --mu 0.1 --rounds 50
python -m src.fl.simulation --strategy clustered --rounds 50

# Run all 5 strategies sequentially
python -m src.fl.simulation --strategy all

# DP epsilon sweep (requires synthetic data)
python -m src.dp.epsilon_sweep

# XAI Scorecard (uses placeholder scores until d1–d4 modules run)
python -m src.xai.scorecard

# Jupyter notebooks
jupyter notebook
```

---

## Testing Approach

- `pytest tests/test_schema.py` — 35 tests, no synthetic data needed, fast (sub-second)
- `pytest tests/test_loaders.py` — 25 tests; data-dependent tests decorated with `@requires_data` and auto-skip if `data/synthetic/` has no CSVs. The mock-data tests run without data.
- Run schema tests routinely. Run loader tests after `python -m src.data.generator` completes.
- Test files pending: `test_generator.py` (Week 4), `test_fidelity.py` (Week 5), `test_fl.py` (Week 6)

---

## Data Pipeline Order

Running out of order will fail with `FileNotFoundError`. Correct order:

```
1. python -m src.data.generator        → produces data/synthetic/*.csv + all_facilities.parquet
2. python -m src.data.fidelity         → produces results/tables/fidelity_*.{csv,json} + Fig 2
3. python -m src.fl.simulation --strategy all   → produces results/logs/results_*.{json,csv}
4. python -m src.dp.epsilon_sweep      → produces results/tables/dp_epsilon_sweep.csv + Fig 5
5. [implement d1–d4 XAI modules]       → produces results/tables/xai_audit_raw.json
6. python -m src.xai.scorecard         → produces results/tables/xai_audit_scorecard.* + Fig 6
```

---

## Paper Figures

| Figure | Generated by | Status |
|---|---|---|
| Fig 1 — System Architecture | `src/evaluation/architecture_figure.py` → `results/figures/fig1_architecture.{png,pdf}` | Generated (programmatic 3-layer diagram; in paper as double-col figure*) |
| Fig 2 — MIMIC-IV Fidelity | `src/data/fidelity.py` → `results/figures/fig2_fidelity_distributions.{png,pdf}` | Generated (Frobenius 0.2196, TSTR gap 0.0199) |
| Fig 3 — FL Convergence Curves | `src/evaluation/figures.py` | Generated |
| Fig 4 — Five-Model Bar Chart | `src/evaluation/figures.py` | Generated (uses held-out AUC from fl_held_out_metrics.json) |
| Fig 5 — Privacy-Utility Tradeoff | `src/dp/epsilon_sweep.py` → `results/figures/fig5_dp_privacy_utility.{png,pdf}` | Generated (ε=10 recommended) |
| Fig 6 — XAI Radar Chart | `src/xai/scorecard.py` → `results/figures/fig6_xai_radar.{png,pdf}` | Generated (REAL SHAP via `run_xai_audit.py`; CFL D3 fairness advantage) |

All figures are saved as both PNG (300 dpi) and PDF for Overleaf.

---

## Paper / LaTeX

- Format: `IEEEtran` document class, journal mode, IEEE JBHI target
- Files: `paper/main.tex` + `paper/references.bib`
- Compile: upload both to Overleaf (they are self-contained)
- Secondary venues: JAMIA · MLHC 2026 · ACM FAccT

---

## Architecture Summary

Three-layer system:

**Layer 1 — Facility Edge:** Each facility runs `FedAcuityClient` (Flower `NumPyClient`). XGBoost trains locally. Only serialised model bytes (~50–200 KB) leave the facility per round. Raw resident data never transmitted.

**Layer 2 — Aggregation Server:** Flower server with three interchangeable strategies — FedAvg, FedProx (proximal term in config), Clustered FL (`ClusteredFLServer`). Clustered FL maintains 3 independent global models (one per care type cluster). Intra-cluster FedAvg only.

**Layer 3 — XAI / Evaluation:** Post-hoc SHAP analysis via `TreeExplainer`. Four audit dimensions (D1 Fidelity · D2 Stability · D3 Fairness · D4 Plausibility) assembled into a normalised scorecard and radar chart.

---

## Non-IID Heterogeneity

Deliberately engineered distributional shift across care types — this is what motivates Clustered FL:

| Feature | MC (Memory Care) | SNF (Skilled Nursing) | IL (Independent Living) |
|---|---|---|---|
| `adl_cognition` mean | 4.5 | 2.5 | 1.0 |
| `medication_count` mean | 11 | 9 | 5 |
| `nursing_hours_rn` mean | 2.5 | 3.0 | 1.0 |
| Mismatch rate target | ~40% | ~28% | ~12% |
| RUG category mode | 6 (high) | 5 (moderate) | 2 (low) |

Global FedAvg must fit all three simultaneously → suboptimal per care type. CFL avoids this.

---

## XAI Audit Dimensions

| Dimension | What it measures | Key parameter |
|---|---|---|
| D1 Fidelity | Spearman ρ of top-10 SHAP feature ranks vs centralised oracle | Target ρ ≥ 0.75 |
| D2 Stability | Mean absolute SHAP shift under ±5% Gaussian noise (100 perturbations) | Lower is better |
| D3 Fairness | Equalized odds + demographic parity across MC/SNF/IL subgroups | Both metrics |
| D4 Plausibility | % of top-5 SHAP features matching LTC clinical literature | Target ≥ 60% |

D4 literature features: `adl_mobility`, `adl_cognition`, `medication_count`, `fall_risk_score`, `pain_assessment_score`, `mds_adl_summary`, `rug_category`.

---

## What Claude Should and Should Not Do

**Do:**
- Follow the 60/20/20 stratified split convention — never change split ratios without updating `config.yaml`
- Keep `SEED = 42` in all new randomness (`np.random.default_rng(SEED)`, `random_state=SEED`)
- Read hyperparams from `cfg` dict, never hardcode them in source
- When adding new XAI modules (`d1_fidelity.py` etc.), write their output to `results/tables/xai_audit_raw.json` in the format `scorecard.py:load_xai_results()` expects
- When new figures are generated, save both PNG and PDF to `results/figures/`
- Keep `HELD_OUT_FACILITIES = [8, 9]` out of all FL training code paths

**Do not:**
- Add training data from held-out facilities 8 and 9 to any FL round
- Change cluster assignments (MC/SNF/IL → facility IDs) without updating both `config.yaml` and `schema.py`
- Implement Opacus DP on XGBoost — it doesn't work; use the PyTorch `StaffingNN` path
- Hardcode epsilon values — they come from `cfg["dp"]["epsilon_values"]`
- Claim MIMIC-IV as training data — it is a statistical anchor for fidelity validation only
- Create `*.md` documentation files unless explicitly asked

---

## Current Status Snapshot (as of Session 8 — 15 Jul 2026)

**ALL THREE CONTRIBUTIONS COMPLETE.** Session 8 built the entire C3 XAI Audit Scorecard on real SHAP, hardened the paper, and produced final deliverables:
- **C3 built:** `src/xai/shap_pipeline.py` + `d1_fidelity.py`, `d2_stability.py`, `d3_fairness.py`, `d4_plausibility.py` + `run_xai_audit.py` orchestrator. Real SHAP (no placeholders). Fig 6 radar + `xai_audit_scorecard.{csv,tex}` generated.
- **DP hardened:** epsilon_sweep rewritten to average over 5 PAIRED seeds (same seed set across ε) → now monotonic with error bars. Canonical changed: no-DP 0.9812, ε=10 → 0.8347 (14.9% drop).
- **Paper hardened:** `paper/main.tex` — Section VI filled with real D1–D4 + Fig 6 + scorecard table; Section VII XAI results written; stale held-out refs / DP numbers / conclusion / fidelity section all fixed. Compiles clean (7 pages, 0 undefined, 20 refs).
- **Final deliverables:** `reports/FedAcuity_Final_Slides.pptx` (15 slides) + `reports/FedAcuity_Final_Report.docx` (via `src/presentation/final_slides.py` + `final_report.py`).
- `/validate-research` 15/15 PASS; 73/73 tests pass.

**Canonical FL numbers (held-out facilities 6 (SNF) + 9 (IL), 50 rounds):**
- CFL: AUC 0.9827, F1 0.8408 | SNF: 0.9917 | IL: 0.9693
- FedAvg: AUC 0.9685, F1 0.8539 | SNF: 0.9881 | IL: 0.9428
- CFL vs FedAvg: +1.42pt overall (+0.36pt SNF, +2.65pt IL) | Mann-Whitney U=400, p<0.001
- Centralised oracle: AUC 0.9824, F1 0.8555 | IL Local (fac 7): 0.9643 | SNF Local (fac 3): 0.9823
- Fidelity C2: MIMIC-IV cohort calibration — 27.2% post-acute discharge ≈ 28% SNF mismatch target (within 0.8%)
- DP: ε=10 recommended — no-DP 0.9812 → 0.8347 (14.9% drop, mean over 5 seeds, monotonic)

**Canonical C3 XAI numbers (real SHAP; scorecard normalised [0,1]):**
- D1 Fidelity (Spearman ρ vs oracle): local 0.85, fedavg 0.85, CFL 0.82 (all ≥0.75 target)
- D2 Stability (relative index): fedavg 1.00, centralised 0.86, local/CFL 0.78 (global models more stable — honest)
- D3 Fairness (1 − equalized-odds gap): **CFL 0.82 vs FedAvg 0.61** — headline. FedAvg MC TPR 0.22, IL AUC 0.67; CFL subgroup AUC ≥0.96, TPR 0.92/0.80/0.58. EO gap FedAvg 0.39 → CFL 0.18.
- D4 Plausibility: all 1.00 (top-5 features all clinically-established; note literature_features expanded to include CMS nurse-HPRD + census)
- KEY NARRATIVE: CFL wins the decisive fairness axis, matches on fidelity/plausibility, trades a little stability. NOT a "wins everything" story — deliberately honest.

**Training set: 8 facilities (IDs 0–5, 7–8) | Held-out: facility 6 (SNF) + facility 9 (IL)**

**Remaining optional polish:** Fig 1 (architecture) still a draw.io placeholder box in paper; XAI audit explains the deployed consensus model (disclosed as a limitation).
