# FedAcuity 🏥🔒

> **Privacy-Preserving Federated Learning with Explainability Auditing**
> for Staffing-Acuity Mismatch Prediction in Long-Term Care

*Tanay Kashyap · M.Tech AI/ML · Work Integrated Learning Programme*

---

## The Problem

**87% of nursing homes** report moderate-to-high staffing shortages — yet **zero cross-facility predictive tools** exist for staffing-acuity mismatch in Long-Term Care (LTC).

HIPAA prohibits sharing resident records, making centralised ML **illegal**.

FedAcuity solves this: facilities collaborate by sharing **model weights only**, never patient data.

---

## Three Standalone Contributions

| # | Contribution | What It Does |
|---|---|---|
| **C1** | 🌐 FedAcuity FL System | Domain-driven clustered federation for LTC non-IID data + Differential Privacy |
| **C2** | 🧪 Synthetic Data Validation | CTGAN generation + MIMIC-IV fidelity proof (KS-test, Frobenius norm, TSTR) |
| **C3** | 🔍 XAI Audit Scorecard | 4-dimension audit: Fidelity · Stability · Fairness · Plausibility |

---

## Quickstart

### Prerequisites

Requires **Python 3.11** — check with:
```bash
python --version
```

### 1. Create & Activate Virtual Environment

```bash
python3.11 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import flwr, sdv, xgboost, shap, opacus; print('All dependencies OK')"
```

✅ If you see `All dependencies OK`, you're ready to go.

---

## Running the Pipeline

### Phase 2 — Synthetic Data Generation & Validation
```bash
python -m src.data.generator     # CTGAN generation (M1.2)
python -m src.data.fidelity       # KS-test, Frobenius norm, TSTR (M1.4)
```

### Phase 3 — Federated Learning Simulation
```bash
# Run individual strategies
python -m src.fl.simulation --strategy local
python -m src.fl.simulation --strategy centralised
python -m src.fl.simulation --strategy fedavg
python -m src.fl.simulation --strategy fedprox --mu 0.1
python -m src.fl.simulation --strategy clustered

# Or run all strategies in one shot
python -m src.fl.simulation --all
```

### Phase 3 — Differential Privacy Sweep
```bash
python -m src.dp.epsilon_sweep    # ε ∈ {1, 2, 5, 10, ∞}
```

### Phase 4 — XAI Audit Scorecard
```bash
python -m src.xai.scorecard       # 4-dimension audit + radar chart
```

### Launch Jupyter
```bash
jupyter notebook
```

---

## Repository Structure

```
fedacuity/
├── 📁 data/
│   ├── raw/                  # Schema definitions, seed data
│   ├── synthetic/            # CTGAN-generated facility datasets (CSV + parquet)
│   ├── mimic_iv/             # MIMIC-IV elderly subset (credentialed access required)
│   └── processed/            # Train/val/test splits per facility
│
├── 📓 notebooks/
│   ├── 01_literature_map.ipynb
│   ├── 02_schema_design.ipynb
│   ├── 03_ctgan_generation.ipynb
│   ├── 04_fidelity_validation.ipynb
│   ├── 05_fl_fedavg_baseline.ipynb
│   ├── 06_fl_comparison.ipynb
│   ├── 07_dp_sweep.ipynb
│   ├── 08_shap_pipeline.ipynb
│   ├── 09_xai_audit.ipynb
│   └── 10_results_figures.ipynb
│
├── 🐍 src/
│   ├── config.py             # Config loader (reads config.yaml)
│   ├── data/
│   │   ├── schema.py         # Data schema, care-type specs, label definition (M1.1)
│   │   ├── generator.py      # CTGAN generation pipeline (M1.2)
│   │   ├── fidelity.py       # KS-test, Frobenius norm, TSTR (M1.4)
│   │   └── loaders.py        # Per-facility data loaders with train/val/test splits
│   ├── fl/
│   │   ├── client.py         # Flower FlowerClient — XGBoost local training (M2.1/2.2)
│   │   ├── clustered_fl.py   # Domain-driven Clustered FL by care type (M2.5 — C1 core)
│   │   └── simulation.py     # End-to-end simulation runner — all 5 strategies (M2.6)
│   ├── dp/
│   │   └── epsilon_sweep.py  # Opacus DP + ε sweep (M3)
│   ├── xai/
│   │   └── scorecard.py      # 4-dimension XAI Audit Scorecard + radar chart (M4.6)
│   └── evaluation/
│       └── logger.py         # Centralised results logger — pandas + JSON (M5.1)
│
├── 📊 results/
│   ├── figures/              # All paper figures (PNG + PDF)
│   ├── tables/               # CSV + LaTeX tables
│   └── logs/                 # Per-run metric logs (JSON)
│
├── 📄 docs/
│   └── architecture.md       # System architecture document
│
├── 🧪 tests/                 # Unit tests
│
├── .venv/                    # Virtual environment (gitignored)
├── .gitignore
├── requirements.txt
├── config.yaml               # Central config: hyperparams, paths, seeds
├── CONTEXT.md                # Project state for AI-assisted development
└── README.md
```

---

## 16-Week Execution Plan

| Phase | Weeks | Focus | Deliverable |
|---|---|---|---|
| 🏗️ Foundation | 1–3 | Literature, MIMIC-IV, Architecture | Architecture doc + lit map |
| 🔧 Data Engineering | 4–5 | CTGAN generation + fidelity | Validated LTC benchmark (**C2 ✓**) |
| 🌐 FL Implementation | 6–9 | All 5 model variants + DP sweep | FL codebase + results (**C1 ✓**) |
| 🔍 XAI Audit | 10–12 | 4-dimension SHAP audit | XAI Audit Scorecard (**C3 ✓**) |
| ✍️ Write-up | 13–16 | Dissertation + paper | IEEE JBHI submission |

---

## Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| 🌐 FL Framework | `flwr` (Flower) | Federated simulation |
| 🧪 Data Generation | `sdv` / `ctgan` | Synthetic LTC dataset |
| 🤖 ML (Primary) | `xgboost` | Tabular prediction |
| 🔦 ML (Secondary) | `torch` | Federated NN for Opacus DP |
| 🔒 Privacy | `opacus` | Differential privacy (DP-SGD) |
| 💡 XAI | `shap` | Explanation fidelity + stability |
| 📐 Fidelity Tests | `scipy.stats` | KS-test vs MIMIC-IV |
| 📏 Evaluation | `scikit-learn` | AUC-ROC, F1, fairness |
| 🐍 Environment | `venv` / Python 3.11 | Reproducibility |

---

## Data Access

MIMIC-IV access requires **PhysioNet credentialed access**.
Apply at: [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)

Place the elderly subset (age ≥ 65, ≥ 3 comorbidities) at:
```
data/mimic_iv/mimic_elderly_subset.parquet
```

> 💡 If access is pending, `fidelity.py` automatically falls back to a synthetic holdout for pipeline testing.

⚠️ **No real PHI is used anywhere in this project.**

---

## Target Venues

| Priority | Venue |
|---|---|
| 🥇 Primary | IEEE Journal of Biomedical and Health Informatics (JBHI) |
| 🥈 Secondary | JAMIA |
| 🎯 Conference | MLHC 2026 |
| 🎯 Conference | ACM FAccT |

---

*Built with ❤️ for Long-Term Care · Privacy First · Explainability Always*
