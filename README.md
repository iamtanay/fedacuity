# FedAcuity 🏥🔒

> **Privacy-Preserving Federated Learning with Explainability Auditing**
> for Staffing-Acuity Mismatch Prediction in Long-Term Care

*Tanay Kashyap · Independent Researcher*

📄 **Paper:** preprint under submission to arXiv — LaTeX source in [`paper/`](paper/), submission bundle at `paper/fedacuity_arxiv.zip`.

![FedAcuity three-layer architecture](final%20architecture.png)

---

## The Problem

**87% of nursing homes** report moderate-to-high staffing shortages — yet **zero cross-facility predictive tools** exist for staffing-acuity mismatch in Long-Term Care (LTC).

HIPAA prohibits sharing resident records, making centralised ML infeasible. FedAcuity lets facilities collaborate by sharing **model parameters only** — raw resident data never leaves a facility.

---

## Three Standalone Contributions

| # | Contribution | What It Does |
|---|---|---|
| **C1** | 🌐 Domain-Driven Clustered FL | Care-type clustering (MC / SNF / IL) for extreme non-IID data, with a differential-privacy feasibility analysis (Opacus DP-SGD) |
| **C2** | 🧪 Synthetic LTC Benchmark | CTGAN-generated 10-facility dataset, honestly anchored to real MIMIC-IV (205k admissions) via within-cohort calibration |
| **C3** | 🔍 XAI Audit Scorecard | Four SHAP dimensions — D1 Fidelity · D2 Stability · D3 Fairness · D4 Plausibility — computed on the model each strategy deploys |

---

## Headline Results

Held-out facilities 6 (SNF) + 9 (IL), 50 rounds, SEED=42 throughout:

| Strategy | AUC-ROC | Note |
|---|---|---|
| **Clustered FL (ours)** | **0.9827** | matches the oracle |
| Centralised Oracle | 0.9824 | HIPAA-violating upper bound |
| FedAvg / FedProx | 0.9685 | single global model |

- **CFL − FedAvg:** ΔAUC +0.0142 — paired instance-level bootstrap 95% CI [−0.001, +0.035], CFL higher in **96.2%** of 2,000 resamples.
- **The decisive result is fairness (D3):** the single global model FedAvg deploys misses **78% of Memory-Care understaffing days** (TPR 0.22 vs CFL 0.92); equalized-odds gap 0.39 → **0.18** under CFL.
- **Differential privacy:** ε = 10 recommended — 14.9% AUC degradation (mean over 5 paired seeds, monotonic in ε).

> ⚖️ **Honest framing:** this is a *controlled benchmark* — the mismatch label is a known deterministic function of the features, so absolute AUCs are structurally high for every method. All claims are **comparative** between aggregation strategies; see the paper's Limitations section.

---

## Quickstart

Requires **Python 3.12**.

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -c "import flwr, sdv, xgboost, shap, opacus; print('All dependencies OK')"
```

---

## Reproducing Every Number in the Paper

Run in this order (each step feeds the next):

```bash
python -m src.data.generator              # 1. CTGAN synthetic data (~30 min CPU)
python -m src.data.fidelity               # 2. Fidelity vs MIMIC-IV → Fig 2
python -m src.fl.simulation --strategy all --rounds 50   # 3. All 5 FL strategies
python -m src.evaluation.eval_held_out    # 4. Held-out Table II metrics
python -m src.evaluation.figures          # 5. Figs 3–4
python -m src.evaluation.bootstrap_ci     # 6. Paired bootstrap CIs (canonical uncertainty)
python -m src.dp.epsilon_sweep            # 7. DP sweep (5 paired seeds) → Fig 5
python -m src.xai.run_xai_audit           # 8. SHAP + D1–D4 + scorecard → Fig 6
python -m src.evaluation.architecture_figure   # 9. Fig 1

pytest tests/ -v                          # 73 tests
```

Individual FL strategies: `--strategy {local|centralised|fedavg|fedprox|clustered}`.

---

## Repository Structure

```
fedacuity/
├── config.yaml                    # Single source of truth: hyperparams, paths, SEED=42
├── src/
│   ├── data/
│   │   ├── schema.py              # Feature specs, non-IID distributions, label definition
│   │   ├── generator.py           # CTGAN pipeline → 10 facility CSVs
│   │   ├── fidelity.py            # KS / Frobenius / TSTR vs MIMIC-IV
│   │   ├── mimic_preprocessor.py  # MIMIC-IV elderly-subset extraction
│   │   ├── mimic_analysis.py      # Within-MIMIC-IV cohort calibration
│   │   └── loaders.py             # Per-facility stratified 60/20/20 splits
│   ├── fl/
│   │   ├── client.py              # FedAcuityClient — XGBoost local training
│   │   └── simulation.py          # All 5 strategies; prediction-consensus aggregation
│   ├── dp/
│   │   └── epsilon_sweep.py       # Opacus DP-SGD, ε ∈ {1,2,5,10,∞}, 5 paired seeds
│   ├── xai/
│   │   ├── shap_pipeline.py       # SHAP over each strategy's deployed model
│   │   ├── d1_fidelity.py … d4_plausibility.py   # The four audit dimensions
│   │   ├── run_xai_audit.py       # Orchestrator → xai_audit_raw.json + Fig 6
│   │   └── scorecard.py           # Normalised scorecard + radar chart
│   └── evaluation/
│       ├── eval_held_out.py       # Table II held-out metrics
│       ├── bootstrap_ci.py        # Paired instance-level bootstrap CIs
│       ├── metrics.py · figures.py · architecture_figure.py · logger.py
├── paper/                         # IEEEtran LaTeX source + figures + arXiv bundle
├── data/synthetic/                # Generated facility CSVs (regenerable)
├── results/                       # Figures, tables, logs (regenerable)
├── notebooks/                     # 01 literature map · 02 EDA/schema · 03 MIMIC exploration
└── tests/                         # 73 unit tests
```

---

## Tech Stack

`flwr` (Flower FL) · `sdv`/CTGAN · `xgboost` · `torch` + `opacus` (DP-SGD) · `shap` · `scikit-learn` · `scipy` · Python 3.12

---

## Data Access

MIMIC-IV requires **PhysioNet credentialed access** ([physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/)). It is used **only as an external fidelity anchor** — no MIMIC-IV records train any model. If absent, `fidelity.py` falls back to a synthetic holdout.

⚠️ **No real PHI is used anywhere in this project.** All facility data is synthetic.

---

*Built with ❤️ for Long-Term Care · Privacy First · Explainability Always*
