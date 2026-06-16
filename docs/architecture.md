# FedAcuity — System Architecture

> **Version**: Session 3 | **Author**: Tanay Kashyap | **Date**: April 2026

---

## Overview

FedAcuity is a three-layer privacy-preserving federated learning framework for predicting staffing-acuity mismatch in Long-Term Care (LTC) facilities. The core design constraint is **HIPAA compliance**: no resident records leave any facility. Only model weights traverse the network.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEDACUITY SYSTEM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LAYER 1 — FACILITY EDGE LAYER                                             │
│   ─────────────────────────────────────────────────────────────────────     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│   │Facility 0│  │Facility 1│  │Facility 2│  │Facility 3│  │Facility 4│   │
│   │  (MC)    │  │  (MC)    │  │  (MC)    │  │  (SNF)   │  │  (SNF)   │   │
│   │          │  │          │  │          │  │          │  │          │   │
│   │ XGBoost  │  │ XGBoost  │  │ XGBoost  │  │ XGBoost  │  │ XGBoost  │   │
│   │ local    │  │ local    │  │ local    │  │ local    │  │ local    │   │
│   │ training │  │ training │  │ training │  │ training │  │ training │   │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│        │              │              │              │              │         │
│        └──────────────┴──── weights only ────┴─────┴──────────────┘         │
│                                      │                                      │
│   LAYER 2 — AGGREGATION SERVER LAYER │                                      │
│   ───────────────────────────────────┼──────────────────────────────────    │
│                              ┌───────┴────────┐                            │
│                              │  Flower Server  │                            │
│                              │                 │                            │
│                   ┌──────────┤  Strategy:      ├──────────┐                │
│                   │          │  FedAvg /        │          │                │
│                   │          │  FedProx /       │          │                │
│                   ▼          │  Clustered FL    │          ▼                │
│           ┌───────────┐      └───────┬─────────┘  ┌───────────┐            │
│           │MC Cluster │              │             │IL Cluster │            │
│           │Global     │      ┌───────┴──────┐      │Global     │            │
│           │Model      │      │SNF Cluster   │      │Model      │            │
│           └───────────┘      │Global Model  │      └───────────┘            │
│                              └──────────────┘                               │
│                                                                             │
│   LAYER 3 — XAI / EVALUATION LAYER                                          │
│   ──────────────────────────────────────────────────────────────────────    │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │D1 Fidelity   │  │D2 Stability  │  │D3 Fairness   │  │D4 Plausibility│ │
│   │SHAP rank     │  │Perturbation  │  │Equalized     │  │Literature     │  │
│   │correlation   │  │SHAP shift    │  │odds across   │  │match for top  │  │
│   │vs oracle     │  │test          │  │care types    │  │SHAP features  │  │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                          │                                  │
│                               ┌──────────┴────────┐                        │
│                               │  XAI Audit        │                        │
│                               │  Scorecard        │                        │
│                               │  + Radar Chart    │                        │
│                               └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Facility Edge Layer

### What runs at each facility

Each simulated facility runs a **Flower client** (`src/fl/client.py`) that:

1. Loads its local dataset from `data/synthetic/facility_NN_TYPE.csv`
2. Performs local model training (XGBoost primary, PyTorch NN secondary)
3. Serialises the trained model to bytes (`serialize_xgb_model()`)
4. Returns model bytes + training metrics to the Flower server
5. Receives the updated global model and deserialises it for the next round

### Data that stays local (never transmitted)

- Raw resident records (ADL scores, nursing hours, medication counts, etc.)
- The binary mismatch label (`staffing_mismatch`)
- Facility-specific metadata (census counts, incident logs)

### Data that leaves the facility (weight exchange only)

- Serialised XGBoost booster bytes (~50–200 KB per round)
- Scalar training metrics (AUC-ROC, F1, loss) for convergence logging

### Privacy guarantees

- **No raw data sharing** by design (HIPAA compliance)
- **Differential Privacy** (Opacus DP-SGD) applied to the PyTorch NN variant in Module 3
- DP is implemented at the gradient level before weight transmission

---

## Layer 2 — Aggregation Server Layer

### Flower server (`src/fl/simulation.py`)

The Flower server coordinates the FL simulation. It implements three strategies:

| Strategy | File | Aggregation Logic |
|---|---|---|
| **FedAvg** | `src/fl/simulation.py` | Prediction-consensus weighted ensemble of client models by data size |
| **FedProx** | `src/fl/simulation.py` | FedAvg + proximal regularisation (μ term) on client side |
| **Clustered FL** | `src/fl/simulation.py` | Intra-cluster prediction-consensus ensemble; 3 independent cluster models (MC / SNF / IL) |

### Federation round lifecycle

```
Round t:
  1. Server broadcasts current global model weights to all clients
  2. Each client runs local_epochs (default: 5) of XGBoost training
  3. Each client returns updated weights + num_examples
  4. Server aggregates weights (strategy-dependent)
  5. Server logs round metrics (AUC-ROC, F1, communication bytes)
  6. Repeat for 50 rounds (configurable in config.yaml → fl.rounds)
```

### Cluster assignment (Clustered FL only)

```
MC  Cluster → Facilities: [0, 1, 2]    — Memory Care
SNF Cluster → Facilities: [3, 4, 5, 6] — Skilled Nursing
IL  Cluster → Facilities: [7, 8, 9]    — Independent Living

Held-out (never in training): Facilities 8 and 9
```

---

## Layer 3 — XAI / Evaluation Layer

The XAI Audit Engine operates on the **trained models from Layer 2**. It is entirely post-hoc — no XAI computation happens during FL training.

### Four audit dimensions

| Dimension | Metric | Tool | Module |
|---|---|---|---|
| **D1 Fidelity** | Spearman ρ of top-10 SHAP feature ranks vs centralised oracle | `shap.TreeExplainer` | `src/xai/d1_fidelity.py` |
| **D2 Stability** | Mean absolute SHAP shift under ±5% Gaussian noise | `shap.TreeExplainer` × 100 perturbations | `src/xai/d2_stability.py` |
| **D3 Fairness** | Equalized odds + demographic parity across MC/SNF/IL | `sklearn` | `src/xai/d3_fairness.py` |
| **D4 Plausibility** | % of top-5 SHAP features matching LTC clinical literature | Manual literature cross-reference | `src/xai/d4_plausibility.py` |

### Audit scorecard output

All four D-scores are normalised to [0, 1] and assembled into:
- `results/tables/xai_audit_scorecard.csv` — the main results table
- `results/figures/fig6_xai_radar.png` — radar chart (5 models × 4 dimensions)

---

## Data Flow

```
config.yaml
    │
    ├── src/data/schema.py       ←── Feature specs, non-IID distributions
    │       │
    │       ▼
    ├── src/data/generator.py    ←── CTGAN training + 10-facility generation
    │       │
    │       ▼  data/synthetic/facility_NN_TYPE.csv
    │
    ├── src/data/fidelity.py     ←── KS-test, Frobenius norm, TSTR (vs MIMIC-IV)
    │
    ├── src/data/loaders.py      ←── Per-facility train/val/test splits
    │       │
    │       ▼
    ├── src/fl/client.py         ←── FedAcuityClient (Flower NumPyClient)
    │
    ├── src/fl/simulation.py     ←── Run all 5 strategies (CLI); CFL intra-cluster aggregation lives here too
    │
    ├── src/dp/epsilon_sweep.py  ←── Opacus DP + ε ∈ {1,2,5,10,∞}
    │
    ├── src/xai/shap_pipeline.py ←── SHAP values for all 5 models (Week 10)
    ├── src/xai/d1_fidelity.py   ←── (Week 10)
    ├── src/xai/d2_stability.py  ←── (Week 11)
    ├── src/xai/d3_fairness.py   ←── (Week 11)
    ├── src/xai/d4_plausibility.py ← (Week 12)
    ├── src/xai/scorecard.py     ←── Aggregates D1–D4 into radar chart
    │
    └── src/evaluation/logger.py ←── Centralised results logger (JSON + CSV)
```

---

## Five Model Variants

| Model | Training Data | Federation | Privacy |
|---|---|---|---|
| **Local** | Per-facility only | None | Implicit (no sharing) |
| **Centralised Oracle** | All facilities pooled | N/A | None (theoretical upper bound) |
| **FedAvg** | All facilities (federated) | Global single model | None |
| **FedProx** | All facilities (federated) | Global + proximal term μ | None |
| **Clustered FL** | Per-cluster (federated) | 3 cluster-specific models | None |

The DP layer is applied orthogonally to FedAvg and Clustered FL using the PyTorch NN variant.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Primary local model** | XGBoost | Better for tabular data; native SHAP support via TreeExplainer |
| **Secondary model** | PyTorch NN (2-3 layers) | Required for Opacus DP (doesn't support XGBoost) |
| **FL framework** | Flower (flwr) | CPU-native simulation; supports custom aggregation strategies |
| **Synthetic data** | CTGAN (SDV) | Conditional GAN; respects feature correlations and categorical boundaries |
| **Non-IID design** | Care-type-specific distributions | Mirrors real-world LTC heterogeneity (MC ≠ SNF ≠ IL) |
| **XGBoost federation** | Serialise booster bytes | No native FL support; weight exchange via `save_model()` / `load_model()` |
| **Cluster assignment** | Fixed by care type | Domain-driven — care type is the strongest source of distributional shift |
| **Held-out facilities** | IDs 8 and 9 (both IL) | Withheld from all FL training; used only for final generalisation evaluation |
| **Config management** | `config.yaml` single source | All hyperparams in one place; no hardcoded values in source files |
| **Reproducibility** | `SEED = 42` everywhere | Deterministic CTGAN, splits, model initialisation, perturbation tests |

---

## Non-IID Heterogeneity Design

The synthetic data is deliberately engineered to exhibit realistic distributional shift across care types — the primary motivation for Clustered FL.

| Feature | MC (Memory Care) | SNF (Skilled Nursing) | IL (Independent Living) |
|---|---|---|---|
| `adl_cognition` mean | 4.5 | 2.5 | 1.0 |
| `medication_count` mean | 11 | 9 | 5 |
| `nursing_hours_rn` mean | 2.5 | 3.0 | 1.0 |
| Mismatch rate | ~40% | ~28% | ~12% |
| RUG category mode | 6 (high) | 5 (moderate) | 2 (low) |

Standard FedAvg produces a single global model that must fit all three distributions simultaneously — leading to suboptimal performance for any one care type. Clustered FL avoids this by maintaining separate global models per cluster.

---

## File Structure Reference

```
fedacuity/
├── config.yaml                    ← Central hyperparameter config
├── requirements.txt               ← Pinned dependencies (Python 3.12)
├── README.md
│
├── src/
│   ├── config.py                  ← Config loader
│   ├── data/
│   │   ├── schema.py              ← Feature specs, non-IID dists, label def
│   │   ├── generator.py           ← CTGAN pipeline
│   │   ├── fidelity.py            ← KS-test, Frobenius, TSTR
│   │   └── loaders.py             ← Facility splits
│   ├── fl/
│   │   ├── client.py              ← Flower client (XGBoost)
│   │   └── simulation.py          ← All 5 strategy runner (CLI); CFL aggregation lives here too
│   ├── dp/
│   │   └── epsilon_sweep.py       ← Opacus DP sweep
│   ├── xai/
│   │   └── scorecard.py           ← XAI Audit Scorecard (radar chart)
│   └── evaluation/
│       └── logger.py              ← Results logger
│
├── data/
│   ├── synthetic/                 ← Generated CSVs (facility_NN_TYPE.csv)
│   ├── mimic_iv/                  ← MIMIC-IV elderly subset (pending access)
│   └── processed/
│
├── results/
│   ├── figures/                   ← Figures 1–6
│   ├── tables/                    ← CSV/JSON results tables
│   └── logs/                      ← Per-round FL metrics
│
├── notebooks/                     ← Jupyter experiment notebooks
├── paper/                         ← LaTeX paper (IEEE JBHI)
├── docs/                          ← Architecture + design docs
└── tests/                         ← Unit tests
```

---

## Communication Overhead Analysis

| Scenario | Data transmitted per round | vs Raw data upload |
|---|---|---|
| FedAvg (XGBoost weights) | ~50–200 KB × 10 clients | ~0.1% of raw data size |
| FedProx (same as FedAvg) | ~50–200 KB × 10 clients | ~0.1% |
| Clustered FL (per cluster) | ~50–200 KB × cluster size | ~0.1% |
| Hypothetical raw data upload | ~2–5 MB per facility per round | 100% baseline |

XGBoost model serialisation produces compact bytecode (~50–200 KB depending on tree count). Over 50 rounds with 10 clients, total communication is roughly 50–100 MB — negligible compared to transmitting raw patient records.

---

## Paper Figure Map

| Figure | Description | Generating Script | Phase |
|---|---|---|---|
| Fig 1 | System Architecture Diagram | This document (draw.io) | Week 2 |
| Fig 2 | MIMIC-IV Fidelity Distributions | `src/data/fidelity.py` | Week 5 |
| Fig 3 | FL Convergence Curves (AUC vs round) | `src/evaluation/figures.py` | Week 8 |
| Fig 4 | Five-Model Bar Chart | `src/evaluation/figures.py` | Week 8 |
| Fig 5 | Privacy-Utility Tradeoff (ε sweep) | `src/dp/epsilon_sweep.py` | Week 9 |
| Fig 6 | XAI Radar Chart | `src/xai/scorecard.py` | Week 12 |

---

*Last updated: Session 3 — Architecture document created*
