# FedAcuity — Execution Plan

**Author:** Tanay Kashyap
**Plan created:** 30 May 2026
**Midsem evaluation:** 13 June 2026 (2 weeks)
**Final sem evaluation:** 11 July 2026 (4 weeks after midsem)

Check off tasks as they are completed. Update status at the top of each session.
Claude reads this file at the start of every session — keep it current.

---

## Milestone Summary

| Milestone | Date | Must-Have | Status |
|---|---|---|---|
| Midsem evaluation | 13 Jun 2026 | FL pipeline running, C1 vs baselines results, C2 fidelity results | ⬜ |
| Final sem evaluation | 11 Jul 2026 | C3 XAI scorecard, all figures, paper draft complete | ⬜ |

---

## Critical Path

```
synthetic data (done)
    │
    ▼
run FL simulation ──► evaluation figures ──► midsem [13 Jun]
    │
    ▼
DP epsilon sweep
    │
    ▼
SHAP pipeline ──► D1–D4 XAI modules ──► XAI scorecard ──► final sem [11 Jul]
    │
    ▼
paper: fill results sections ──► full draft ──► final sem [11 Jul]
```

---

## Week 1 — 30 May to 6 Jun 2026
**Goal: Get the FL pipeline running end-to-end and producing numbers.**

### Day 1–2 (30–31 May) — Debug & Run FL Simulation
- [ ] Run `pytest tests/test_schema.py -v` — confirm all 35 pass
- [ ] Run `pytest tests/test_loaders.py -v` — confirm non-data tests pass
- [ ] Run `python -m src.fl.simulation --strategy local` — fix any runtime errors
- [ ] Run `python -m src.fl.simulation --strategy centralised` — fix any runtime errors
- [ ] Run `python -m src.fl.simulation --strategy fedavg --rounds 10` (short run first)
- [ ] Run `python -m src.fl.simulation --strategy fedprox --mu 0.1 --rounds 10`
- [ ] Run `python -m src.fl.simulation --strategy clustered --rounds 10`
- [ ] Confirm `results/logs/results_*.json` is produced for each strategy

### Day 3–4 (1–2 Jun) — Run Full Simulation + Fidelity
- [ ] Run full simulation: `python -m src.fl.simulation --all --rounds 50`
- [ ] Run `python -m src.data.fidelity` (will use synthetic proxy — MIMIC-IV not available yet)
- [ ] Confirm `results/tables/fidelity_ks_test.csv` and `fidelity_frobenius.json` produced
- [ ] Confirm `results/figures/fig2_fidelity_distributions.png` produced

### Day 5–7 (3–6 Jun) — Build evaluation/metrics.py and evaluation/figures.py
- [ ] Create `src/evaluation/metrics.py` — AUC aggregation, Mann-Whitney U test, bootstrap CI
- [ ] Create `src/evaluation/figures.py` — Fig 3 (convergence curves) + Fig 4 (five-model bar chart)
- [ ] Run figures script → `results/figures/fig3_convergence.png` and `fig4_model_comparison.png`
- [ ] Run DP sweep: `python -m src.dp.epsilon_sweep` → `fig5_dp_privacy_utility.png`

### Week 1 Exit Criteria
- All 5 FL strategies have run 50 rounds and produced logged results
- Fidelity validation has run (synthetic proxy is fine)
- Figs 2, 3, 4, 5 generated (even if not final quality)

---

## Week 2 — 7 Jun to 13 Jun 2026
**Goal: Polish results for midsem. Do not start new modules this week.**

### Day 1–3 (7–9 Jun) — Results Analysis & Paper Tables
- [ ] Tabulate final AUC-ROC results for all 5 strategies (fill `paper/main.tex` Table II)
- [ ] Run statistical test: Mann-Whitney U on CFL vs FedAvg AUC distributions
- [ ] Compute bootstrap 95% CI on AUC for each strategy
- [ ] Fill paper Abstract with actual numbers (2–3 key results)
- [ ] Fill paper Section V (Experimental Setup) and Section VI (Results)

### Day 4–5 (10–11 Jun) — Midsem Presentation Prep
- [ ] Create midsem slide deck (suggest: 12–15 slides)
  - Problem + motivation (HIPAA, LTC staffing crisis) — 2 slides
  - Three contributions overview — 1 slide
  - Data pipeline + fidelity results (Fig 2) — 2 slides
  - FL architecture diagram (Fig 1 from architecture.md) — 1 slide
  - Results: 5-model comparison (Fig 4) — 1 slide
  - Convergence curves (Fig 3) — 1 slide
  - DP privacy-utility tradeoff (Fig 5) — 1 slide
  - XAI plan (what comes after midsem) — 1 slide
  - Timeline to final sem — 1 slide
- [ ] Dry run presentation (≤ 15 min)

### Day 6–7 (12–13 Jun) — Buffer + Midsem
- [ ] Buffer day: fix any last-minute issues
- [ ] **MIDSEM EVALUATION — 13 June 2026** ✅

### Midsem Must-Have Deliverables
- Quantitative results: AUC-ROC for all 5 strategies, with CFL > FedAvg demonstrated
- Fidelity metrics: KS-test pass rate, Frobenius norm vs baseline, TSTR gap
- DP results: privacy-utility tradeoff curve (Fig 5)
- All figures: Figs 2–5 generated and in slide deck
- Paper: Abstract + Sections I–VI drafted (results can be preliminary)

---

## Week 3 — 14 Jun to 20 Jun 2026
**Goal: Build the SHAP pipeline and D1 + D2 XAI dimensions.**

### Day 1–3 (14–16 Jun) — SHAP Pipeline
- [ ] Create `src/xai/shap_pipeline.py`
  - Load trained models from each of the 5 strategies (from `results/logs/`)
  - Compute SHAP values via `shap.TreeExplainer` for XGBoost models
  - Save SHAP arrays per model to `results/tables/shap_values_<strategy>.npy`
  - Config: `background_samples=100`, `test_samples=500` (from `config.yaml → xai.shap`)
- [ ] Run `python -m src.xai.shap_pipeline` — confirm SHAP files produced for all 5 models

### Day 4–5 (17–18 Jun) — D1 Fidelity Module
- [ ] Create `src/xai/d1_fidelity.py`
  - Compute Spearman ρ of top-10 SHAP feature ranks vs centralised oracle model
  - Target: ρ ≥ 0.75 for CFL (from `config.yaml → xai.d1_fidelity.target_rho`)
  - Output: `{ "local": ρ, "fedavg": ρ, "fedprox": ρ, "clustered_fl": ρ }` → `results/tables/d1_fidelity.json`
- [ ] Run and verify output

### Day 6–7 (19–20 Jun) — D2 Stability Module
- [ ] Create `src/xai/d2_stability.py`
  - 100 perturbation runs with ±5% Gaussian noise on continuous features
  - Measure mean absolute SHAP shift per model
  - Output: `{ model: mean_shap_shift }` → `results/tables/d2_stability.json`
  - CFL should show lower SHAP shift than global FedAvg (hypothesis)
- [ ] Run and verify output

### Week 3 Exit Criteria
- SHAP values computed for all 5 models
- D1 and D2 scores in `results/tables/`

---

## Week 4 — 21 Jun to 27 Jun 2026
**Goal: Build D3 + D4, assemble full XAI scorecard, update paper.**

### Day 1–3 (21–23 Jun) — D3 Fairness + D4 Plausibility
- [ ] Create `src/xai/d3_fairness.py`
  - Compute equalized odds and demographic parity across MC/SNF/IL subgroups
  - Normalise to [0, 1] score (higher = more fair)
  - Output: `results/tables/d3_fairness.json`
- [ ] Create `src/xai/d4_plausibility.py`
  - Check what % of each model's top-5 SHAP features are in the literature list
  - Literature features: `adl_mobility`, `adl_cognition`, `medication_count`, `fall_risk_score`, `pain_assessment_score`, `mds_adl_summary`, `rug_category`
  - Target: ≥ 60% match rate (from `config.yaml → xai.d4_plausibility.target_match_rate`)
  - Output: `results/tables/d4_plausibility.json`
- [ ] Run both modules

### Day 4–5 (24–25 Jun) — Assemble XAI Scorecard
- [ ] Consolidate D1–D4 outputs into `results/tables/xai_audit_raw.json` (format expected by `scorecard.py`)
- [ ] Run `python -m src.xai.scorecard` → Fig 6 radar chart + `xai_audit_scorecard.csv`
- [ ] Verify CFL covers more area than FedAvg on D2 Stability and D3 Fairness

### Day 6–7 (26–27 Jun) — Paper: XAI Section
- [ ] Write Section VII (XAI Audit) in `paper/main.tex` with real D1–D4 scores
- [ ] Add `xai_audit_scorecard.csv` numbers to Table II
- [ ] Add Fig 6 (radar chart) to paper

### Week 4 Exit Criteria
- All 4 XAI dimensions computed with real data
- Fig 6 generated
- Paper Sections I–VII drafted

---

## Week 5 — 28 Jun to 4 Jul 2026
**Goal: Complete paper draft. All sections filled. All 6 figures final quality.**

### Day 1–3 (28–30 Jun) — Paper Writing Sprint
- [ ] Section VIII — Discussion: interpret CFL vs FedAvg gap, DP trade-off, XAI findings
- [ ] Section IX — Conclusion: restate 3 contributions + limitations + future work
- [ ] Add 2 missing BibTeX entries (`bates2021ml_ltc`, `dellefield2015staffing`)
- [ ] Cross-reference: every claim in the paper has a figure or table number
- [ ] Write paper captions for all 6 figures

### Day 4–5 (1–2 Jul) — Figure Polish Pass
- [ ] Fig 1 (System Architecture) — create from `docs/architecture.md` using draw.io, export as PDF
- [ ] Figs 2–6 — final quality check: labels readable, legend correct, DPI 300, PDF exported
- [ ] Copy all 6 figures to `paper/figures/` for Overleaf

### Day 6–7 (3–4 Jul) — Overleaf Compile + Stats
- [ ] Upload `paper/main.tex` + `references.bib` + all figures to Overleaf
- [ ] Compile and fix any LaTeX errors
- [ ] Verify statistical tests are reported correctly (Mann-Whitney U p-values, bootstrap CIs)

### Week 5 Exit Criteria
- Full paper draft compiled without errors on Overleaf
- All 6 figures final and in paper
- All `% TBD` placeholders replaced with real values

---

## Week 6 — 5 Jul to 11 Jul 2026
**Goal: Polish, rehearse, submit.**

### Day 1–2 (5–6 Jul) — Final Paper Review
- [ ] Read full paper end-to-end — fix logic gaps, inconsistent numbers, tense issues
- [ ] Check all figure/table cross-references are correct
- [ ] Confirm IEEE JBHI formatting: page limit, author block, column layout

### Day 3–4 (7–8 Jul) — Final Presentation Prep
- [ ] Update slide deck for final sem (extend midsem deck with Weeks 3–6 results)
  - XAI Scorecard radar chart — 2 slides
  - D1–D4 individual results — 1 slide
  - Contributions validated — 1 slide
  - Limitations + future work — 1 slide
- [ ] Add: demo script (run `python -m src.xai.scorecard` live if possible)
- [ ] Dry run (≤ 20 min)

### Day 5–6 (9–10 Jul) — Buffer
- [ ] Fix any issues from dry run
- [ ] Ensure all code runs clean from scratch in a fresh venv
- [ ] Final git commit + push of paper and code

### Day 7 (11 Jul) — **FINAL SEM EVALUATION** ✅

### Final Sem Must-Have Deliverables
- All 5 FL strategies benchmarked with AUC-ROC, F1, statistical significance
- C2: Fidelity validation with KS-test, Frobenius norm, TSTR gap < 8%
- C3: Full XAI Audit Scorecard (D1–D4) with radar chart (Fig 6)
- DP privacy-utility tradeoff curve (Fig 5) with ε recommendation justified
- Complete paper draft ready for submission
- Clean, runnable codebase

---

## Backlog (If Time Allows)

These are stretch goals — only attempt if ahead of schedule:

- [ ] Apply for MIMIC-IV PhysioNet access (takes ~2 weeks) — start application immediately so access may arrive during Week 3–4. Re-run fidelity validation with real MIMIC-IV data if access granted.
- [ ] True XGBoost tree merging in `clustered_fl.py:_weighted_average_xgb()` — horizontal FL with tree concatenation (current placeholder returns largest client model)
- [ ] `notebooks/03_mimic_exploration.ipynb` — only if MIMIC-IV access granted
- [ ] FedProx μ sweep (test μ ∈ {0.01, 0.1, 1.0} from `config.yaml`) and report best μ
- [ ] `conftest.py` — set up shared pytest fixtures to reduce test boilerplate

---

## Session Log

Use this table to track what was done each working session.

| Date | Session | Done | Blocked on |
|---|---|---|---|
| 30 May 2026 | Setup | Created CLAUDE.md, removed CONTEXT.md, created PLAN.md | — |

---

## How Claude Should Use This File

- **At the start of every session**: read this file, note the current week and unchecked tasks
- **During a session**: check off tasks as they are completed
- **If a task is blocked**: note the blocker in the Session Log, skip to next unblocked task
- **Do not reorder weeks** unless a dependency forces it — the institution sees this plan
- **Update the Session Log** at the end of every session
