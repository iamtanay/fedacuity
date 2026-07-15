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
| Midsem evaluation | 13 Jun 2026 | FL pipeline running, C1 vs baselines results, C2 fidelity results | ✅ COMPLETE (7 Jun 2026) |
| Bug Fix Sprint | 7 Jun 2026 | All 6 critical bugs fixed, data regenerated, simulation rerun, /validate-research passes 17/17 | [x] COMPLETE |
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
BUG FIX SPRINT [14–16 Jun] ◄─── BLOCKING: XAI cannot start until bugs fixed
    │   Fix aggregation + FedProx + tree growth + mds_adl_summary
    │   + nursing_hours_lpn + seeding + eval_local + eval_held_out rounds
    │   + Regenerate data + Rerun simulation + /validate-research 15/15
    ▼
DP epsilon sweep (rerun after seed fix)
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
- [x] Run `pytest tests/test_schema.py -v` — 44/44 pass
- [x] Run `pytest tests/test_loaders.py -v` — 72/72 pass (fixed IL mismatch rate bug)
- [x] Run `python -m src.fl.simulation --strategy local` — AUC 0.9749
- [x] Run `python -m src.fl.simulation --strategy centralised` — AUC 0.9777
- [x] Run `python -m src.fl.simulation --strategy fedavg --rounds 10` — AUC 0.9643
- [x] Run `python -m src.fl.simulation --strategy fedprox --mu 0.1 --rounds 10` — AUC 0.9643
- [x] Run `python -m src.fl.simulation --strategy clustered --rounds 10` — AUC 0.9828 ✓ CFL > FedAvg
- [x] Confirm `results/logs/results_*.json` is produced for each strategy

### Day 3–4 (1–2 Jun) — Run Full Simulation + Fidelity
- [x] Run full simulation: `python -m src.fl.simulation --strategy all --rounds 50` — CFL 0.9826, FedAvg 0.9630
- [x] Run `python -m src.data.fidelity` (synthetic proxy — MIMIC-IV access pending)
- [x] Confirm `results/tables/fidelity_ks_test.csv` and `fidelity_frobenius.json` produced (14/14 KS pass, Frobenius 0.2436)
- [x] Confirm `results/figures/fig2_fidelity_distributions.png` produced

### Day 5–7 (3–6 Jun) → Completed 7 Jun
- [x] Create `src/evaluation/metrics.py` — AUC aggregation, Mann-Whitney U test (p=3.6e-5), bootstrap CI
- [x] Create `src/evaluation/figures.py` — Fig 3 (convergence curves) + Fig 4 (five-model bar chart)
- [x] Run figures script → `results/figures/fig3_convergence.png` and `fig4_model_comparison.png`
- [x] Run DP sweep: `python -m src.dp.epsilon_sweep` → `fig5_dp_privacy_utility.png` (ε=10 recommended, 15.7% degradation)

### Week 1 Exit Criteria
- [x] All 5 FL strategies have run 50 rounds and produced logged results
- [x] Fidelity validation has run (synthetic proxy) — 14/14 KS pass, TSTR gap 0.0199
- [x] Figs 2, 3, 4, 5 generated

---

## Week 2 — 7 Jun to 13 Jun 2026
**Goal: Polish results for midsem. Do not start new modules this week.**

### Day 1–3 (7–9 Jun) — Results Analysis & Paper Tables
- [x] Tabulate final AUC-ROC results for all 5 strategies (fill `paper/main.tex` Table II)
- [x] Run statistical test: Mann-Whitney U on CFL vs FedAvg AUC distributions (U=364, p=0.0039)
- [x] Compute bootstrap 95% CI on AUC for each strategy — saved to `fl_metrics_summary.json`
- [x] Fill paper Abstract with actual numbers (held-out CFL 0.9677 vs FedAvg 0.9057, 6.2pt gap)
- [x] Fill paper Sections: Experimental Setup, FL Results, DP Results, Discussion, Conclusion

### Day 4–5 (10–11 Jun) — Midsem Presentation Prep
- [x] Create midsem slide deck — generated `results/FedAcuity_Midsem_Slides.pptx` (13 slides)
  - Problem + motivation (HIPAA, LTC staffing crisis) — 2 slides
  - Three contributions overview — 1 slide
  - Data pipeline + fidelity results (Fig 2) — 2 slides
  - FL architecture diagram — 1 slide
  - Results: 5-model comparison (Fig 4) — 1 slide
  - Convergence curves (Fig 3) — 1 slide
  - DP privacy-utility tradeoff (Fig 5) — 1 slide
  - XAI plan (what comes after midsem) — 1 slide
  - Timeline to final sem — 1 slide
- [ ] Dry run presentation (≤ 15 min) — **do this yourself before 13 Jun**

### Day 6–7 (12–13 Jun) — Buffer + Midsem
- [ ] Buffer day: fix any last-minute issues
- [ ] **MIDSEM EVALUATION — 13 June 2026** ✅

### Midsem Must-Have Deliverables
- [x] Quantitative results: AUC-ROC for all 5 strategies — CFL 0.9677 vs FedAvg 0.9057 (held-out, 6.2pt gap, U=364, p=0.0039)
- [x] Fidelity metrics: KS 14/14, Frobenius 0.2196 vs baseline 5.9407 (27x), TSTR gap 0.0199
- [x] DP results: privacy-utility tradeoff — ε=10 recommended (15.7% degradation)
- [x] All figures: Figs 2–5 generated, in paper/figures/ and slide deck
- [x] Paper: Abstract + Sections I–VIII + Conclusion drafted with real numbers
- [x] Slide deck: 13-slide PPTX at results/FedAcuity_Midsem_Slides.pptx
- [ ] Dry run: 15-min practice before 13 Jun

---

---

## BUG FIX SPRINT — 13 Jun to 16 Jun 2026 (immediately post-MidSEM)
**Goal: Fix all scientific validity bugs found in the research audit before any XAI work begins.**
**Gate: Run `/validate-research` and achieve 15/15 PASS before proceeding to Week 3.**
**Invoke the skill with:** `/validate-research`

> Source of truth for this section: research audit conducted 7 Jun 2026.
> Each item is mapped to the `/validate-research` check number (C1–C15).

---

### CRITICAL BUGS — Must Fix (affect correctness of results)

#### BUG-01 [C1] — FedAvg Aggregation Is Not Weighted Averaging
**File:** `src/fl/simulation.py`, `src/fl/clustered_fl.py`
**Problem:** `_aggregate_xgb()` calls `max(client_results.values(), key=lambda x: x[1])[0]`. Since all 8 clients have exactly 657 training rows, `max()` returns whichever client Python iterates last. No actual federation or averaging occurs.
**Fix:** Replace with prediction ensemble averaging: train all clients, load each as an `XGBClassifier`, compute mean predicted probabilities across all clients on a shared reference set, and use the client whose predictions are closest to the ensemble mean as the representative model. OR implement proper serialized tree merging.
**Downstream:** Must re-run 50-round simulation after fix. All reported AUC numbers will change.
- [ ] Fix `_aggregate_xgb()` in `src/fl/simulation.py`
- [ ] Fix `_weighted_average_xgb()` in `src/fl/clustered_fl.py`
- [ ] Verify: multiple clients with different data now produce a different aggregate than single client

#### BUG-02 [C2] — FedProx Proximal Term Not Implemented for XGBoost
**File:** `src/fl/client.py:122-124`
**Problem:** `self.mu` is stored but never used in `fit()`. FedProx produces byte-for-bit identical results to FedAvg across all 50 rounds. Presenting it as a genuine comparison strategy is scientifically invalid.
**Fix (chosen approach — discuss with supervisor):**
- Option A: Implement FedProx via the PyTorch `StaffingNN` path (the proximal term is correctly implementable there) and report both XGBoost (CFL vs FedAvg) and NN (CFL vs FedProx) comparisons separately.
- Option B: Keep XGBoost as primary; explicitly document in paper that "FedProx with XGBoost reduces to FedAvg due to the absence of gradient-level access, and results are intentionally identical. The proximal term is correctly implemented in the PyTorch NN variant."
- [ ] Implement chosen option
- [ ] Update paper experimental setup section to disclose this clearly
- [ ] Update Table II footnote if results remain identical

#### BUG-03 [C3] — XGBoost Warm-Start Creates 2,600 Trees by Round 50
**File:** `src/fl/client.py:114`
**Problem:** `self.model.set_params(n_estimators=self.model.n_estimators + local_epochs * 10)` adds 50 trees per round. After 50 rounds: 2,600 trees. Local baseline: 100 trees. Comparison is deeply unfair.
**Fix:** Cap `n_estimators` at the configured value (100). After receiving a global model, do not add more trees — instead re-train with the same tree budget. Change the warm-start to retrain with fixed budget: `self.model.set_params(n_estimators=XGB_CFG["n_estimators"])` and retrain.
**Downstream:** Must re-run simulation after fix. Results will change.
- [ ] Fix `fit()` in `src/fl/client.py` to use fixed tree budget
- [ ] Verify: `model.n_estimators` stays at 100 throughout all rounds
- [ ] Re-run `pytest tests/` — confirm 72/72 still pass

#### BUG-04 [C4] — Torch Seed Not Reset Per Epsilon Run (Non-Monotonic DP Results)
**File:** `src/dp/epsilon_sweep.py:39`, `src/dp/epsilon_sweep.py:63`
**Problem:** `torch.manual_seed(SEED)` set once at module import. Each of 5 epsilon runs inherits different cumulative RNG state. Result: ε=10 AUC=0.829 < ε=5 AUC=0.873, violating basic DP theory (larger ε = less noise = should not decrease AUC).
**Fix:** Add `torch.manual_seed(SEED)` as the first line inside `train_with_epsilon()`. Also set `np.random.seed(SEED)` for reproducibility.
**Downstream:** Must re-run DP sweep. DP recommendation of ε=5 may be confirmed or may need to be revised. All DP results in paper must be updated.
- [ ] Add per-call seeding inside `train_with_epsilon()`
- [ ] Re-run `python -m src.dp.epsilon_sweep`
- [ ] Verify: AUC values are now monotonically non-decreasing as ε increases
- [ ] Update paper Table (DP results section) with new values
- [ ] Update Fig 5 with corrected numbers

#### BUG-05 [C8] — `eval_local()` Is an 8-Model Ensemble, Not a Local Baseline
**File:** `src/evaluation/eval_held_out.py:111-132`
**Problem:** `eval_local()` trains 8 models and averages their predictions on held-out facilities. This is a cross-facility ensemble, not a local baseline. Table II labels it "Local (no FL)" which is misleading — it has far more information than any single facility.
**Fix:** Rename to "Cross-Facility Ensemble (no aggregation)" in Table II and paper text. Also add a true local baseline: train a single XGBoost on all IL training data (facility 7 only, care-type matched) and report as "Care-Type Local Baseline."
- [ ] Add proper care-type-matched local baseline (facility 7 only → test on held-out 8,9)
- [ ] Rename ensemble baseline clearly in `eval_held_out.py` and all paper references
- [ ] Update Table II with corrected labels and new baseline row

#### BUG-06 [C7] — `eval_held_out.py` Uses 10 Rounds, Paper Describes 50 Rounds
**File:** `src/evaluation/eval_held_out.py:43`
**Problem:** `EVAL_ROUNDS = 10` hardcoded. The paper's experimental setup section says "50 communication rounds." Table II numbers come from a 10-round fresh re-simulation, inconsistent with the stated experimental design.
**Fix:** Change `EVAL_ROUNDS = 50`. Re-run `eval_held_out.py` after other bugs are fixed. Update Table II.
- [ ] Change `EVAL_ROUNDS = 50` in `eval_held_out.py`
- [ ] Re-run after BUG-01 through BUG-03 are fixed

---

### DATA & SCHEMA ISSUES — Must Fix (affect scientific validity of dataset)

#### DATA-01 [C5] — `mds_adl_summary` Has Negative Correlation with ADL Subscores
**File:** `src/data/generator.py`
**Problem:** CTGAN generates `mds_adl_summary` as an independent feature. In MDS 3.0, it IS definitionally the sum of 4 ADL subscores. Correlation with components is near-zero or negative — clinically impossible. An LTC clinician reviewer will immediately reject this.
**Fix:** After CTGAN generation in `generator.py`, overwrite `mds_adl_summary` as a deterministic function:
```python
df['mds_adl_summary'] = (df['adl_eating'] + df['adl_mobility'] +
                          df['adl_toileting'] + df['adl_cognition']) * (28.0 / 24.0)
df['mds_adl_summary'] = df['mds_adl_summary'].clip(0, 28)
```
**Downstream:** Must re-generate synthetic data. All downstream results will change.
- [ ] Add post-generation override in `generator.py`
- [ ] Re-run `python -m src.data.generator`
- [ ] Verify: Spearman correlation of each ADL subscore with `mds_adl_summary` >= 0.5
- [ ] Re-run all tests (72/72 must still pass)

#### DATA-02 [C6] — `nursing_hours_lpn` Not in `NON_IID_SPEC`; IL Values Clinically Implausible
**File:** `src/data/schema.py:NON_IID_SPEC`
**Problem:** `nursing_hours_lpn` (and `fall_risk_score`, `pain_assessment_score`, `resident_census`, `incident_count`) missing from `NON_IID_SPEC`. All three care types use the default N(6, 1.8), producing ~5.3 LPN hours/resident/day for IL — 7x the real-world CMS benchmark (~0.7). IL facilities would have MORE LPN hours than Memory Care in the synthetic data, which is backwards.
**Fix:** Add the following to `NON_IID_SPEC` for each care type:
```python
# In MC:
"nursing_hours_lpn": {"mean": 1.8, "std": 0.4, "clip": (0.5, 4.0)},
"fall_risk_score":   {"mean": 7.2, "std": 1.5, "clip": (0, 10)},
"pain_assessment_score": {"mean": 5.5, "std": 1.8, "clip": (0, 10)},
"resident_census":   {"mean": 60,  "std": 15,  "clip": (30, 100)},
"incident_count":    {"mean": 4,   "std": 2,   "clip": (0, 12)},
# In SNF:
"nursing_hours_lpn": {"mean": 2.2, "std": 0.5, "clip": (0.5, 5.0)},
"fall_risk_score":   {"mean": 6.0, "std": 1.8, "clip": (0, 10)},
"pain_assessment_score": {"mean": 5.0, "std": 2.0, "clip": (0, 10)},
"resident_census":   {"mean": 90,  "std": 20,  "clip": (50, 120)},
"incident_count":    {"mean": 3,   "std": 2,   "clip": (0, 10)},
# In IL:
"nursing_hours_lpn": {"mean": 0.4, "std": 0.15, "clip": (0.0, 1.5)},
"fall_risk_score":   {"mean": 2.5, "std": 1.2, "clip": (0, 7)},
"pain_assessment_score": {"mean": 2.0, "std": 1.2, "clip": (0, 6)},
"resident_census":   {"mean": 120, "std": 30,  "clip": (60, 200)},
"incident_count":    {"mean": 1,   "std": 1,   "clip": (0, 5)},
```
**Downstream:** Must re-generate synthetic data after this and DATA-01.
- [ ] Add all 5 features to `NON_IID_SPEC` for all 3 care types in `schema.py`
- [ ] Also check `rug_category` and `medication_count` are clinically appropriate per care type
- [ ] Re-run `python -m src.data.generator`
- [ ] Verify: IL nursing_hours_lpn mean is ~0.4, MC is ~1.8, SNF is ~2.2

---

### SCIENTIFIC DISCLOSURE — Must Disclose in Paper

#### DISCLOSE-01 [C9] — IL Cluster Has Only One Training Client
**Location:** `paper/main.tex` Section V (Experimental Setup)
**Problem:** CFL's "3 independent global cluster models" claim: only IL cluster has 1 training client (facility 7). No federation occurs in IL cluster. The CFL IL result is a single-facility model evaluated on 2 unseen IL facilities.
- [ ] Add sentence to paper: "The IL cluster contains a single training facility (facility 7), as facilities 8 and 9 are held-out. IL cluster performance therefore reflects zero-shot transfer from one IL facility to two unseen IL facilities, rather than federated averaging."
- [ ] Note this as a limitation in Section VIII Discussion

#### DISCLOSE-02 [C3 gap] — 13.2pt CFL vs FedAvg Gap Is Structurally Predictable
**Location:** `paper/main.tex` Section VIII (Discussion)
**Problem:** Both held-out facilities are IL type. FedAvg is trained on 7/8 MC+SNF clients and struggles on IL — this is predictable from the experimental design, not a surprising result.
- [ ] Add to Discussion: "We acknowledge that the evaluation design (holding out IL facilities) is favorable to CFL. Future work should evaluate CFL on held-out facilities from each care type to generalize the finding."

#### DISCLOSE-03 [C14] — Fig 4 Shows Training-Client AUC, Not Held-Out AUC
**Location:** `src/evaluation/figures.py`, `paper/main.tex`
**Problem:** Fig 4 bar chart shows FedAvg=0.9630 (training clients) while Table II shows FedAvg=0.8474 (held-out). 11.6pt discrepancy unexplained.
- [ ] Fix: Regenerate Fig 4 using `fl_held_out_metrics.json` as data source
- [ ] Update caption in paper to clarify data source

---

### CONFIGURATION & DOCUMENTATION ERRORS — Quick Fixes

#### CONFIG-01 [C10] — `config.yaml` Comment Is Wrong
**File:** `config.yaml:60`
**Problem:** `held_out_facilities: [8, 9]  # 1 SNF + 1 MC reserved` — facilities 8 and 9 are both IL.
- [ ] Fix comment: `held_out_facilities: [8, 9]  # 2 IL facilities — held out for final evaluation`

#### CONFIG-02 [C11] — Feature Count Mismatch (Paper: 15, Code: 14)
**File:** `paper/main.tex`, `src/data/schema.py`
**Problem:** Paper says "15-feature schema." `FEATURE_SPECS` has 14 entries. `care_type` is listed in Table I but is not in `FEATURE_NAMES` and is excluded from training.
- [ ] Option A: Update paper to say "14-feature schema" and note care_type is used for clustering only.
- [ ] Option B: Add `care_type` as an encoded training feature and re-run everything.
- [ ] Update Table I caption accordingly.

#### CONFIG-03 [C12] — `dataset_metadata.json` Is Stale (Shows IL Mismatch ~42%)
**File:** `data/synthetic/dataset_metadata.json`
**Problem:** Metadata written before the IL mismatch rate bug was fixed. Shows ~42% for IL instead of ~12%.
- [ ] Will be fixed automatically when data is regenerated in DATA-01/DATA-02.
- [ ] After regeneration, verify: IL metadata shows ~12% mismatch rate.

#### CONFIG-04 [C15] — Fig 4 Y-Axis Truncated (Starts at 0.93)
**File:** `src/evaluation/figures.py:126`
**Problem:** Y-axis `bottom=max(0.92, min(aucs) - 0.02)` visually exaggerates small differences. AUC difference of ~2% appears as ~30% of chart height.
- [ ] After fixing Fig 4 data source (DISCLOSE-03), also fix Y-axis to start at 0.5 or add a broken-axis indicator. For IEEE JBHI, at minimum add a note in the caption: "Note: Y-axis truncated for readability; see Table II for absolute values."

---

### BUG FIX SPRINT — Execution Order

**IMPORTANT: Steps must be done in this exact order due to data dependencies.**

```
Step 1: Fix schema.py (DATA-02: NON_IID_SPEC additions)
Step 2: Fix generator.py (DATA-01: mds_adl_summary computed)
Step 3: Fix client.py (BUG-03: tree count bounded; BUG-02: FedProx disclosure)
Step 4: Fix simulation.py + clustered_fl.py (BUG-01: real aggregation)
Step 5: Regenerate synthetic data: python -m src.data.generator
Step 6: Fix config.yaml comment (CONFIG-01) — quick, do any time
Step 7: Fix epsilon_sweep.py (BUG-04: per-call seeding)
Step 8: Run pytest — must be 72/72
Step 9: Run full 50-round simulation: python -m src.fl.simulation --strategy all --rounds 50
Step 10: Fix eval_held_out.py (BUG-05, BUG-06: rounds=50, local baseline)
Step 11: Run eval_held_out.py to get corrected Table II numbers
Step 12: Re-run DP sweep: python -m src.dp.epsilon_sweep
Step 13: Fix figures.py (DISCLOSE-03: Fig 4 uses held-out AUC; CONFIG-04: Y-axis)
Step 14: Regenerate all figures: python -m src.evaluation.figures
Step 15: Fix paper disclosures (DISCLOSE-01, DISCLOSE-02, CONFIG-02)
Step 16: Run /validate-research — must achieve 15/15 PASS
Step 17: Regenerate midsem report and slides with corrected numbers
Step 18: Commit all changes
```

### Bug Fix Sprint Exit Criteria
- [x] `/validate-research` produces 15/15 PASS
- [x] 72/72 tests still pass
- [x] All FL result numbers updated in paper and slides
- [x] No `mds_adl_summary` negative correlation with subscores
- [x] DP results are monotonically non-decreasing with ε
- [x] Fig 4 shows held-out AUC, not training-client AUC
- [x] dataset_metadata.json shows IL mismatch ~12%

---

## Week 3 — 14 Jun to 20 Jun 2026
**Goal: Build the SHAP pipeline and D1 + D2 XAI dimensions.**
**PREREQUISITE: Bug Fix Sprint must be complete (/validate-research 15/15) before starting Week 3.**

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

- [x] MIMIC-IV PhysioNet access: **DONE** (17 Jun 2026). `mimic_preprocessor.py` and `mimic_analysis.py` committed. `fidelity.py` now uses real MIMIC-IV. Fig 2 two-panel: KS distributions + within-MIMIC-IV cohort calibration.
- [ ] `notebooks/03_mimic_exploration.ipynb` — only if MIMIC-IV access granted
- [ ] FedProx μ sweep (test μ ∈ {0.01, 0.1, 1.0} from `config.yaml`) on PyTorch NN — implement properly and report best μ
- [ ] `conftest.py` — set up shared pytest fixtures to reduce test boilerplate
- [ ] Add test coverage for `src/fl/simulation.py` and `src/evaluation/` modules (currently 0% covered by tests)
- [ ] Statistical test for SHAP stability: Friedman test across ε perturbation runs for D2 module

Note: "True XGBoost tree merging" has been PROMOTED to BUG-01 in the Bug Fix Sprint — it is now a required fix, not a backlog item.

---

## Session Log

Use this table to track what was done each working session.

| Date | Session | Done | Blocked on |
|---|---|---|---|
| 30 May 2026 | Setup | Created CLAUDE.md, removed CONTEXT.md, created PLAN.md | — |
| 30 May 2026 | Day 1–2 | All tests green (72/72). Fixed 3 bugs: IL NON_IID_SPEC missing adl_eating/adl_toileting (41%→12% mismatch), _calibrate_threshold hi ceiling too low (5→50), XGBoost 3.x BytesIO incompatibility. Rewrote simulation.py with manual FL loop (no Ray). All 5 strategies run: Local 0.9749, Centralised 0.9777, FedAvg 0.9643, FedProx 0.9643, CFL 0.9828. CFL > FedAvg confirmed. | — |
| 7 Jun 2026 | Week 2 Day 1 | Full 50-round simulation: CFL 0.9826 on training clients. Held-out eval: CFL 0.9790, FedAvg 0.8474 (13.2pt gap). Fidelity: 14/14 KS pass, TSTR gap 0.0183. DP sweep: ε=5 recommended (12.3% degradation; ε=2 causes 22.7% drop). Created metrics.py, figures.py, eval_held_out.py. Generated Figs 2–5. Filled paper Abstract, Table II, Sections V–IX with real numbers. Added 2 BibTeX entries. Generated 13-slide PPTX deck. config.yaml: recommended_epsilon updated to 5. | — |
| 7 Jun 2026 | Research Audit | Deep research audit conducted. Found 6 critical bugs, 8 scientific validity issues, 4 config errors. Key findings: (1) aggregation is not FedAvg — picks one client; (2) FedProx is identical to FedAvg; (3) XGBoost grows to 2,600 trees by round 50; (4) torch seed causes non-monotonic DP; (5) mds_adl_summary has negative correlation with subscores; (6) nursing_hours_lpn not in NON_IID_SPEC. Full registry in Bug Fix Sprint section. Created /validate-research skill. Created midsem_slides_v2.py and midsem_report.py. | Bug Fix Sprint must complete before Week 3 XAI work |
| 7 Jun 2026 | Bug Fix Sprint | ALL 17 CHECKS PASS. Fixed: (1) prediction-consensus FedAvg aggregation in simulation.py + clustered_fl.py; (2) FedProx documented as equivalent to FedAvg for XGBoost; (3) fixed tree budget (100 trees always, no warm-start growth); (4) per-call torch seeding in epsilon_sweep.py — DP results now monotonic; (5) mds_adl_summary computed from ADL subscores (correlations 0.58-0.87); (6) 5 features added to NON_IID_SPEC (nursing_hours_lpn IL=0.39 vs MC=1.71 vs SNF=2.27); (7) eval_held_out.py: EVAL_ROUNDS=50, proper IL local baseline (facility 7), cross-facility ensemble labelled correctly; (8) sentinel check fixed (len <= 1). Data regenerated. Simulation rerun: FedAvg 0.9853, CFL 0.9906 (training clients). Held-out: CFL=0.9677 = IL Local (IL cluster has 1 client), FedAvg=0.9057 = ensemble. DP: eps=10 recommended (15.7% drop), monotonic results. Paper, report, and slides regenerated. | — |
| 17 Jun 2026 | Session 7 — Research hardening sprint | (1) HELD_OUT_FACILITIES extended to [6,9] — SNF (fac 6) + IL (fac 9). (2) Facility 8 moved to IL training cluster (2 IL training clients). (3) Deleted dead clustered_fl.py; production aggregation confirmed in simulation.py. (4) MIMIC-IV access: committed mimic_preprocessor.py + mimic_analysis.py; Fig 2 two-panel (KS + cohort calibration). (5) Full pipeline rerun — new held-out canonical numbers: CFL 0.9827 vs FedAvg 0.9685 (+1.42pt; SNF: +0.36pt, IL: +2.65pt), U=400 p<0.001. (6) Figs 3–5 regenerated. (7) paper/main.tex updated with all new numbers. (8) CLAUDE.md + PLAN.md synced. (9) /validate-research passes. (10) Pushed to origin/main. | — |
| 15 Jul 2026 | Session 8 — C3 build + ironclad hardening | **Entire C3 XAI Audit Scorecard built on real SHAP** (Weeks 3–6 delivered). (1) shap_pipeline.py reconstructs each strategy's deployed model + computes SHAP per care type. (2) d1_fidelity/d2_stability/d3_fairness/d4_plausibility modules + run_xai_audit.py orchestrator + real xai_audit_raw.json → Fig 6 radar + scorecard CSV/TeX. (3) **D3 fairness is the headline:** single global FedAvg model collapses to SNF-like, MC TPR 0.22, IL AUC 0.67, EO gap 0.39; CFL balanced (subgroup AUC ≥0.96), EO gap 0.18. Honest trade-off (CFL lower on D2 stability). (4) **DP bug fixed:** epsilon_sweep now averages 5 PAIRED seeds → monotonic; canonical no-DP 0.9812 → ε=10 0.8347 (14.9%). (5) paper/main.tex fully hardened (Section VI/VII real numbers, stale held-out/DP/conclusion/fidelity fixed) — compiles clean, 7 pages. (6) final_slides.py (15-slide PPTX) + final_report.py (DOCX) deliverables. (7) /validate-research 15/15, 73/73 tests. | — |

---

## How Claude Should Use This File

- **At the start of every session**: read this file, note the current week and unchecked tasks
- **During a session**: check off tasks as they are completed
- **If a task is blocked**: note the blocker in the Session Log, skip to next unblocked task
- **Do not reorder weeks** unless a dependency forces it — the institution sees this plan
- **Update the Session Log** at the end of every session
