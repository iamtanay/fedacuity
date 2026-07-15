"""
FedAcuity -- M4.6 XAI Audit Orchestrator (Contribution C3)

End-to-end driver for the XAI Audit Scorecard. Reconstructs each strategy's
deployed model ONCE (the FL simulations are the expensive step), then:

  1. computes + saves SHAP values for all 5 strategies  (shap_pipeline)
  2. D1 Explanation Fidelity      -> results/tables/d1_fidelity.json
  3. D2 Explanation Stability     -> results/tables/d2_stability.json
  4. D3 Outcome Fairness          -> results/tables/d3_fairness.json
  5. D4 Clinical Plausibility     -> results/tables/d4_plausibility.json
  6. assembles the normalised scorecard input -> results/tables/xai_audit_raw.json
  7. runs scorecard.py            -> Fig 6 radar + xai_audit_scorecard.{csv,tex}

Every score is derived from real SHAP computations -- no placeholders.

Usage:
    python -m src.xai.run_xai_audit
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.config import cfg
from src.data.loaders import load_facility
from src.xai import shap_pipeline as sp
from src.xai.d1_fidelity import compute_d1
from src.xai.d2_stability import compute_d2
from src.xai.d3_fairness import compute_d3
from src.xai.d4_plausibility import compute_d4
from src.xai.scorecard import run_scorecard, DIMENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = Path(cfg["paths"]["results"]["tables"])


def main():
    logger.info("=== XAI AUDIT (C3) — full pipeline ===")

    # ── Shared setup: reconstruct deployed models ONCE ─────────────────────────
    test_sets = sp.get_care_type_test_sets()
    features = sp._features_for(load_facility(sp.TRAINING_FIDS[0]))
    models = sp.reconstruct_strategy_models()

    # ── 1. SHAP values (compute + persist) ─────────────────────────────────────
    shap_results = {}
    for strat in sp.STRATEGIES:
        out = sp.compute_shap_for_strategy(models[strat], test_sets, features)
        shap_results[strat] = out
        np.savez_compressed(
            TABLES_DIR / f"shap_values_{strat}.npz",
            shap=out["shap"], X=out["X"], care_type=out["care_type"],
            y=out["y"], mean_abs=out["mean_abs"], features=out["features"],
        )
    logger.info("SHAP values computed for all 5 strategies")

    # ── 2-5. Dimensions ────────────────────────────────────────────────────────
    d1 = compute_d1(shap_results)
    d2 = compute_d2(models, test_sets, features)
    d3 = compute_d3(models, test_sets, features)
    d4 = compute_d4(shap_results)

    for name, obj in [("d1_fidelity", d1), ("d2_stability", d2),
                      ("d3_fairness", d3), ("d4_plausibility", d4)]:
        with open(TABLES_DIR / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=2)
        logger.info(f"Saved {name}.json")

    # ── 6. Assemble normalised scorecard input ────────────────────────────────
    raw_scores = {}
    for strat in sp.STRATEGIES:
        raw_scores[strat] = {
            "D1 Fidelity":     d1[strat]["score"],
            "D2 Stability":    d2[strat]["score"],
            "D3 Fairness":     d3[strat]["score"],
            "D4 Plausibility": d4[strat]["score"],
        }
    raw_path = TABLES_DIR / "xai_audit_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_scores, f, indent=2)
    logger.info(f"Saved {raw_path}")

    # ── 7. Scorecard + Fig 6 ───────────────────────────────────────────────────
    run_scorecard()
    logger.info("=== XAI AUDIT complete — Fig 6 + scorecard written ===")


if __name__ == "__main__":
    main()
