"""
FedAcuity -- M4.2 D1 Explanation Fidelity (Contribution C3)

D1 asks: does a federated model *reason* like the Centralised Oracle?

For each strategy we take the global SHAP importance vector (mean|SHAP| per
feature, from shap_pipeline) and compute the Spearman rank correlation of its
feature-importance ranking against the Centralised Oracle's ranking. High rho
means the federated model attributes importance to the same clinical features
in the same order as the data-pooling upper bound -- i.e. federation preserved
the model's reasoning, not just its accuracy.

  score (D1) = max(0, spearman_rho)          # oracle scores 1.0 by definition
  top-k overlap reported alongside for transparency (config: xai.shap.top_k).

Output: results/tables/d1_fidelity.json

Usage:
    python -m src.xai.d1_fidelity
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

from src.config import cfg
from src.xai.shap_pipeline import load_shap_results, STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = Path(cfg["paths"]["results"]["tables"])
TOP_K = cfg["xai"]["shap"]["top_k_features"]
TARGET_RHO = cfg["xai"]["d1_fidelity"]["target_rho"]
REFERENCE = "centralised"


def compute_d1(shap_results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, dict]:
    """Return {strategy: {rho, score, topk_overlap, meets_target}}."""
    ref_imp = shap_results[REFERENCE]["mean_abs"]
    features = shap_results[REFERENCE]["features"]
    ref_rank_order = set(np.array(features)[np.argsort(ref_imp)[::-1][:TOP_K]])

    out = {}
    for strat in STRATEGIES:
        imp = shap_results[strat]["mean_abs"]
        if strat == REFERENCE:
            rho = 1.0
        else:
            rho = float(spearmanr(imp, ref_imp).correlation)
        strat_topk = set(np.array(features)[np.argsort(imp)[::-1][:TOP_K]])
        overlap = len(ref_rank_order & strat_topk) / TOP_K
        out[strat] = {
            "rho": round(rho, 4),
            "score": round(max(0.0, rho), 4),
            "topk_overlap": round(overlap, 4),
            "meets_target": bool(rho >= TARGET_RHO),
        }
    return out


def main():
    shap_results = load_shap_results()
    d1 = compute_d1(shap_results)
    out_path = TABLES_DIR / "d1_fidelity.json"
    with open(out_path, "w") as f:
        json.dump(d1, f, indent=2)
    logger.info(f"Saved {out_path}")

    print("\n-- D1 Explanation Fidelity (Spearman rho vs Centralised Oracle) --")
    print(f"  {'Strategy':<16} {'rho':>7} {'score':>7} {'top10 overlap':>14} {'>=%.2f?':>8}"
          % TARGET_RHO)
    for strat, v in d1.items():
        print(f"  {strat:<16} {v['rho']:>7.3f} {v['score']:>7.3f} "
              f"{v['topk_overlap']:>13.0%} {'YES' if v['meets_target'] else 'no':>8}")
    return d1


if __name__ == "__main__":
    main()
