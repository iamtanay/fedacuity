"""
FedAcuity -- M4.5 D4 Clinical Plausibility (Contribution C3)

D4 asks: are a model's most influential features clinically established
determinants of LTC staffing-acuity mismatch, or spurious signal?

For each strategy we take the top-k features by mean|SHAP| and compute the
fraction that fall within the evidence-based predictor set defined in
config (xai.d4_plausibility.literature_features) -- see that block for the
two-stream clinical rationale (acuity/demand + staffing-supply/scale).

  score (D4) = |top_k features INTERSECT literature_features| / top_k

This is a "no spurious features" sanity check: a faithful model's top features
should all be clinically recognized. High scores across every strategy are the
expected, desirable outcome; the CFL advantage is concentrated in D1-D3.

Output: results/tables/d4_plausibility.json

Usage:
    python -m src.xai.d4_plausibility
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from src.config import cfg
from src.xai.shap_pipeline import load_shap_results, STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = Path(cfg["paths"]["results"]["tables"])
TOP_K = cfg["xai"]["d4_plausibility"]["top_k"]
LIT_FEATURES = set(cfg["xai"]["d4_plausibility"]["literature_features"])
TARGET = cfg["xai"]["d4_plausibility"]["target_match_rate"]


def compute_d4(shap_results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, dict]:
    """Return {strategy: {top_features, matched, score, meets_target}}."""
    out = {}
    for strat in STRATEGIES:
        imp = shap_results[strat]["mean_abs"]
        features = np.array(shap_results[strat]["features"])
        top = list(features[np.argsort(imp)[::-1][:TOP_K]])
        matched = [f for f in top if f in LIT_FEATURES]
        score = len(matched) / TOP_K
        out[strat] = {
            "top_features": top,
            "matched": matched,
            "score": round(score, 4),
            "meets_target": bool(score >= TARGET),
        }
    return out


def main():
    shap_results = load_shap_results()
    d4 = compute_d4(shap_results)
    out_path = TABLES_DIR / "d4_plausibility.json"
    with open(out_path, "w") as f:
        json.dump(d4, f, indent=2)
    logger.info(f"Saved {out_path}")

    print("\n-- D4 Clinical Plausibility (top-5 SHAP features in literature set) --")
    for strat, v in d4.items():
        print(f"  {strat:<16} score={v['score']:.2f}  "
              f"{'YES' if v['meets_target'] else 'no ':>3}  top5={v['top_features']}")
    return d4


if __name__ == "__main__":
    main()
