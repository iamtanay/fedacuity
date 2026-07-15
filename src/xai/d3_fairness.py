"""
FedAcuity -- M4.4 D3 Explanation / Outcome Fairness (Contribution C3)

D3 asks: does a model make equally reliable decisions across the three care-type
subgroups (MC, SNF, IL), or does it systematically under-serve one?

We report the two standard group-fairness criteria across MC/SNF/IL, each model
scored on the subgroup via the exact model it deploys there (global model for
FedAvg/FedProx/Centralised; the care-type cluster model for CFL; the care-type
local model for the local baseline):

  * Equalized Odds gap  = 0.5 * (range(TPR) + range(FPR)) across subgroups.
    This is the HEADLINE fairness metric because it conditions on the true label
    and is therefore not confounded by the deliberately different base mismatch
    rates (MC~40%, SNF~28%, IL~12%).
  * Demographic Parity gap = range(positive-prediction rate) across subgroups.
    Reported for completeness only; it is expected to be large by construction
    (the subgroups genuinely differ in mismatch prevalence), so it is NOT used
    for the D3 score.

  score (D3) = 1 - equalized_odds_gap                (clipped to [0,1])

Hypothesis: CFL's per-cluster specialisation yields more uniform TPR/FPR across
care types than a single global model forced to compromise across all three.

Output: results/tables/d3_fairness.json

Usage:
    python -m src.xai.d3_fairness
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import cfg
from src.data.schema import CARE_TYPES
from src.xai.shap_pipeline import (
    reconstruct_strategy_models, get_care_type_test_sets, _features_for, STRATEGIES,
)
from src.data.loaders import load_facility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = Path(cfg["paths"]["results"]["tables"])


def _tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    pos = y_true == 1
    neg = y_true == 0
    tpr = float(np.sum(y_pred[pos] == 1) / max(np.sum(pos), 1))
    fpr = float(np.sum(y_pred[neg] == 1) / max(np.sum(neg), 1))
    return tpr, fpr


def compute_d3(strategy_models_by_strat, test_sets, features: List[str]) -> Dict[str, dict]:
    """Return {strategy: {per_subgroup, equalized_odds_gap, demographic_parity_gap, score}}."""
    out = {}
    for strat in STRATEGIES:
        models = strategy_models_by_strat[strat]
        tprs, fprs, ppr = {}, {}, {}
        per_subgroup = {}
        for ct in CARE_TYPES:
            X_df, y_ser = test_sets[ct]
            X = X_df[features].values
            y = y_ser.values
            proba = models[ct].predict_proba(X)[:, 1]
            y_pred = (proba > 0.5).astype(int)
            tpr, fpr = _tpr_fpr(y, y_pred)
            tprs[ct], fprs[ct] = tpr, fpr
            ppr[ct] = float(np.mean(y_pred))
            per_subgroup[ct] = {"tpr": round(tpr, 4), "fpr": round(fpr, 4),
                                "pos_pred_rate": round(ppr[ct], 4)}

        eo_gap = 0.5 * ((max(tprs.values()) - min(tprs.values())) +
                        (max(fprs.values()) - min(fprs.values())))
        dp_gap = max(ppr.values()) - min(ppr.values())
        score = float(np.clip(1.0 - eo_gap, 0.0, 1.0))
        out[strat] = {
            "per_subgroup": per_subgroup,
            "equalized_odds_gap": round(float(eo_gap), 4),
            "demographic_parity_gap": round(float(dp_gap), 4),
            "score": round(score, 4),
        }
    return out


def main():
    logger.info("Reconstructing models for D3 fairness (this runs the FL sims) ...")
    models = reconstruct_strategy_models()
    test_sets = get_care_type_test_sets()
    features = _features_for(load_facility(0))
    d3 = compute_d3(models, test_sets, features)

    out_path = TABLES_DIR / "d3_fairness.json"
    with open(out_path, "w") as f:
        json.dump(d3, f, indent=2)
    logger.info(f"Saved {out_path}")

    print("\n-- D3 Fairness (Equalized Odds across MC/SNF/IL) --")
    print(f"  {'Strategy':<16} {'EO gap':>8} {'DP gap':>8} {'score':>8}")
    for strat, v in d3.items():
        print(f"  {strat:<16} {v['equalized_odds_gap']:>8.3f} "
              f"{v['demographic_parity_gap']:>8.3f} {v['score']:>8.3f}")
    return d3


if __name__ == "__main__":
    main()
