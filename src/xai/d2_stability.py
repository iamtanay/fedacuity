"""
FedAcuity -- M4.3 D2 Explanation Stability (Contribution C3)

D2 asks: how much does a model's SHAP explanation move when its inputs are
perturbed by clinically-negligible measurement noise?

Method (config: xai.d2_stability):
  * Compute base SHAP on the per-care-type test partition (shap_pipeline).
  * For n_perturbations (100) draws, add +/-noise_level (5%) multiplicative
    Gaussian noise to the continuous features, recompute SHAP, and measure the
    mean absolute element-wise SHAP shift |SHAP_perturbed - SHAP_base|.
  * raw_shift = mean over all perturbations, instances and features. Lower is
    more stable (an explanation that swings under trivial noise is untrustworthy
    for clinical audit).

Normalisation to a [0,1] "stability score" (higher = better) uses a relative
stability index so the axis is comparable across models and readable on the
radar chart:

  score(strategy) = min_shift_over_strategies / raw_shift(strategy)   in (0,1]

The most stable strategy scores 1.0; others scale down proportionally. Raw
shifts are stored alongside for full transparency.

Hypothesis: care-type-specialised CFL cluster models explain a narrower, more
homogeneous input distribution and should be more stable than a single global
FedAvg model that must straddle all three care types.

Output: results/tables/d2_stability.json

Usage:
    python -m src.xai.d2_stability
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import shap

from src.config import cfg
from src.data.schema import CARE_TYPES
from src.xai.shap_pipeline import (
    reconstruct_strategy_models, get_care_type_test_sets, _shap_2d,
    _features_for, STRATEGIES,
)
from src.data.loaders import load_facility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = Path(cfg["paths"]["results"]["tables"])
SEED = cfg["project"]["seed"]
D2_CFG = cfg["xai"]["d2_stability"]
N_PERTURB = D2_CFG["n_perturbations"]
NOISE = D2_CFG["noise_level"]
CONTINUOUS = D2_CFG["continuous_features"]


def _raw_shift_for_strategy(
    strategy_models,
    test_sets: Dict[str, Tuple],
    features: List[str],
    rng: np.random.Generator,
) -> float:
    """Mean absolute element-wise SHAP shift under 5% Gaussian input noise."""
    cont_idx = [features.index(f) for f in CONTINUOUS if f in features]
    per_ct_shifts = []

    for ct in CARE_TYPES:
        model = strategy_models[ct]
        X_df, _ = test_sets[ct]
        X = X_df[features].values.astype(float)
        explainer = shap.TreeExplainer(model)
        base = _shap_2d(explainer, X)

        shifts = []
        for _ in range(N_PERTURB):
            Xp = X.copy()
            noise = rng.normal(1.0, NOISE, size=(X.shape[0], len(cont_idx)))
            Xp[:, cont_idx] = X[:, cont_idx] * noise
            sv = _shap_2d(explainer, Xp)
            shifts.append(float(np.mean(np.abs(sv - base))))
        per_ct_shifts.append(np.mean(shifts))

    return float(np.mean(per_ct_shifts))


def compute_d2(strategy_models_by_strat, test_sets, features) -> Dict[str, dict]:
    """Return {strategy: {raw_shift, score}} with relative-index normalisation."""
    raw = {}
    for strat in STRATEGIES:
        rng = np.random.default_rng(SEED)  # reset per strategy for reproducibility
        raw[strat] = _raw_shift_for_strategy(
            strategy_models_by_strat[strat], test_sets, features, rng)
        logger.info(f"  [{strat}] raw SHAP shift = {raw[strat]:.6f}")

    min_shift = min(raw.values())
    out = {}
    for strat in STRATEGIES:
        score = min_shift / raw[strat] if raw[strat] > 0 else 1.0
        out[strat] = {"raw_shift": round(raw[strat], 6), "score": round(float(score), 4)}
    return out


def main():
    logger.info("Reconstructing models for D2 stability (this runs the FL sims) ...")
    models = reconstruct_strategy_models()
    test_sets = get_care_type_test_sets()
    features = _features_for(load_facility(0))
    d2 = compute_d2(models, test_sets, features)

    out_path = TABLES_DIR / "d2_stability.json"
    with open(out_path, "w") as f:
        json.dump(d2, f, indent=2)
    logger.info(f"Saved {out_path}")

    print(f"\n-- D2 Explanation Stability ({N_PERTURB} x +/-{NOISE:.0%} noise) --")
    print(f"  {'Strategy':<16} {'raw shift':>12} {'stability score':>16}")
    for strat, v in d2.items():
        print(f"  {strat:<16} {v['raw_shift']:>12.6f} {v['score']:>16.3f}")
    return d2


if __name__ == "__main__":
    main()
