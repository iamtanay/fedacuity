"""
FedAcuity -- M4.1 SHAP Pipeline (Contribution C3 foundation)

Reconstructs the *deployed* model for each of the five strategies and computes
SHAP values with ``shap.TreeExplainer`` on a per-care-type test partition.

Design decisions (documented so the audit is defensible):

  * WHICH MODEL IS EXPLAINED. Each strategy is explained via the exact model it
    would deploy for a given care type:
      - centralised : the single pooled oracle (results/logs/centralised_model.json)
      - local       : the care-type-matched single-facility baseline
                      (MC->facility 0, SNF->facility 3, IL->facility 7) -- the
                      same local models used in Table II (eval_held_out.py)
      - fedavg      : the final-round global *representative consensus* model
                      (the model that is actually broadcast to clients each
                      round -- ``_aggregate_xgb_consensus``)
      - fedprox     : identical to fedavg for XGBoost (documented equivalence;
                      the proximal term is a no-op without gradient access)
      - clustered_fl: the final-round representative consensus model of the
                      care type's OWN cluster (MC/SNF/IL cluster respectively)
    Explaining the representative consensus model (rather than the evaluation
    ensemble) is the honest choice: it is the single tree model each strategy
    hands to a facility, and it is what a clinician would audit.

  * WHICH DATA. SHAP is computed on a per-care-type test partition covering all
    three distributions. SNF and IL are drawn from the *fully held-out*
    facilities (6 and 9); MC is drawn from a representative MC facility's test
    split (no MC facility is held out by design). This mirrors Table II and lets
    D3 Fairness be measured across all three subgroups.

Outputs (per strategy) -> results/tables/shap_values_<strategy>.npz:
    shap        : (N, F) SHAP matrix, rows concatenated across care types
    X           : (N, F) feature matrix aligned with shap
    care_type   : (N,)   care-type label per row ('MC'/'SNF'/'IL')
    y           : (N,)   ground-truth label per row
    mean_abs    : (F,)   mean(|SHAP|) per feature (global importance vector)
    features    : (F,)   feature names

Usage:
    python -m src.xai.shap_pipeline
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import shap
import xgboost as xgb

from src.config import cfg
from src.data.loaders import load_facility, get_facility_splits
from src.data.schema import (
    FEATURE_NAMES, FACILITY_CARE_TYPES, HELD_OUT_FACILITIES,
    CLUSTER_ASSIGNMENTS, CARE_TYPES,
)
from src.fl.client import deserialize_xgb_model
from src.evaluation.eval_held_out import _run_fl_50rounds, _aggregate_xgb_consensus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = Path(cfg["paths"]["results"]["tables"])
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path(cfg["paths"]["results"]["logs"])

SEED = cfg["project"]["seed"]
XGB_CFG = cfg["fl"]["xgboost"]
SHAP_CFG = cfg["xai"]["shap"]

STRATEGIES = ["local", "centralised", "fedavg", "fedprox", "clustered_fl"]

TRAINING_FIDS = sorted(fid for fid in FACILITY_CARE_TYPES if fid not in HELD_OUT_FACILITIES)

# Which facility supplies the per-care-type test partition used for the XAI audit.
# SNF + IL come from the held-out facilities (6, 9); MC has no held-out facility,
# so a representative MC facility's own test split is used.
CARE_TYPE_TEST_FACILITY: Dict[str, int] = {}
for ct in CARE_TYPES:
    held = [f for f in CLUSTER_ASSIGNMENTS[ct] if f in HELD_OUT_FACILITIES]
    CARE_TYPE_TEST_FACILITY[ct] = held[0] if held else min(CLUSTER_ASSIGNMENTS[ct])

# Which single facility trains the care-type-matched "local" baseline model.
CARE_TYPE_LOCAL_FACILITY: Dict[str, int] = {
    ct: min(f for f in CLUSTER_ASSIGNMENTS[ct] if f in TRAINING_FIDS)
    for ct in CARE_TYPES
}


# ── Feature matrix helper ───────────────────────────────────────────────────────

def _features_for(df) -> List[str]:
    return [f for f in FEATURE_NAMES if f in df.columns]


# ── Test partitions ─────────────────────────────────────────────────────────────

def get_care_type_test_sets() -> Dict[str, Tuple["pd.DataFrame", "pd.Series"]]:
    """Return {care_type: (X_test_df, y_test_series)} for MC, SNF, IL."""
    sets = {}
    for ct, fid in CARE_TYPE_TEST_FACILITY.items():
        df = load_facility(fid)
        _, _, (X_test, y_test) = get_facility_splits(fid, df)
        sets[ct] = (X_test, y_test)
        logger.info(f"Test partition {ct}: facility {fid}, {len(X_test)} rows "
                    f"(held-out={fid in HELD_OUT_FACILITIES})")
    return sets


# ── Local baseline models (care-type matched, single facility) ──────────────────

def _train_local_model(fid: int) -> xgb.XGBClassifier:
    df = load_facility(fid)
    (X_train, y_train), (X_val, y_val), _ = get_facility_splits(fid, df)
    model = xgb.XGBClassifier(
        n_estimators=XGB_CFG["n_estimators"], max_depth=XGB_CFG["max_depth"],
        learning_rate=XGB_CFG["learning_rate"], eval_metric=XGB_CFG["eval_metric"],
        random_state=SEED + fid, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _load_centralised_model() -> xgb.XGBClassifier:
    model_path = LOGS_DIR / "centralised_model.json"
    model = xgb.XGBClassifier(
        n_estimators=XGB_CFG["n_estimators"], max_depth=XGB_CFG["max_depth"],
        learning_rate=XGB_CFG["learning_rate"], verbosity=0,
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} missing -- run `python -m src.fl.simulation --strategy centralised` first."
        )
    model.load_model(str(model_path))
    return model


def _reference_X() -> np.ndarray:
    """Shared reference set for consensus aggregation (mirrors eval_held_out)."""
    from src.fl.client import FedAcuityClient
    parts = []
    for fid in TRAINING_FIDS:
        c = FedAcuityClient(facility_id=fid)
        parts.append(c.X_val.values[:25])
    return np.vstack(parts)


def reconstruct_strategy_models() -> Dict[str, Dict[str, xgb.XGBClassifier]]:
    """
    Build {strategy: {care_type: XGBClassifier}} -- the model each strategy
    deploys for each care type. See module docstring for the rationale.
    """
    logger.info("Reconstructing deployed models for all 5 strategies ...")
    reference_X = _reference_X()

    # Centralised oracle -- one model for every care type
    oracle = _load_centralised_model()
    centralised = {ct: oracle for ct in CARE_TYPES}

    # Care-type-matched local baselines
    local = {ct: _train_local_model(fid) for ct, fid in CARE_TYPE_LOCAL_FACILITY.items()}

    # FedAvg global representative consensus (final round)
    logger.info("  running 50-round FedAvg simulation for global consensus model ...")
    global_results = _run_fl_50rounds("fedavg")
    fedavg_rep = deserialize_xgb_model(_aggregate_xgb_consensus(global_results, reference_X))
    fedavg = {ct: fedavg_rep for ct in CARE_TYPES}

    # FedProx == FedAvg for XGBoost (documented)
    fedprox = fedavg

    # Clustered FL -- each care type explained by its OWN cluster consensus
    logger.info("  running 50-round Clustered FL simulation for per-cluster consensus ...")
    cluster_results = _run_fl_50rounds("clustered")
    clustered = {}
    for ct in CARE_TYPES:
        ct_res = cluster_results.get(ct, {})
        if not ct_res:
            logger.warning(f"  no cluster results for {ct}; falling back to fedavg model")
            clustered[ct] = fedavg_rep
        else:
            clustered[ct] = deserialize_xgb_model(_aggregate_xgb_consensus(ct_res, reference_X))

    return {
        "local": local,
        "centralised": centralised,
        "fedavg": fedavg,
        "fedprox": fedprox,
        "clustered_fl": clustered,
    }


# ── SHAP computation ────────────────────────────────────────────────────────────

def _shap_2d(explainer: "shap.TreeExplainer", X: np.ndarray) -> np.ndarray:
    """Return SHAP values as a clean (n, F) float array for the positive class."""
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):          # [class0, class1] -> positive class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, F, classes) -> positive class
        sv = sv[..., -1]
    return sv.astype(float)


def compute_shap_for_strategy(
    strategy_models: Dict[str, xgb.XGBClassifier],
    test_sets: Dict[str, Tuple["pd.DataFrame", "pd.Series"]],
    features: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute SHAP for one strategy across all care-type partitions, then
    concatenate. Returns dict with keys: shap, X, care_type, y, mean_abs, features.
    """
    shap_parts, X_parts, ct_parts, y_parts = [], [], [], []
    for ct in CARE_TYPES:
        model = strategy_models[ct]
        X_df, y_ser = test_sets[ct]
        X = X_df[features].values
        explainer = shap.TreeExplainer(model)
        sv = _shap_2d(explainer, X)
        shap_parts.append(sv)
        X_parts.append(X)
        ct_parts.append(np.array([ct] * len(X)))
        y_parts.append(y_ser.values)

    shap_all = np.vstack(shap_parts)
    X_all = np.vstack(X_parts)
    ct_all = np.concatenate(ct_parts)
    y_all = np.concatenate(y_parts)
    mean_abs = np.mean(np.abs(shap_all), axis=0)
    return {
        "shap": shap_all, "X": X_all, "care_type": ct_all, "y": y_all,
        "mean_abs": mean_abs, "features": np.array(features),
    }


def run_pipeline() -> Dict[str, Dict[str, np.ndarray]]:
    """Full pipeline: reconstruct models, compute + save SHAP for all strategies."""
    test_sets = get_care_type_test_sets()
    # feature list is identical across facilities
    any_df = load_facility(TRAINING_FIDS[0])
    features = _features_for(any_df)
    logger.info(f"Using {len(features)} features: {features}")

    models = reconstruct_strategy_models()
    results = {}
    for strat in STRATEGIES:
        logger.info(f"Computing SHAP for [{strat}] ...")
        out = compute_shap_for_strategy(models[strat], test_sets, features)
        results[strat] = out
        out_path = TABLES_DIR / f"shap_values_{strat}.npz"
        np.savez_compressed(
            out_path,
            shap=out["shap"], X=out["X"], care_type=out["care_type"],
            y=out["y"], mean_abs=out["mean_abs"], features=out["features"],
        )
        top = np.array(features)[np.argsort(out["mean_abs"])[::-1][:5]]
        logger.info(f"  saved {out_path.name} | N={len(out['shap'])} | top-5: {list(top)}")

    return results


def load_shap_results() -> Dict[str, Dict[str, np.ndarray]]:
    """Load previously-saved shap_values_<strategy>.npz for all strategies."""
    results = {}
    for strat in STRATEGIES:
        path = TABLES_DIR / f"shap_values_{strat}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run `python -m src.xai.shap_pipeline` first.")
        with np.load(path, allow_pickle=True) as npz:
            results[strat] = {k: npz[k] for k in npz.files}
    return results


if __name__ == "__main__":
    run_pipeline()
    print("\nSHAP pipeline complete -- shap_values_*.npz written to results/tables/")
