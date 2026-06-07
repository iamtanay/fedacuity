"""
FedAcuity -- Held-Out Evaluation (Table II)
Evaluates all 5 strategies on held-out facilities 8 and 9 (both IL).
Computes AUC-ROC, F1, Precision, Recall on the same 50-round setting
described in the paper's experimental setup section.

Local baselines provided:
  - "IL Care-Type Local": trained on facility 7 only (same care type, no federation)
  - "Cross-Facility Ensemble": 8-model ensemble (labelled accurately, not as "Local")

For FL strategies: runs a 50-round simulation, then evaluates the final ensemble.

Usage:
    python -m src.evaluation.eval_held_out
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.fl.client import FedAcuityClient, deserialize_xgb_model
from src.data.loaders import load_all_facilities, load_held_out, pool_all_data, load_facility, get_facility_splits
from src.data.schema import FACILITY_CARE_TYPES, HELD_OUT_FACILITIES, FEATURE_NAMES
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOGS_DIR   = Path(cfg["paths"]["results"]["logs"])
TABLES_DIR = Path(cfg["paths"]["results"]["tables"])
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SEED     = cfg["project"]["seed"]
FL_CFG   = cfg["fl"]
XGB_CFG  = FL_CFG["xgboost"]

# Must match paper's stated experimental setup
EVAL_ROUNDS   = 50
TRAINING_FIDS = sorted([fid for fid in FACILITY_CARE_TYPES if fid not in HELD_OUT_FACILITIES])


def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba > 0.5).astype(int)
    return {
        "auc_roc":   round(float(roc_auc_score(y_true, y_proba)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def _aggregate_xgb_consensus(
    client_results: Dict[int, Tuple[bytes, int]],
    reference_X: np.ndarray,
) -> bytes:
    """Prediction-consensus aggregation (mirrors simulation.py)."""
    total_samples = sum(n for _, n in client_results.values())
    predictions, weights, bytes_list = [], [], []
    for fid, (model_bytes, n_samples) in client_results.items():
        model = deserialize_xgb_model(model_bytes)
        proba = model.predict_proba(reference_X)[:, 1]
        weight = n_samples / total_samples
        predictions.append(proba)
        weights.append(weight)
        bytes_list.append(model_bytes)
    ensemble_proba = np.average(predictions, axis=0, weights=weights)
    distances = [float(np.mean((p - ensemble_proba) ** 2)) for p in predictions]
    return bytes_list[int(np.argmin(distances))]


def _ensemble_predict_held_out(
    client_results: Dict[int, Tuple[bytes, int]],
) -> dict:
    """Weighted ensemble prediction across all client models on held-out facilities."""
    held_out = load_held_out()
    per_facility = {}
    all_y_true, all_y_proba = [], []

    for fid, splits in held_out.items():
        X_test, y_test = splits["test"]
        total_samples = sum(n for _, n in client_results.values())
        proba_sum = np.zeros(len(X_test))

        for _, (model_bytes, n_samples) in client_results.items():
            model = deserialize_xgb_model(model_bytes)
            weight = n_samples / total_samples
            proba_sum += model.predict_proba(X_test.values)[:, 1] * weight

        per_facility[str(fid)] = _compute_metrics(y_test.values, proba_sum)
        all_y_true.append(y_test.values)
        all_y_proba.append(proba_sum)

    overall = _compute_metrics(np.concatenate(all_y_true), np.concatenate(all_y_proba))
    return {"overall": overall, "per_facility": per_facility}


def _run_fl_50rounds(strategy: str) -> Dict[int, Tuple[bytes, int]]:
    """Run 50-round FL simulation and return final round client results."""
    clients = {fid: FedAcuityClient(facility_id=fid) for fid in TRAINING_FIDS}

    cluster_models: Dict[str, bytes | None] = {"MC": None, "SNF": None, "IL": None}
    cluster_all_results: Dict[str, Dict[int, Tuple[bytes, int]]] = {"MC": {}, "SNF": {}, "IL": {}}
    global_model_bytes: bytes | None = None
    global_all_results: Dict[int, Tuple[bytes, int]] = {}

    ref_parts = [c.X_val.values[:25] for c in clients.values()]
    reference_X = np.vstack(ref_parts)

    config = {"round": 1, "local_epochs": FL_CFG["fedavg"]["local_epochs"]}

    for rnd in range(1, EVAL_ROUNDS + 1):
        config["round"] = rnd
        client_results: Dict[int, Tuple[bytes, int]] = {}

        for fid, client in clients.items():
            care_type = FACILITY_CARE_TYPES[fid]
            model_bytes = cluster_models[care_type] if strategy == "clustered" else global_model_bytes
            params_in = ([np.frombuffer(model_bytes, dtype=np.uint8)]
                         if model_bytes is not None else [np.zeros(1)])
            new_params, n_samples, _ = client.fit(params_in, config)
            client_results[fid] = (new_params[0].tobytes(), n_samples)

        if strategy == "clustered":
            for care_type in ["MC", "SNF", "IL"]:
                ct_res = {fid: res for fid, res in client_results.items()
                          if FACILITY_CARE_TYPES[fid] == care_type}
                if ct_res:
                    cluster_models[care_type] = _aggregate_xgb_consensus(ct_res, reference_X)
                    cluster_all_results[care_type] = ct_res
        else:
            global_model_bytes = _aggregate_xgb_consensus(client_results, reference_X)
            global_all_results = client_results

        if rnd % 10 == 0:
            logger.info(f"  [{strategy}] round {rnd}/{EVAL_ROUNDS}")

    if strategy == "clustered":
        # Return IL cluster results (held-out facilities are IL)
        return cluster_all_results.get("IL", {})
    return global_all_results


# ── Strategy evaluations ──────────────────────────────────────────────────────

def eval_il_local() -> dict:
    """
    IL Care-Type Local Baseline: train on facility 7 only (same care type as held-out).
    This is the correct local baseline — care-type-matched, single facility, no federation.
    """
    logger.info("Evaluating IL CARE-TYPE LOCAL baseline (facility 7 only)")
    df7 = load_facility(7)
    (X_train, y_train), (X_val, y_val), _ = get_facility_splits(7, df7)

    model = xgb.XGBClassifier(
        n_estimators=XGB_CFG["n_estimators"], max_depth=XGB_CFG["max_depth"],
        learning_rate=XGB_CFG["learning_rate"], random_state=SEED + 7, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    held_out = load_held_out()
    per_facility, all_y_true, all_y_proba = {}, [], []
    for fid, splits in held_out.items():
        X_test, y_test = splits["test"]
        y_proba = model.predict_proba(X_test.values)[:, 1]
        per_facility[str(fid)] = _compute_metrics(y_test.values, y_proba)
        all_y_true.append(y_test.values)
        all_y_proba.append(y_proba)
    overall = _compute_metrics(np.concatenate(all_y_true), np.concatenate(all_y_proba))
    return {"overall": overall, "per_facility": per_facility}


def eval_cross_facility_ensemble() -> dict:
    """
    Cross-Facility Ensemble (no aggregation): 8 local models, weighted average predictions.
    This is NOT a 'local' baseline — it represents the best possible ensemble without
    any federation protocol. Labelled accurately in Table II.
    """
    logger.info("Evaluating CROSS-FACILITY ENSEMBLE (8 models, no aggregation)")
    facilities = load_all_facilities()
    held_out = load_held_out()

    per_facility, all_y_true, all_y_proba = {}, [], []
    for fid_h, splits in held_out.items():
        X_test, y_test = splits["test"]
        proba_sum = np.zeros(len(X_test))
        total_n = sum(len(s["train"][0]) for s in facilities.values())

        for fid_t, train_splits in facilities.items():
            X_train, y_train = train_splits["train"]
            X_val, y_val     = train_splits["val"]
            model = xgb.XGBClassifier(
                n_estimators=XGB_CFG["n_estimators"], max_depth=XGB_CFG["max_depth"],
                learning_rate=XGB_CFG["learning_rate"], random_state=SEED + fid_t, verbosity=0,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            weight = len(X_train) / total_n
            proba_sum += model.predict_proba(X_test.values)[:, 1] * weight

        per_facility[str(fid_h)] = _compute_metrics(y_test.values, proba_sum)
        all_y_true.append(y_test.values)
        all_y_proba.append(proba_sum)

    overall = _compute_metrics(np.concatenate(all_y_true), np.concatenate(all_y_proba))
    return {"overall": overall, "per_facility": per_facility}


def eval_centralised() -> dict:
    logger.info("Evaluating CENTRALISED oracle on held-out facilities")
    model_path = LOGS_DIR / "centralised_model.json"
    if model_path.exists():
        model = xgb.XGBClassifier(
            n_estimators=XGB_CFG["n_estimators"], max_depth=XGB_CFG["max_depth"],
            learning_rate=XGB_CFG["learning_rate"], verbosity=0,
        )
        model.load_model(str(model_path))
    else:
        logger.warning("centralised_model.json not found — retraining")
        X_train, y_train = pool_all_data()
        model = xgb.XGBClassifier(
            n_estimators=XGB_CFG["n_estimators"], max_depth=XGB_CFG["max_depth"],
            learning_rate=XGB_CFG["learning_rate"], random_state=SEED, verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)

    held_out = load_held_out()
    per_facility, all_y_true, all_y_proba = {}, [], []
    for fid, splits in held_out.items():
        X_test, y_test = splits["test"]
        y_proba = model.predict_proba(X_test.values)[:, 1]
        per_facility[str(fid)] = _compute_metrics(y_test.values, y_proba)
        all_y_true.append(y_test.values)
        all_y_proba.append(y_proba)
    overall = _compute_metrics(np.concatenate(all_y_true), np.concatenate(all_y_proba))
    return {"overall": overall, "per_facility": per_facility}


def eval_fedavg() -> dict:
    logger.info(f"Evaluating FEDAVG (ensemble, {EVAL_ROUNDS} rounds) on held-out")
    final_results = _run_fl_50rounds("fedavg")
    return _ensemble_predict_held_out(final_results)


def eval_fedprox() -> dict:
    logger.info(f"Evaluating FEDPROX ({EVAL_ROUNDS} rounds) on held-out")
    # FedProx with XGBoost is equivalent to FedAvg (proximal term requires gradient access).
    # We run the same simulation as FedAvg with mu logged but unused.
    final_results = _run_fl_50rounds("fedprox")
    return _ensemble_predict_held_out(final_results)


def eval_clustered_fl() -> dict:
    logger.info(f"Evaluating CLUSTERED FL (IL cluster, {EVAL_ROUNDS} rounds) on held-out")
    # Note: IL cluster has 1 training client (facility 7). No aggregation occurs in IL.
    # This is a single-client IL model evaluated on 2 unseen IL facilities.
    final_results = _run_fl_50rounds("clustered")
    if not final_results:
        logger.warning("IL cluster returned no results")
        return {}
    return _ensemble_predict_held_out(final_results)


# ── Main ──────────────────────────────────────────────────────────────────────

EVALUATORS = {
    "il_local_baseline":        eval_il_local,
    "cross_facility_ensemble":  eval_cross_facility_ensemble,
    "centralised":              eval_centralised,
    "fedavg":                   eval_fedavg,
    "fedprox":                  eval_fedprox,
    "clustered_fl":             eval_clustered_fl,
}

DISPLAY_NAMES = {
    "il_local_baseline":       "IL Local (facility 7 only)",
    "cross_facility_ensemble": "Cross-Facility Ensemble (no aggregation)",
    "centralised":             "Centralised Oracle",
    "fedavg":                  "FedAvg",
    "fedprox":                 "FedProx (mu=0.1, equiv. FedAvg for XGBoost)",
    "clustered_fl":            "Clustered FL [C1]",
}


def main():
    all_results = {}
    for name, fn in EVALUATORS.items():
        try:
            all_results[name] = fn()
            logger.info(f"[{name}] done")
        except Exception as e:
            logger.error(f"[{name}] FAILED: {e}")
            all_results[name] = {"error": str(e)}

    out_path = TABLES_DIR / "fl_held_out_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved: {out_path}")

    print("\n-- Table II: Held-Out Evaluation (Facilities 8 & 9, IL) ----------------")
    print(f"  {'Strategy':<45} {'AUC-ROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*80}")
    for name, display in DISPLAY_NAMES.items():
        result = all_results.get(name, {})
        if "error" in result:
            print(f"  {display:<45}  ERROR: {result['error'][:40]}")
        else:
            ov = result.get("overall", {})
            print(f"  {display:<45} {ov.get('auc_roc', 0):.4f}   "
                  f"{ov.get('f1', 0):.4f}   "
                  f"{ov.get('precision', 0):.4f}     "
                  f"{ov.get('recall', 0):.4f}")


if __name__ == "__main__":
    main()
