"""
FedAcuity — M2.6 FL Simulation Runner
Runs all five model variants and logs results.

Uses a manual round loop (no Ray dependency) — compatible with Flower 1.x.

Usage:
    python -m src.fl.simulation --strategy fedavg
    python -m src.fl.simulation --strategy fedprox --mu 0.1
    python -m src.fl.simulation --strategy clustered
    python -m src.fl.simulation --all   # Run all strategies sequentially
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

from src.fl.client import FedAcuityClient
from src.data.loaders import load_all_facilities, load_held_out, pool_all_data
from src.data.schema import FACILITY_CARE_TYPES, HELD_OUT_FACILITIES
from src.evaluation.logger import ResultsLogger
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(cfg["paths"]["results"]["logs"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = cfg["project"]["seed"]
FL_CFG = cfg["fl"]

TRAINING_FIDS = sorted([fid for fid in FACILITY_CARE_TYPES if fid not in HELD_OUT_FACILITIES])


# ── Local Baseline ─────────────────────────────────────────────────────────────

def run_local_baseline(results_logger: ResultsLogger) -> float:
    """M2: Local-only training — no federation."""
    logger.info("Running LOCAL baseline (no federation)")
    facilities = load_all_facilities()
    per_facility_auc = {}

    for fid, splits in facilities.items():
        X_train, y_train = splits["train"]
        X_val,   y_val   = splits["val"]

        model = xgb.XGBClassifier(
            n_estimators=FL_CFG["xgboost"]["n_estimators"],
            max_depth=FL_CFG["xgboost"]["max_depth"],
            learning_rate=FL_CFG["xgboost"]["learning_rate"],
            random_state=SEED + fid, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        X_test, y_test = splits["test"]
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        per_facility_auc[fid] = auc

    overall_auc = float(np.mean(list(per_facility_auc.values())))
    logger.info(f"LOCAL — Overall AUC: {overall_auc:.4f}")
    results_logger.log_round("local", 0, {"overall_auc": overall_auc, "per_facility": per_facility_auc})
    return overall_auc


# ── Centralised Oracle ─────────────────────────────────────────────────────────

def run_centralised_oracle(results_logger: ResultsLogger) -> float:
    """M2: Centralised oracle — pool all data (ignores HIPAA)."""
    logger.info("Running CENTRALISED oracle")
    X_train, y_train = pool_all_data()

    model = xgb.XGBClassifier(
        n_estimators=FL_CFG["xgboost"]["n_estimators"],
        max_depth=FL_CFG["xgboost"]["max_depth"],
        learning_rate=FL_CFG["xgboost"]["learning_rate"],
        random_state=SEED, verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)

    held_out = load_held_out()
    per_facility_auc = {}
    for fid, splits in held_out.items():
        X_test, y_test = splits["test"]
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        per_facility_auc[fid] = auc

    overall_auc = float(np.mean(list(per_facility_auc.values())))
    logger.info(f"CENTRALISED — Overall AUC: {overall_auc:.4f}")
    results_logger.log_round("centralised", 0, {"overall_auc": overall_auc})

    model_path = RESULTS_DIR / "centralised_model.json"
    model.get_booster().save_model(str(model_path))
    return overall_auc


# ── Manual FL Round Loop ───────────────────────────────────────────────────────

def _aggregate_xgb(client_results: Dict[int, Tuple[bytes, int]]) -> bytes:
    """
    Aggregate XGBoost models across clients.
    Strategy: return the model from the client with the most training data.
    This is a proxy for FedAvg — true tree merging is a Week 8 enhancement.
    """
    return max(client_results.values(), key=lambda x: x[1])[0]


def _run_manual_simulation(
    strategy_name: str,
    n_rounds: int,
    local_epochs_key: str = "fedavg",
    mu: Optional[float] = None,
) -> List[Dict]:
    """
    Self-contained FL simulation loop. No Ray or Flower server required.

    Aggregation: largest-client-model proxy (same for FedAvg, FedProx, CFL).
    CFL differentiator: aggregation happens *within* care-type clusters only,
    so each cluster maintains a specialised global model.
    """
    clients = {fid: FedAcuityClient(facility_id=fid, mu=mu) for fid in TRAINING_FIDS}

    # Model state — per-cluster for CFL, single global for others
    cluster_models: Dict[str, Optional[bytes]] = {"MC": None, "SNF": None, "IL": None}
    global_model_bytes: Optional[bytes] = None

    history: List[Dict] = []
    local_epochs = FL_CFG.get(local_epochs_key, FL_CFG["fedavg"])["local_epochs"]

    for round_num in range(1, n_rounds + 1):
        config = {"round": round_num, "local_epochs": local_epochs}
        if mu is not None:
            config["mu"] = mu

        # ── Fit phase ────────────────────────────────────────────────────────
        client_results: Dict[int, Tuple[bytes, int]] = {}

        for fid, client in clients.items():
            care_type = FACILITY_CARE_TYPES[fid]
            if strategy_name == "clustered":
                model_bytes = cluster_models.get(care_type)
            else:
                model_bytes = global_model_bytes

            params_in = ([np.frombuffer(model_bytes, dtype=np.uint8)]
                         if model_bytes is not None else [np.zeros(1)])

            new_params, n_samples, _ = client.fit(params_in, config)
            client_results[fid] = (new_params[0].tobytes(), n_samples)

        # ── Aggregate phase ───────────────────────────────────────────────────
        if strategy_name == "clustered":
            for care_type in ["MC", "SNF", "IL"]:
                ct_results = {fid: res for fid, res in client_results.items()
                              if FACILITY_CARE_TYPES[fid] == care_type}
                if ct_results:
                    cluster_models[care_type] = _aggregate_xgb(ct_results)
        else:
            global_model_bytes = _aggregate_xgb(client_results)

        # ── Evaluate (every 5 rounds or final round) ──────────────────────────
        if round_num % 5 == 0 or round_num == n_rounds:
            all_aucs, care_type_aucs = [], {"MC": [], "SNF": [], "IL": []}

            for fid, client in clients.items():
                care_type = FACILITY_CARE_TYPES[fid]
                eval_bytes = (cluster_models.get(care_type)
                              if strategy_name == "clustered" else global_model_bytes)
                if eval_bytes is None:
                    continue
                eval_params = [np.frombuffer(eval_bytes, dtype=np.uint8)]
                _, _, eval_metrics = client.evaluate(eval_params, config)
                auc = eval_metrics.get("auc", 0.0)
                all_aucs.append(auc)
                care_type_aucs[care_type].append(auc)

            metrics: Dict = {"round": round_num,
                             "overall_auc": float(np.mean(all_aucs)) if all_aucs else 0.0}
            for ct, aucs in care_type_aucs.items():
                if aucs:
                    metrics[f"{ct}_auc"] = float(np.mean(aucs))

            logger.info(f"Round {round_num:>3}/{n_rounds} [{strategy_name}] "
                        f"— AUC: {metrics['overall_auc']:.4f}  "
                        + "  ".join(f"{ct}:{metrics[f'{ct}_auc']:.3f}"
                                    for ct in ["MC", "SNF", "IL"] if f"{ct}_auc" in metrics))
            history.append(metrics)

    return history


# ── Strategy Runners ───────────────────────────────────────────────────────────

def run_fedavg(results_logger: ResultsLogger, n_rounds: int = None) -> List[Dict]:
    n_rounds = n_rounds or FL_CFG["rounds"]
    logger.info(f"Running FEDAVG ({n_rounds} rounds)")
    history = _run_manual_simulation("fedavg", n_rounds, local_epochs_key="fedavg")
    for entry in history:
        results_logger.log_round("fedavg", entry["round"], entry)
    if history:
        logger.info(f"FEDAVG — Final AUC: {history[-1]['overall_auc']:.4f}")
    return history


def run_fedprox(results_logger: ResultsLogger, mu: float = None, n_rounds: int = None) -> List[Dict]:
    mu = mu or FL_CFG["fedprox"]["mu_selected"]
    n_rounds = n_rounds or FL_CFG["rounds"]
    logger.info(f"Running FEDPROX (μ={mu}, {n_rounds} rounds)")
    history = _run_manual_simulation("fedprox", n_rounds, local_epochs_key="fedprox", mu=mu)
    for entry in history:
        results_logger.log_round(f"fedprox_mu{mu}", entry["round"], entry)
    if history:
        logger.info(f"FEDPROX — Final AUC: {history[-1]['overall_auc']:.4f}")
    return history


def run_clustered_fl(results_logger: ResultsLogger, n_rounds: int = None) -> List[Dict]:
    n_rounds = n_rounds or FL_CFG["rounds"]
    logger.info(f"Running CLUSTERED FL ({n_rounds} rounds) — C1 Core")
    history = _run_manual_simulation("clustered", n_rounds, local_epochs_key="clustered")
    for entry in history:
        results_logger.log_round("clustered_fl", entry["round"], entry)
    if history:
        logger.info(f"CFL — Final AUC: {history[-1]['overall_auc']:.4f}")
    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedAcuity FL Simulation")
    parser.add_argument("--strategy",
                        choices=["local", "centralised", "fedavg", "fedprox", "clustered", "all"],
                        default="fedavg")
    parser.add_argument("--mu", type=float, default=None, help="FedProx μ parameter")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of FL rounds")
    args = parser.parse_args()

    rlog = ResultsLogger()

    if args.strategy in ("local", "all"):
        run_local_baseline(rlog)
    if args.strategy in ("centralised", "all"):
        run_centralised_oracle(rlog)
    if args.strategy in ("fedavg", "all"):
        run_fedavg(rlog, n_rounds=args.rounds)
    if args.strategy in ("fedprox", "all"):
        run_fedprox(rlog, mu=args.mu, n_rounds=args.rounds)
    if args.strategy in ("clustered", "all"):
        run_clustered_fl(rlog, n_rounds=args.rounds)

    rlog.save()
    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
