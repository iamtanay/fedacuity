"""
FedAcuity — M2.6 FL Simulation Runner
Runs all five model variants and logs results.

Usage:
    python src/fl/simulation.py --strategy fedavg
    python src/fl/simulation.py --strategy fedprox --mu 0.1
    python src/fl/simulation.py --strategy clustered
    python src/fl/simulation.py --all   # Run all strategies sequentially
"""

import argparse
import json
import logging
from pathlib import Path

import flwr as fl
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

from src.fl.client import FedAcuityClient
from src.fl.clustered_fl import ClusteredFLServer
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


# ── Local Baseline ─────────────────────────────────────────────────────────────

def run_local_baseline(results_logger: ResultsLogger):
    """M2: Local-only training — no federation."""
    logger.info("Running LOCAL baseline (no federation)")
    facilities = load_all_facilities()
    held_out = load_held_out()
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
    held_out = load_held_out()

    model = xgb.XGBClassifier(
        n_estimators=FL_CFG["xgboost"]["n_estimators"],
        max_depth=FL_CFG["xgboost"]["max_depth"],
        learning_rate=FL_CFG["xgboost"]["learning_rate"],
        random_state=SEED, verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)

    per_facility_auc = {}
    for fid, splits in held_out.items():
        X_test, y_test = splits["test"]
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        per_facility_auc[fid] = auc

    overall_auc = float(np.mean(list(per_facility_auc.values())))
    logger.info(f"CENTRALISED — Overall AUC: {overall_auc:.4f}")
    results_logger.log_round("centralised", 0, {"overall_auc": overall_auc})

    # Save model for XAI (centralised is the oracle reference)
    model_path = RESULTS_DIR / "centralised_model.json"
    model.get_booster().save_model(str(model_path))
    return overall_auc


# ── Flower Simulation Helpers ──────────────────────────────────────────────────

def client_fn(cid: str, mu: float = None):
    """Factory: create FedAcuityClient for a given CID."""
    return FedAcuityClient(facility_id=int(cid), mu=mu)


def run_flower_simulation(strategy_name: str, strategy, n_rounds: int, mu: float = None) -> dict:
    """Generic Flower simulation runner."""
    training_fids = [str(fid) for fid in FACILITY_CARE_TYPES if fid not in HELD_OUT_FACILITIES]

    def _client_fn(cid):
        return client_fn(cid, mu=mu)

    history = fl.simulation.start_simulation(
        client_fn=_client_fn,
        clients_ids=training_fids,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )
    return history


# ── FedAvg ────────────────────────────────────────────────────────────────────

def run_fedavg(results_logger: ResultsLogger, n_rounds: int = None):
    n_rounds = n_rounds or FL_CFG["rounds"]
    logger.info(f"Running FEDAVG ({n_rounds} rounds)")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FL_CFG["fraction_fit"],
        min_available_clients=FL_CFG["min_available_clients"],
        on_fit_config_fn=lambda r: {"local_epochs": FL_CFG["fedavg"]["local_epochs"], "round": r},
    )
    history = run_flower_simulation("fedavg", strategy, n_rounds)

    for rnd, (loss, metrics) in enumerate(zip(
        history.losses_distributed, history.metrics_distributed.get("auc", [])
    )):
        results_logger.log_round("fedavg", rnd + 1, {"auc": metrics[1] if metrics else None})

    return history


# ── FedProx ───────────────────────────────────────────────────────────────────

def run_fedprox(results_logger: ResultsLogger, mu: float = None, n_rounds: int = None):
    mu = mu or FL_CFG["fedprox"]["mu_selected"]
    n_rounds = n_rounds or FL_CFG["rounds"]
    logger.info(f"Running FEDPROX (μ={mu}, {n_rounds} rounds)")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FL_CFG["fraction_fit"],
        min_available_clients=FL_CFG["min_available_clients"],
        on_fit_config_fn=lambda r: {
            "local_epochs": FL_CFG["fedprox"]["local_epochs"],
            "mu": mu,
            "round": r,
        },
    )
    history = run_flower_simulation("fedprox", strategy, n_rounds, mu=mu)

    for rnd, (loss, metrics) in enumerate(zip(
        history.losses_distributed, history.metrics_distributed.get("auc", [])
    )):
        results_logger.log_round(f"fedprox_mu{mu}", rnd + 1, {"auc": metrics[1] if metrics else None})

    return history


# ── Clustered FL (C1 Core) ────────────────────────────────────────────────────

def run_clustered_fl(results_logger: ResultsLogger, n_rounds: int = None):
    n_rounds = n_rounds or FL_CFG["rounds"]
    logger.info(f"Running CLUSTERED FL ({n_rounds} rounds) — C1 Core")

    strategy = ClusteredFLServer(min_available_clients=FL_CFG["min_available_clients"])
    history = run_flower_simulation("clustered", strategy, n_rounds)

    for rnd, (_, metrics) in enumerate(zip(
        history.losses_distributed or [None] * n_rounds,
        history.metrics_distributed.get("overall_auc", [])
    )):
        results_logger.log_round("clustered_fl", rnd + 1, {"overall_auc": metrics[1] if metrics else None})

    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedAcuity FL Simulation")
    parser.add_argument("--strategy", choices=["local", "centralised", "fedavg", "fedprox", "clustered", "all"],
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
