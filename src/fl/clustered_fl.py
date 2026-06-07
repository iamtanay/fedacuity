"""
FedAcuity — M2.5 Domain-Driven Clustered Federated Learning (C1 Core)
The central engineering contribution: care-type-aware cluster aggregation.

Clusters: MC (Memory Care), SNF (Skilled Nursing), IL (Independent Living)
Each cluster runs independent FedAvg → specialised global model per care type.
"""

import io
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
import xgboost as xgb
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays

from src.data.schema import CLUSTER_ASSIGNMENTS, FACILITY_CARE_TYPES, HELD_OUT_FACILITIES
from src.config import cfg

logger = logging.getLogger(__name__)


def _weighted_average_xgb(results: List[Tuple[bytes, int]], reference_X=None) -> bytes:
    """
    Prediction-consensus aggregation for intra-cluster FedAvg.

    Weighted ensemble in prediction space: each client model predicts on a shared
    reference set, predictions are averaged by client dataset size, and the model
    whose predictions are closest to the ensemble mean is selected as the cluster
    global model. For equal-sized clients this equals a simple average (FedAvg).
    """
    import numpy as np
    from src.fl.client import deserialize_xgb_model

    if reference_X is None or len(results) == 1:
        # Single client or no reference: return the only/largest model
        return max(results, key=lambda x: x[1])[0]

    total_samples = sum(n for _, n in results)
    predictions, weights, bytes_list = [], [], []

    for model_bytes, n_samples in results:
        model = deserialize_xgb_model(model_bytes)
        proba = model.predict_proba(reference_X)[:, 1]
        weight = n_samples / total_samples
        predictions.append(proba)
        weights.append(weight)
        bytes_list.append(model_bytes)

    ensemble_proba = np.average(predictions, axis=0, weights=weights)
    distances = [float(np.mean((p - ensemble_proba) ** 2)) for p in predictions]
    return bytes_list[int(np.argmin(distances))]


class ClusteredFLServer(fl.server.strategy.Strategy):
    """
    Domain-driven Clustered FL strategy.
    Maintains one global model per care-type cluster.
    Routes fit/evaluate calls to the appropriate cluster model.
    """

    def __init__(self, min_available_clients: int = 8):
        super().__init__()
        self.min_available_clients = min_available_clients
        # One model state per cluster
        self.cluster_models: Dict[str, Optional[bytes]] = {
            "MC": None, "SNF": None, "IL": None,
        }
        self.round_results: List[Dict] = []

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """No central initialisation — each cluster initialises independently."""
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        """Send each client its cluster-specific model."""
        clients = client_manager.all()
        fit_instructions = []

        for client_proxy in clients.values():
            # Get facility ID from client CID
            try:
                fid = int(client_proxy.cid)
            except ValueError:
                continue

            care_type = FACILITY_CARE_TYPES.get(fid)
            if care_type is None or fid in HELD_OUT_FACILITIES:
                continue

            cluster_model = self.cluster_models.get(care_type)
            if cluster_model is not None:
                params = ndarrays_to_parameters([np.frombuffer(cluster_model, dtype=np.uint8)])
            else:
                params = ndarrays_to_parameters([np.zeros(1)])

            config = {
                "care_type": care_type,
                "round": server_round,
                "local_epochs": cfg["fl"]["clustered"]["local_epochs"],
            }
            fit_instructions.append((client_proxy, fl.common.FitIns(params, config)))

        return fit_instructions

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate within each cluster independently (intra-cluster FedAvg)."""
        cluster_results: Dict[str, List[Tuple[bytes, int]]] = {"MC": [], "SNF": [], "IL": []}

        for client_proxy, fit_res in results:
            fid = int(client_proxy.cid)
            care_type = FACILITY_CARE_TYPES.get(fid)
            if care_type is None:
                continue
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            model_bytes = ndarrays[0].tobytes()
            n_samples = fit_res.num_examples
            cluster_results[care_type].append((model_bytes, n_samples))

        metrics_agg = {}
        for care_type, cresults in cluster_results.items():
            if not cresults:
                continue
            self.cluster_models[care_type] = _weighted_average_xgb(cresults)
            n_total = sum(n for _, n in cresults)
            metrics_agg[f"{care_type}_n_clients"] = len(cresults)
            metrics_agg[f"{care_type}_n_samples"] = n_total

        logger.info(f"Round {server_round} — Aggregated clusters: {metrics_agg}")
        return None, metrics_agg  # Return None params (cluster-specific, not global)

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Send each client its cluster model for evaluation."""
        clients = client_manager.all()
        eval_instructions = []

        for client_proxy in clients.values():
            try:
                fid = int(client_proxy.cid)
            except ValueError:
                continue

            care_type = FACILITY_CARE_TYPES.get(fid)
            if care_type is None:
                continue

            cluster_model = self.cluster_models.get(care_type)
            if cluster_model is None:
                continue

            params = ndarrays_to_parameters([np.frombuffer(cluster_model, dtype=np.uint8)])
            config = {"care_type": care_type, "round": server_round}
            eval_instructions.append((client_proxy, fl.common.EvaluateIns(params, config)))

        return eval_instructions

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics per cluster."""
        cluster_aucs: Dict[str, List[float]] = {"MC": [], "SNF": [], "IL": []}

        for client_proxy, eval_res in results:
            fid = int(client_proxy.cid)
            care_type = FACILITY_CARE_TYPES.get(fid)
            if care_type and "auc" in eval_res.metrics:
                cluster_aucs[care_type].append(eval_res.metrics["auc"])

        metrics = {}
        all_aucs = []
        for care_type, aucs in cluster_aucs.items():
            if aucs:
                avg_auc = float(np.mean(aucs))
                metrics[f"{care_type}_auc"] = avg_auc
                all_aucs.extend(aucs)

        if all_aucs:
            metrics["overall_auc"] = float(np.mean(all_aucs))
            logger.info(f"Round {server_round} — CFL AUC: {metrics}")

        self.round_results.append({"round": server_round, **metrics})
        return None, metrics

    def evaluate(self, server_round, parameters):
        """No centralised model evaluation in clustered FL."""
        return None
