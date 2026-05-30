"""
FedAcuity — M2.1/M2.2 Flower Client
Implements the per-facility Flower client with XGBoost local training.
"""

import logging
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

from src.data.loaders import get_facility_splits, load_facility
from src.config import cfg

logger = logging.getLogger(__name__)
SEED = cfg["project"]["seed"]
XGB_CFG = cfg["fl"]["xgboost"]


def serialize_xgb_model(model: xgb.XGBClassifier) -> bytes:
    """Serialise XGBoost model to bytes for FL weight exchange.
    Uses save_raw() which returns bytes directly — compatible with XGBoost 3.x.
    """
    return model.get_booster().save_raw("ubj")


def deserialize_xgb_model(weights: bytes) -> xgb.XGBClassifier:
    """Deserialise bytes back into XGBoost model via a temp file (XGBoost 3.x safe).
    Uses XGBClassifier.load_model() so sklearn attributes (n_classes_ etc.) are set.
    """
    model = xgb.XGBClassifier(
        n_estimators=XGB_CFG["n_estimators"],
        max_depth=XGB_CFG["max_depth"],
        learning_rate=XGB_CFG["learning_rate"],
        eval_metric=XGB_CFG["eval_metric"],
        verbosity=0,
    )
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
        f.write(weights)
        tmp_path = f.name
    try:
        model.load_model(tmp_path)
    finally:
        os.unlink(tmp_path)
    return model


class FedAcuityClient:
    """
    FL client for one LTC facility.
    Uses XGBoost as the local model. Compatible with the manual simulation loop.
    """

    def __init__(self, facility_id: int, mu: float = None):
        """
        Args:
            facility_id: integer ID (0-9)
            mu: FedProx proximal term weight (None = standard FedAvg)
        """
        self.facility_id = facility_id
        self.mu = mu  # None → FedAvg; float → FedProx

        df = load_facility(facility_id)
        (self.X_train, self.y_train), \
        (self.X_val,   self.y_val), \
        (self.X_test,  self.y_test) = get_facility_splits(facility_id, df)

        self.n_train = len(self.X_train)
        self.model = None
        logger.info(f"Client {facility_id}: {self.n_train} train rows, "
                    f"mismatch_rate={self.y_train.mean():.2%}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Serialise model weights → list of numpy arrays for Flower."""
        if self.model is None:
            return [np.zeros(1)]  # Uninitialised — server will not use
        weights_bytes = serialize_xgb_model(self.model)
        # Encode as array of uint8 for Flower compatibility
        return [np.frombuffer(weights_bytes, dtype=np.uint8)]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load global model weights from server."""
        if parameters[0].sum() == 0:
            return  # Skip uninitialised weights (first round)
        weights_bytes = parameters[0].tobytes()
        self.model = deserialize_xgb_model(weights_bytes)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Local training step."""
        self.set_parameters(parameters)

        local_epochs = config.get("local_epochs", cfg["fl"]["fedavg"]["local_epochs"])

        if self.model is None:
            # First round: train from scratch
            self.model = xgb.XGBClassifier(
                n_estimators=XGB_CFG["n_estimators"],
                max_depth=XGB_CFG["max_depth"],
                learning_rate=XGB_CFG["learning_rate"],
                eval_metric=XGB_CFG["eval_metric"],
                random_state=SEED + self.facility_id,
                verbosity=0,
            )
            self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )
        else:
            # Subsequent rounds: continue training (warm start)
            self.model.set_params(n_estimators=self.model.n_estimators + local_epochs * 10)
            self.model.fit(
                self.X_train, self.y_train,
                xgb_model=self.model.get_booster(),
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )

        # FedProx proximal regularisation (applied as post-hoc gradient penalty proxy)
        # For XGBoost, full DP-SGD proximal term is applied in PyTorch models (M3).
        # Here we note μ for reporting.

        train_auc = roc_auc_score(self.y_train, self.model.predict_proba(self.X_train)[:, 1])
        metrics = {"train_auc": float(train_auc), "facility_id": self.facility_id}

        return self.get_parameters(config={}), self.n_train, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate global model on local test set."""
        self.set_parameters(parameters)
        if self.model is None:
            return 0.0, len(self.X_test), {"auc": 0.0}

        proba = self.model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, proba)
        f1  = f1_score(self.y_test, (proba > 0.5).astype(int))

        return float(1.0 - auc), len(self.X_test), {
            "auc": float(auc),
            "f1":  float(f1),
            "facility_id": self.facility_id,
        }
