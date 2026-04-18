"""
FedAcuity — M3 Differential Privacy ε-Sweep
Trains PyTorch model with Opacus DP at ε ∈ {1, 2, 5, 10, ∞} and plots privacy-utility tradeoff.

Usage:
    python src/dp/epsilon_sweep.py
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from src.data.loaders import load_all_facilities
from src.data.schema import FEATURE_NAMES
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(cfg["paths"]["results"]["tables"])
FIGURES_DIR = Path(cfg["paths"]["results"]["figures"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DP_CFG  = cfg["dp"]
NN_CFG  = cfg["fl"]["pytorch_nn"]
SEED    = cfg["project"]["seed"]
torch.manual_seed(SEED)


# ── PyTorch Shallow NN ────────────────────────────────────────────────────────

class StaffingNN(nn.Module):
    """Shallow NN for federated DP training (XGBoost doesn't support Opacus)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ── Single ε Training ──────────────────────────────────────────────────────────

def train_with_epsilon(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    epsilon: Optional[float],
    n_epochs: int = 20,
) -> dict:
    """Train with Opacus DP at given ε (None = no DP, ε=∞)."""
    device = torch.device("cpu")
    input_dim = X_train.shape[1]

    # Datasets
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=NN_CFG["batch_size"], shuffle=True)

    model = StaffingNN(
        input_dim=input_dim,
        hidden_dims=NN_CFG["hidden_dims"],
        dropout=NN_CFG["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=NN_CFG["learning_rate"])
    criterion = nn.BCELoss()

    actual_epsilon = None

    if epsilon is not None:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=n_epochs,
            target_epsilon=epsilon,
            target_delta=DP_CFG["delta"],
            max_grad_norm=DP_CFG["max_grad_norm"],
        )
        logger.info(f"ε={epsilon}: noise_multiplier={optimizer.noise_multiplier:.4f}")

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        proba = model(X_test_t).numpy()
        auc = roc_auc_score(y_test, proba)

    if epsilon is not None:
        actual_epsilon = privacy_engine.get_epsilon(DP_CFG["delta"])

    result = {
        "target_epsilon": epsilon,
        "actual_epsilon": actual_epsilon,
        "auc": round(auc, 4),
    }
    label = f"ε={epsilon}" if epsilon else "No DP (ε=∞)"
    logger.info(f"{label} → AUC={auc:.4f}")
    return result


# ── ε Sweep ───────────────────────────────────────────────────────────────────

def run_epsilon_sweep():
    """M3.2: Run ε sweep across all target epsilon values."""
    facilities = load_all_facilities()

    # Pool training data
    X_parts, y_parts = [], []
    for fid, splits in facilities.items():
        X_train, y_train = splits["train"]
        X_parts.append(X_train.values)
        y_parts.append(y_train.values)

    X_train = np.vstack(X_parts).astype(np.float32)
    y_train = np.concatenate(y_parts).astype(np.float32)

    # Use one held-out facility for test
    from src.data.loaders import load_held_out
    held_out = load_held_out()
    fid_test = list(held_out.keys())[0]
    X_test_df, y_test = held_out[fid_test]["test"]
    X_test = X_test_df.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    results = []
    epsilon_values = DP_CFG["epsilon_values"]  # [1, 2, 5, 10, null]

    for eps in epsilon_values:
        result = train_with_epsilon(X_train, y_train, X_test, y_test, epsilon=eps)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "dp_epsilon_sweep.csv", index=False)
    logger.info(f"ε-sweep results saved.")

    # Plot Figure 5: Privacy-Utility Tradeoff
    _plot_privacy_utility(df)

    return df


def _plot_privacy_utility(df: pd.DataFrame):
    """Paper Figure 5: AUC vs ε with recommended threshold annotated."""
    fig, ax = plt.subplots(figsize=(8, 5))

    eps_labels = [str(row["target_epsilon"]) if row["target_epsilon"] else "∞" for _, row in df.iterrows()]
    eps_vals   = [row["target_epsilon"] if row["target_epsilon"] else 12 for _, row in df.iterrows()]
    aucs       = df["auc"].tolist()

    ax.plot(eps_vals, aucs, marker="o", color="#1f77b4", linewidth=2, markersize=8, label="AUC-ROC")

    # Annotate recommended threshold
    rec_eps = DP_CFG["recommended_epsilon"]
    ax.axvline(x=rec_eps, color="red", linestyle="--", linewidth=1.5, label=f"Recommended ε={rec_eps}")
    ax.axhline(y=max(aucs) * 0.95, color="grey", linestyle=":", linewidth=1, label="95% of max AUC")

    ax.set_xticks(eps_vals)
    ax.set_xticklabels(eps_labels, fontsize=11)
    ax.set_xlabel("Privacy Budget ε (lower = stronger privacy)", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff Curve (Figure 5)\nOpacus DP-SGD, δ=1e-5", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    save_path = FIGURES_DIR / "fig5_dp_privacy_utility.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(str(save_path).replace(".png", ".pdf"), bbox_inches="tight")
    logger.info(f"Figure 5 saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_epsilon_sweep()
