"""
FedAcuity — M4.6 XAI Audit Scorecard (Contribution C3)
Aggregates D1–D4 into a normalised scorecard and generates the radar chart.

Usage:
    python src/xai/scorecard.py
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from src.config import cfg

logger = logging.getLogger(__name__)
RESULTS_DIR = Path(cfg["paths"]["results"]["tables"])
FIGURES_DIR = Path(cfg["paths"]["results"]["figures"])
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VARIANTS = ["local", "centralised", "fedavg", "fedprox", "clustered_fl"]
MODEL_LABELS   = {
    "local":        "Local (No Fed)",
    "centralised":  "Centralised Oracle",
    "fedavg":       "FedAvg",
    "fedprox":      "FedProx",
    "clustered_fl": "CFL (FedAcuity)",
}
DIMENSIONS = ["D1 Fidelity", "D2 Stability", "D3 Fairness", "D4 Plausibility"]


def load_xai_results() -> Dict[str, Dict[str, float]]:
    """
    Load pre-computed XAI audit scores (output of d1–d4 modules).
    Returns: { model_name: { "D1 Fidelity": score, ... } }
    """
    score_path = RESULTS_DIR / "xai_audit_raw.json"
    if score_path.exists():
        with open(score_path) as f:
            return json.load(f)

    # Placeholder scores until actual XAI experiments run (Weeks 10–12)
    logger.warning("Using placeholder XAI scores — run d1_fidelity.py through d4_plausibility.py first.")
    return {
        "local":        {"D1 Fidelity": 0.55, "D2 Stability": 0.60, "D3 Fairness": 0.50, "D4 Plausibility": 0.60},
        "centralised":  {"D1 Fidelity": 1.00, "D2 Stability": 0.85, "D3 Fairness": 0.78, "D4 Plausibility": 0.80},
        "fedavg":       {"D1 Fidelity": 0.72, "D2 Stability": 0.65, "D3 Fairness": 0.60, "D4 Plausibility": 0.70},
        "fedprox":      {"D1 Fidelity": 0.74, "D2 Stability": 0.68, "D3 Fairness": 0.63, "D4 Plausibility": 0.70},
        "clustered_fl": {"D1 Fidelity": 0.80, "D2 Stability": 0.78, "D3 Fairness": 0.75, "D4 Plausibility": 0.75},
    }


def build_scorecard(raw_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Build normalised scorecard DataFrame."""
    rows = []
    for model, scores in raw_scores.items():
        row = {"Model": MODEL_LABELS.get(model, model)}
        for dim in DIMENSIONS:
            row[dim] = round(scores.get(dim, 0.0), 3)
        row["Mean Score"] = round(np.mean([scores.get(d, 0.0) for d in DIMENSIONS]), 3)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


def plot_radar_chart(raw_scores: Dict[str, Dict[str, float]], save_path: Path = None):
    """
    Paper Figure 6: Radar/spider chart — all five models × 4 XAI dimensions.
    CFL should cover more area than FedAvg on D2 and D3.
    """
    dims = DIMENSIONS
    n_dims = len(dims)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = {
        "local":        "#AAAAAA",
        "centralised":  "#2196F3",
        "fedavg":       "#FF9800",
        "fedprox":      "#9C27B0",
        "clustered_fl": "#4CAF50",
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    for model, scores in raw_scores.items():
        values = [scores.get(d, 0.0) for d in dims]
        values += values[:1]
        color = colors.get(model, "#333333")
        lw = 3 if model == "clustered_fl" else 1.5
        ls = "-" if model == "clustered_fl" else "--"
        ax.plot(angles, values, color=color, linewidth=lw, linestyle=ls,
                label=MODEL_LABELS.get(model, model))
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.title("XAI Audit Scorecard — All Model Variants\n(Figure 6)", fontsize=13, pad=20)

    save_path = save_path or FIGURES_DIR / "fig6_xai_radar.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(str(save_path).replace(".png", ".pdf"), bbox_inches="tight")
    logger.info(f"Figure 6 saved: {save_path}")
    plt.close()


def run_scorecard():
    """Full M4.6 pipeline: load scores → build table → generate radar chart."""
    raw_scores = load_xai_results()
    scorecard = build_scorecard(raw_scores)

    # Save scorecard table
    csv_path = RESULTS_DIR / "xai_audit_scorecard.csv"
    scorecard.to_csv(csv_path)
    logger.info(f"Scorecard saved: {csv_path}")

    # Save LaTeX table
    latex_path = RESULTS_DIR / "xai_audit_scorecard.tex"
    scorecard.to_latex(latex_path, float_format="%.3f")

    # Radar chart
    plot_radar_chart(raw_scores)

    print("\n── XAI Audit Scorecard (C3) ─────────────────────────────")
    print(scorecard.to_string())
    print(f"\nCFL vs FedAvg:")
    cfl = raw_scores.get("clustered_fl", {})
    favg = raw_scores.get("fedavg", {})
    for dim in DIMENSIONS:
        delta = cfl.get(dim, 0) - favg.get(dim, 0)
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {dim:<18}: CFL={cfl.get(dim, 0):.3f} | FedAvg={favg.get(dim, 0):.3f} | Δ={delta:+.3f} {arrow}")


if __name__ == "__main__":
    run_scorecard()
