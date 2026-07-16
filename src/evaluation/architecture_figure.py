"""
FedAcuity -- Figure 1: System Architecture Diagram

Programmatic (matplotlib) three-layer architecture figure so Fig 1 is
reproducible and consistent with the rest of the figures (no manual draw.io).

Layer 1  Facility Edge      -- 10 facilities, care-type clusters, XGBoost local,
                               held-out 6 (SNF) + 9 (IL); raw PHI never leaves.
Layer 2  Aggregation Server -- intra-cluster FedAvg -> 3 care-type global models
                               (Clustered FL); FedAvg / FedProx baselines.
Layer 3  Evaluation & XAI   -- held-out eval, DP sweep, SHAP XAI Audit Scorecard.

Usage:
    python -m src.evaluation.architecture_figure
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from src.config import cfg

FIG_DIR = Path(cfg["paths"]["results"]["figures"])
FIG_DIR.mkdir(parents=True, exist_ok=True)

# palette (consistent with slide deck)
NAVY = "#0B1D3A"; TEAL = "#0A7EA4"; BLUE = "#1B466E"; GREEN = "#1B884B"
GREY = "#5A6A7E"; PALE = "#EAF2F6"; RED = "#C0392B"; INK = "#12232E"
MC_C, SNF_C, IL_C = "#0A7EA4", "#1B466E", "#1B884B"

CARE = {"MC": ([0, 1, 2], MC_C), "SNF": ([3, 4, 5, 6], SNF_C), "IL": ([7, 8, 9], IL_C)}
HELD_OUT = {6, 9}


def _box(ax, x, y, w, h, fc, ec, lw=1.2, rounded=0.02, alpha=1.0, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounded}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, linestyle=ls,
                       mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def _text(ax, x, y, s, size=9, color=INK, weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, style=style, zorder=5)


def _arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, zorder=4, shrinkA=2, shrinkB=2))


def build():
    fig, ax = plt.subplots(figsize=(11, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    _text(ax, 50, 97.5, "FedAcuity System Architecture", size=16, color=NAVY, weight="bold")
    _text(ax, 50, 94, "HIPAA-compliant by design: only model parameters cross the facility boundary",
          size=9.5, color=GREY, style="italic")

    # ── band labels (left rail) ──────────────────────────────────────────────
    for (yc, lbl) in [(78, "LAYER 1\nFacility Edge"), (49, "LAYER 2\nAggregation Server"),
                      (19, "LAYER 3\nEvaluation & XAI")]:
        _box(ax, 1.5, yc - 7, 11, 14, PALE, TEAL, lw=1.2, rounded=0.06)
        _text(ax, 7, yc, lbl, size=9, color=TEAL, weight="bold")

    # ── LAYER 1: facilities grouped by care type ─────────────────────────────
    # HIPAA boundary
    hb = Rectangle((15, 66.5), 83, 23.5, fill=False, edgecolor=RED, lw=1.4, ls=(0, (6, 4)), zorder=2)
    ax.add_patch(hb)
    _text(ax, 96.6, 88.4, "HIPAA boundary", size=8, color=RED, ha="right", style="italic")

    group_x = {"MC": 17, "SNF": 42, "IL": 76}
    fac_w, fac_h, gap = 6.6, 8.5, 1.4
    cluster_centers = {}
    for care, (fids, col) in CARE.items():
        gx = group_x[care]
        total_w = len(fids) * fac_w + (len(fids) - 1) * gap
        # cluster backing panel
        _box(ax, gx - 2, 70, total_w + 4, 15.5, "#FFFFFF", col, lw=1.3, rounded=0.03)
        _text(ax, gx + total_w / 2, 87.2, f"{care} cluster", size=9.5, color=col, weight="bold")
        cluster_centers[care] = gx + total_w / 2
        for i, fid in enumerate(fids):
            fx = gx + i * (fac_w + gap)
            held = fid in HELD_OUT
            _box(ax, fx, 72, fac_w, fac_h, col if not held else "#FFFFFF",
                 RED if held else col, lw=1.6 if held else 1.0, ls="--" if held else "-",
                 alpha=0.92 if not held else 1.0, rounded=0.06)
            _text(ax, fx + fac_w / 2, 72 + fac_h * 0.62, f"F{fid}", size=10,
                  color="#FFFFFF" if not held else RED, weight="bold")
            _text(ax, fx + fac_w / 2, 72 + fac_h * 0.26, "XGBoost", size=6.6,
                  color="#EAF2F6" if not held else GREY)
            if held:
                _text(ax, fx + fac_w / 2, 70.2, "held-out", size=6.2, color=RED, style="italic")

    _text(ax, 56, 68.3, "Each facility trains XGBoost locally on resident records  •  raw PHI never leaves the site",
          size=8.5, color=GREY, style="italic")

    # ── arrows L1 -> L2 (weights up) ─────────────────────────────────────────
    for care in CARE:
        _arrow(ax, cluster_centers[care], 70, cluster_centers[care], 58, color=TEAL, lw=1.8)
    _text(ax, 50, 63.6, "serialised model bytes only (~50–200 KB / round)",
          size=8.5, color=TEAL, weight="bold")

    # ── LAYER 2: aggregation ────────────────────────────────────────────────
    _box(ax, 15, 40, 83, 17.5, "#F5F9FB", BLUE, lw=1.3, rounded=0.02)
    agg_w = 22
    agg_x = {"MC": 18, "SNF": 42.5, "IL": 68}
    for care, (fids, col) in CARE.items():
        ax_x = agg_x[care]
        _box(ax, ax_x, 44, agg_w, 10.5, col, col, lw=1.2, alpha=0.16, rounded=0.05)
        _text(ax, ax_x + agg_w / 2, 51.5, f"{care} aggregator", size=9, color=col, weight="bold")
        _text(ax, ax_x + agg_w / 2, 48, "intra-cluster FedAvg", size=7.6, color=INK)
        _text(ax, ax_x + agg_w / 2, 45.6, "→ care-type global model", size=7.2, color=GREY, style="italic")
        _arrow(ax, cluster_centers[care], 58, ax_x + agg_w / 2, 55, color=BLUE, lw=1.4)
    _text(ax, 56.5, 42.0,
          "Clustered FL (C1): 3 independent global models  •  baselines: FedAvg / FedProx (≡ FedAvg for XGBoost) / Centralised / Local",
          size=8, color=GREY, ha="center", style="italic")

    # ── arrow L2 -> L3 ───────────────────────────────────────────────────────
    _arrow(ax, 56, 40, 56, 32, color=BLUE, lw=1.8)
    _text(ax, 56, 36.2, "trained models", size=8.5, color=BLUE, weight="bold")

    # ── LAYER 3: evaluation & XAI ────────────────────────────────────────────
    _box(ax, 15, 6, 83, 24, "#F5F9FB", GREEN, lw=1.3, rounded=0.02)
    cards = [
        (18, "Held-out evaluation", ["facilities 6 (SNF) + 9 (IL)", "AUC-ROC, F1, Mann-Whitney U"]),
        (43, "Differential privacy", ["Opacus DP-SGD on StaffingNN", "ε-sweep {1,2,5,10,∞}, 5 seeds"]),
        (68, "XAI Audit Scorecard (C3)", ["SHAP TreeExplainer", "D1 Fidelity · D2 Stability", "D3 Fairness · D4 Plausibility"]),
    ]
    for cx, title, lines in cards:
        _box(ax, cx, 9, 22, 17, "#FFFFFF", GREEN, lw=1.2, rounded=0.04)
        _text(ax, cx + 11, 23, title, size=9, color=GREEN, weight="bold")
        for j, ln in enumerate(lines):
            _text(ax, cx + 11, 19.6 - j * 3.0, ln, size=7.4, color=INK)
    _text(ax, 56.5, 7.4, "Outputs → results/figures (Fig 2–6) · results/tables (metrics, scorecard)",
          size=8, color=GREY, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    png = FIG_DIR / "fig1_architecture.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(str(png).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved: {png} (+ .pdf)")


if __name__ == "__main__":
    build()
