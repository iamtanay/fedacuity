"""
FedAcuity -- M5.2 Publication-Quality Figure Generation
Fig 3: FL convergence curves (AUC-ROC vs communication rounds)
Fig 4: Five-model bar chart (held-out AUC comparison)

All figures target IEEE JBHI submission quality:
  - 300 DPI, both PNG and PDF
  - CB-safe color palette
  - Clean minimal style, no chart junk
  - Correct data source: Fig 4 uses held-out AUC from fl_held_out_metrics.json

Usage:
    python -m src.evaluation.figures
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import cfg
from src.evaluation.metrics import load_all_logs, _strategy_rows, compute_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(cfg["paths"]["results"]["figures"])
TABLES_DIR  = Path(cfg["paths"]["results"]["tables"])
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = cfg["evaluation"]["figures"]["dpi"]

# ── Publication style ─────────────────────────────────────────────────────────
# Colorblind-safe palette (Wong 2011, widely used in Nature/IEEE)
CB_BLUE   = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN  = "#009E73"
CB_RED    = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_GREY   = "#999999"
CB_SKY    = "#56B4E9"
CB_YELLOW = "#F0E442"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#E5E5E5",
    "grid.linewidth":     0.8,
    "lines.linewidth":    2.0,
    "figure.dpi":         100,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

STRATEGY_STYLE = {
    "local":        ("Local (no FL)",          CB_GREY,   "--",  "s", 6),
    "centralised":  ("Centralised Oracle",      CB_GREEN,  "-.",  "D", 7),
    "fedavg":       ("FedAvg",                  CB_BLUE,   "-",   "o", 7),
    "fedprox":      ("FedProx (μ=0.1)",    CB_ORANGE, ":",   "^", 7),
    "clustered_fl": ("Clustered FL (ours)",     CB_RED,    "-",   "*", 9),
}

STRATEGY_SEARCH = {
    "local":        "local",
    "centralised":  "centralised",
    "fedavg":       "fedavg",
    "fedprox":      "fedprox",
    "clustered_fl": "clustered",
}


def _save(fig: plt.Figure, stem: str):
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    logger.info(f"Saved: {png.name} + {pdf.name}")
    plt.close(fig)


# ── Figure 3 -- Convergence Curves ────────────────────────────────────────────

def plot_convergence(df: pd.DataFrame):
    """
    FL convergence curves: AUC-ROC vs communication round.
    Training-client AUC for the 3 FL strategies.
    Non-federated baselines shown as horizontal reference lines.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.0))  # IEEE double-column width

    fl_keys = ["fedavg", "fedprox", "clustered_fl"]
    ref_keys = ["local", "centralised"]

    drawn = False
    for key in fl_keys:
        rows = _strategy_rows(df, STRATEGY_SEARCH[key])
        if rows.empty:
            logger.warning(f"No convergence data for {key}")
            continue
        label, color, ls, marker, ms = STRATEGY_STYLE[key]
        ax.plot(rows["round"], rows["overall_auc"],
                label=label, color=color, linestyle=ls,
                marker=marker, markersize=ms, markevery=2,
                markeredgewidth=0.8, markeredgecolor="white",
                linewidth=2.2, zorder=3)
        drawn = True

    for key in ref_keys:
        rows = _strategy_rows(df, STRATEGY_SEARCH[key])
        if rows.empty:
            continue
        final_auc = rows["overall_auc"].iloc[-1]
        label, color, ls, marker, ms = STRATEGY_STYLE[key]
        ax.axhline(y=final_auc, color=color, linestyle=ls, linewidth=1.6,
                   label=f"{label} ({final_auc:.4f})", zorder=2, alpha=0.85)

    ax.set_xlabel("Communication Rounds", fontsize=10, labelpad=6)
    ax.set_ylabel("AUC-ROC (training clients)", fontsize=10, labelpad=6)
    ax.set_title("FL Convergence Curves — FedAcuity", fontsize=11, fontweight="bold", pad=8)

    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC",
              handlelength=2.4, ncol=1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Tight Y range — don't over-truncate but focus on meaningful range
    if drawn:
        all_vals = []
        for key in fl_keys + ref_keys:
            rows = _strategy_rows(df, STRATEGY_SEARCH[key])
            if not rows.empty:
                all_vals.extend(rows["overall_auc"].tolist())
        if all_vals:
            lo = max(0.88, min(all_vals) - 0.005)
            hi = min(1.001, max(all_vals) + 0.008)
            ax.set_ylim(lo, hi)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    fig.tight_layout()
    _save(fig, "fig3_convergence")


# ── Figure 4 -- Five-Model Comparison Bar Chart ───────────────────────────────

def plot_model_comparison():
    """
    Five-model AUC-ROC comparison on held-out facilities (Table II data).
    Data source: fl_held_out_metrics.json (not training-client logs).
    Full Y-axis from 0 to 1 for honest visual comparison.
    """
    held_path = TABLES_DIR / "fl_held_out_metrics.json"
    if not held_path.exists():
        logger.warning("fl_held_out_metrics.json not found -- skipping Fig 4")
        return

    with open(held_path) as f:
        held = json.load(f)

    # Ordered for display -- add new baselines if present
    order = [
        ("il_local_baseline",       "IL Local\n(fac. 7 only)", CB_GREY),
        ("cross_facility_ensemble",  "Ensemble\n(no protocol)", CB_SKY),
        ("centralised",              "Centralised\nOracle",      CB_GREEN),
        ("fedavg",                   "FedAvg",                   CB_BLUE),
        ("fedprox",                  "FedProx\n(μ=0.1)",   CB_ORANGE),
        ("clustered_fl",             "Clustered FL\n(ours)",     CB_RED),
    ]
    # Fall back to old key name if new one not present
    fallback = {"il_local_baseline": "local", "cross_facility_ensemble": None}

    labels, aucs, colors = [], [], []
    for key, display, color in order:
        v = held.get(key) or held.get(fallback.get(key, ""))
        if v and "overall" in v and "auc_roc" in v["overall"]:
            labels.append(display)
            aucs.append(v["overall"]["auc_roc"])
            colors.append(color)

    if not labels:
        logger.warning("No held-out data to plot for Fig 4")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    x = np.arange(len(labels))
    bars = ax.bar(x, aucs, color=colors, width=0.62,
                  edgecolor="white", linewidth=1.0, zorder=3, alpha=0.92)

    # Value labels above bars
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{auc:.4f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="#333333")

    # Highlight CFL bar with bold edge
    cfl_idx = next((i for i, (k, _, _) in enumerate(order) if k == "clustered_fl"), None)
    if cfl_idx is not None and cfl_idx < len(bars):
        bars[cfl_idx].set_edgecolor(CB_RED)
        bars[cfl_idx].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC-ROC (held-out IL facilities 8 & 9)", fontsize=10, labelpad=6)
    ax.set_title("Five-Model Comparison on Held-Out Facilities", fontsize=11,
                 fontweight="bold", pad=8)

    # Full Y-axis: honest comparison, no truncation
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    # Annotation for CFL primary contribution
    if cfl_idx is not None and cfl_idx < len(aucs):
        ax.annotate("C1 Contribution",
                    xy=(cfl_idx, aucs[cfl_idx] + 0.01),
                    xytext=(cfl_idx + 0.6, aucs[cfl_idx] + 0.08),
                    fontsize=8.5, color=CB_RED, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=CB_RED, lw=1.2),
                    ha="left")

    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig4_model_comparison")


# ── Figure 2 -- Fidelity Distributions (re-export with publication style) ─────

def plot_fidelity_distributions():
    """
    Replot Fig 2 with publication styling.
    Requires: data/synthetic/all_facilities.parquet
    """
    from pathlib import Path as P
    parquet = P(cfg["paths"]["data"]["synthetic"]) / "all_facilities.parquet"
    if not parquet.exists():
        logger.warning("Synthetic parquet not found -- skipping Fig 2 replot")
        return

    import pandas as pd
    from src.data.schema import CONTINUOUS_FEATURES

    df = pd.read_parquet(parquet)
    # Use 20% holdout as proxy reference (MIMIC-IV access pending)
    rng = np.random.default_rng(cfg["project"]["seed"])
    idx = rng.choice(len(df), size=int(0.2 * len(df)), replace=False)
    reference_df = df.iloc[idx]
    synthetic_df = df.drop(df.index[idx])

    # Pick 6 most clinically meaningful features for the plot
    plot_features = [
        "adl_cognition", "nursing_hours_rn", "nursing_hours_lpn",
        "medication_count", "fall_risk_score", "mds_adl_summary",
    ]
    plot_features = [f for f in plot_features if f in df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    axes = axes.flatten()

    for i, feat in enumerate(plot_features):
        ax = axes[i]
        s_vals = synthetic_df[feat].dropna()
        r_vals = reference_df[feat].dropna()

        ax.hist(s_vals, bins=30, alpha=0.65, color=CB_BLUE, density=True,
                label="Synthetic (train)", zorder=3)
        ax.hist(r_vals, bins=30, alpha=0.65, color=CB_ORANGE, density=True,
                label="Synthetic proxy\n(MIMIC-IV pending)", zorder=2)

        from scipy import stats
        ks_stat, pval = stats.ks_2samp(s_vals, r_vals)
        pval_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"

        ax.set_title(f"{feat.replace('_', ' ')}\nKS={ks_stat:.3f}, {pval_str}",
                     fontsize=8.5, pad=4)
        ax.set_xlabel(feat.replace("_", " "), fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7.5)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(len(plot_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Synthetic Data Fidelity Validation (Figure 2)\n"
        "Note: MIMIC-IV access pending PhysioNet approval; "
        "20% synthetic holdout used as proxy.",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    _save(fig, "fig2_fidelity_distributions")


# ── Figure 5 -- DP Privacy-Utility Tradeoff (re-export with publication style) ─

def plot_privacy_utility():
    """Replot Fig 5 with publication styling and corrected recommended epsilon."""
    dp_path = TABLES_DIR / "dp_epsilon_sweep.csv"
    if not dp_path.exists():
        logger.warning("dp_epsilon_sweep.csv not found -- skipping Fig 5 replot")
        return

    df = pd.read_csv(dp_path)
    rec_eps = cfg["dp"]["recommended_epsilon"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    eps_numeric = [row["target_epsilon"] if not pd.isna(row["target_epsilon"])
                   else 12.5 for _, row in df.iterrows()]
    eps_labels  = [str(int(row["target_epsilon"])) if not pd.isna(row["target_epsilon"])
                   else "∞ (no DP)" for _, row in df.iterrows()]
    aucs = df["auc"].tolist()

    ax.plot(eps_numeric, aucs, color=CB_BLUE, linewidth=2.2,
            marker="o", markersize=8, markeredgecolor="white",
            markeredgewidth=1.2, zorder=4, label="AUC-ROC")

    # Annotate each point
    for ex, ey, el in zip(eps_numeric, aucs, eps_labels):
        ax.annotate(f"{ey:.4f}", (ex, ey),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#333333")

    # Recommended epsilon line
    ax.axvline(x=rec_eps, color=CB_RED, linestyle="--", linewidth=1.6,
               label=f"Recommended ε={rec_eps}", zorder=3)

    # 95% of max AUC reference
    max_auc = max(aucs)
    ax.axhline(y=max_auc * 0.95, color=CB_GREY, linestyle=":", linewidth=1.2,
               label="95% of max AUC", zorder=2)

    ax.set_xticks(eps_numeric)
    ax.set_xticklabels(eps_labels, fontsize=9)
    ax.set_xlabel("Privacy budget ε (smaller = stronger privacy)", fontsize=10, labelpad=6)
    ax.set_ylabel("AUC-ROC", fontsize=10, labelpad=6)
    ax.set_title("Privacy-Utility Tradeoff Curve (Figure 5)\n"
                 "Opacus DP-SGD · StaffingNN · δ=10⁻⁵",
                 fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=9, loc="upper left", frameon=True, edgecolor="#CCCCCC")
    ax.set_ylim(0.6, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _save(fig, "fig5_dp_privacy_utility")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_all_logs()
    summary = compute_summary(df)

    logger.info("Generating Fig 2 (fidelity distributions)...")
    plot_fidelity_distributions()

    logger.info("Generating Fig 3 (convergence curves)...")
    plot_convergence(df)

    logger.info("Generating Fig 4 (model comparison bar chart)...")
    plot_model_comparison()

    logger.info("Generating Fig 5 (DP privacy-utility)...")
    plot_privacy_utility()

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("  fig2_fidelity_distributions.{png,pdf}")
    print("  fig3_convergence.{png,pdf}")
    print("  fig4_model_comparison.{png,pdf}")
    print("  fig5_dp_privacy_utility.{png,pdf}")


if __name__ == "__main__":
    main()
