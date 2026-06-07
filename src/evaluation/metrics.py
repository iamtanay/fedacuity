"""
FedAcuity — M5.1 Evaluation Metrics
Bootstrap CI and Mann-Whitney U test across FL strategies.

Usage:
    python -m src.evaluation.metrics
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOGS_DIR   = Path(cfg["paths"]["results"]["logs"])
TABLES_DIR = Path(cfg["paths"]["results"]["tables"])
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SEED = cfg["project"]["seed"]
CI   = cfg["evaluation"]["stat_tests"]["bootstrap_ci"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_logs() -> pd.DataFrame:
    """Merge every results_*.json from the logs directory into one DataFrame."""
    records = []
    for p in sorted(LOGS_DIR.glob("results_*.json")):
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)
    if not records:
        raise FileNotFoundError(f"No results_*.json found in {LOGS_DIR}. Run simulation first.")
    return pd.DataFrame(records)


def _strategy_rows(df: pd.DataFrame, strategy_substr: str) -> pd.DataFrame:
    mask = df["strategy"].astype(str).str.contains(strategy_substr, case=False, na=False)
    return df[mask].dropna(subset=["overall_auc"]).sort_values("round")


# ── Statistics ────────────────────────────────────────────────────────────────

def bootstrap_ci(values: list, n_bootstrap: int = 2000) -> tuple[float, float, float]:
    """Return (mean, lo, hi) bootstrap CI at configured CI level."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    alpha = (1 - CI) / 2
    return float(arr[-1]), float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


def mann_whitney_u(a: list, b: list) -> dict:
    """Mann-Whitney U (non-parametric, two-sided)."""
    stat, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "U_statistic": round(float(stat), 2),
        "p_value":     round(float(pval), 6),
        "significant": bool(pval < 0.05),
    }


# ── Main computation ──────────────────────────────────────────────────────────

STRATEGY_KEYS = {
    "local":        "local",
    "centralised":  "centralised",
    "fedavg":       "fedavg",
    "fedprox":      "fedprox",
    "clustered_fl": "clustered",
}


def compute_summary(df: pd.DataFrame) -> dict:
    summary: dict = {}

    for label, search in STRATEGY_KEYS.items():
        rows = _strategy_rows(df, search)
        if rows.empty:
            logger.warning(f"No rows for strategy '{label}' — skipping.")
            summary[label] = None
            continue

        aucs = rows["overall_auc"].tolist()
        final_auc, ci_lo, ci_hi = bootstrap_ci(aucs)

        per_care: dict = {}
        for ct in ["MC", "SNF", "IL"]:
            col = f"{ct}_auc"
            if col in rows.columns:
                ct_aucs = rows[col].dropna().tolist()
                if ct_aucs:
                    per_care[ct] = round(ct_aucs[-1], 4)

        summary[label] = {
            "final_auc":      round(final_auc, 4),
            "ci_lo":          round(ci_lo, 4),
            "ci_hi":          round(ci_hi, 4),
            "n_log_points":   len(aucs),
            "per_care_type":  per_care,
        }

    # Mann-Whitney U: CFL vs FedAvg
    cfl_aucs    = _strategy_rows(df, "clustered")["overall_auc"].tolist()
    fedavg_aucs = _strategy_rows(df, "fedavg")["overall_auc"].tolist()
    if cfl_aucs and fedavg_aucs:
        summary["mann_whitney_cfl_vs_fedavg"] = mann_whitney_u(cfl_aucs, fedavg_aucs)
        logger.info(f"Mann-Whitney U (CFL vs FedAvg): {summary['mann_whitney_cfl_vs_fedavg']}")

    return summary


def main():
    df = load_all_logs()
    logger.info(f"Loaded {len(df)} log records from {LOGS_DIR}")

    summary = compute_summary(df)

    out_path = TABLES_DIR / "fl_metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty print
    print("\n-- FL Metrics Summary (Bootstrap 95% CI) -----------------------------------")
    for label in STRATEGY_KEYS:
        v = summary.get(label)
        if v is None:
            print(f"  {label:20s}: NO DATA")
        else:
            ct_str = "  ".join(f"{k}:{val:.4f}" for k, val in v["per_care_type"].items())
            print(f"  {label:20s}: AUC={v['final_auc']:.4f}  "
                  f"CI=[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]  "
                  f"({v['n_log_points']} pts)"
                  + (f"  [{ct_str}]" if ct_str else ""))
    mw = summary.get("mann_whitney_cfl_vs_fedavg")
    if mw:
        sig = "SIGNIFICANT" if mw["significant"] else "not significant"
        print(f"\n  Mann-Whitney U (CFL vs FedAvg): U={mw['U_statistic']}, "
              f"p={mw['p_value']:.4f} -> {sig}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
