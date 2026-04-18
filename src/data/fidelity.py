"""
FedAcuity — M1.4 Fidelity Validation (Contribution C2)
Validates synthetic LTC data against MIMIC-IV via:
  - KS-test (feature-wise distributional similarity)
  - Frobenius norm (correlation matrix similarity)
  - TSTR experiment (train-on-synthetic, test-on-real)

Usage:
    python src/data/fidelity.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from src.data.schema import CONTINUOUS_FEATURES, LABEL_COL
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYNTHETIC_DIR  = Path(cfg["paths"]["data"]["synthetic"])
MIMIC_DIR      = Path(cfg["paths"]["data"]["mimic_iv"])
RESULTS_DIR    = Path(cfg["paths"]["results"]["tables"])
FIGURES_DIR    = Path(cfg["paths"]["results"]["figures"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

KS_ALPHA            = cfg["data"]["fidelity"]["ks_alpha"]
TSTR_GAP_THRESHOLD  = cfg["data"]["fidelity"]["tstr_gap_threshold"]


# ── 1. KS-Test ────────────────────────────────────────────────────────────────

def run_ks_tests(synthetic_df: pd.DataFrame, mimic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-sample KS-test for each continuous feature.
    Returns DataFrame with statistic, p-value, and pass/fail flag.
    """
    results = []
    common_features = [f for f in CONTINUOUS_FEATURES if f in synthetic_df.columns and f in mimic_df.columns]

    for feat in common_features:
        stat, pval = stats.ks_2samp(
            synthetic_df[feat].dropna(),
            mimic_df[feat].dropna(),
        )
        results.append({
            "feature": feat,
            "ks_statistic": round(stat, 4),
            "p_value": round(pval, 4),
            "passes_alpha": pval >= KS_ALPHA,
        })

    df = pd.DataFrame(results)
    n_pass = df["passes_alpha"].sum()
    logger.info(f"KS-test: {n_pass}/{len(df)} features pass α={KS_ALPHA}")
    return df


# ── 2. Frobenius Norm ─────────────────────────────────────────────────────────

def frobenius_norm_comparison(
    synthetic_df: pd.DataFrame,
    mimic_df: pd.DataFrame,
    random_baseline: bool = True,
) -> dict:
    """
    Compare correlation matrices between synthetic and MIMIC-IV.
    Also computes a shuffled-data baseline for context.
    """
    common = [f for f in CONTINUOUS_FEATURES if f in synthetic_df.columns and f in mimic_df.columns]

    R_synth = synthetic_df[common].corr().values
    R_mimic = mimic_df[common].corr().values

    frobenius = np.linalg.norm(R_synth - R_mimic, ord="fro")

    result = {
        "frobenius_norm": round(frobenius, 4),
        "n_features": len(common),
    }

    if random_baseline:
        rng = np.random.default_rng(cfg["project"]["seed"])
        shuffled = synthetic_df[common].copy()
        for col in shuffled.columns:
            shuffled[col] = rng.permutation(shuffled[col].values)
        R_shuffled = shuffled.corr().values
        baseline = np.linalg.norm(R_shuffled - R_mimic, ord="fro")
        result["frobenius_norm_baseline"] = round(baseline, 4)
        result["improvement_over_baseline"] = round(baseline - frobenius, 4)

    logger.info(f"Frobenius norm (synthetic vs MIMIC-IV): {result['frobenius_norm']:.4f}")
    if random_baseline:
        logger.info(f"Frobenius norm (random baseline):       {result['frobenius_norm_baseline']:.4f}")

    return result


# ── 3. TSTR Experiment ────────────────────────────────────────────────────────

def tstr_experiment(
    synthetic_df: pd.DataFrame,
    mimic_df: pd.DataFrame,
    mimic_label_col: str = LABEL_COL,
    seed: int = None,
) -> dict:
    """
    Train on Synthetic, Test on Real (TSTR).
    Also runs Train on Real, Test on Real (TRTR) as oracle baseline.
    Target: TSTR AUC gap < 8% vs TRTR.
    """
    seed = seed or cfg["project"]["seed"]
    common = [f for f in CONTINUOUS_FEATURES if f in synthetic_df.columns and f in mimic_df.columns]

    # Prepare synthetic training set
    X_train_synth = synthetic_df[common].fillna(0)
    y_train_synth = synthetic_df[LABEL_COL]

    # Prepare MIMIC-IV test set (80/20 split)
    mimic_shuffled = mimic_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(0.8 * len(mimic_shuffled))
    X_real_train = mimic_shuffled[common][:split].fillna(0)
    y_real_train = mimic_shuffled[mimic_label_col][:split]
    X_real_test  = mimic_shuffled[common][split:].fillna(0)
    y_real_test  = mimic_shuffled[mimic_label_col][split:]

    xgb_params = dict(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric="auc", random_state=seed, verbosity=0,
    )

    # TSTR: train on synthetic, test on real
    model_tstr = xgb.XGBClassifier(**xgb_params)
    model_tstr.fit(X_train_synth, y_train_synth)
    tstr_auc = roc_auc_score(y_real_test, model_tstr.predict_proba(X_real_test)[:, 1])

    # TRTR: train on real, test on real (oracle)
    model_trtr = xgb.XGBClassifier(**xgb_params)
    model_trtr.fit(X_real_train, y_real_train)
    trtr_auc = roc_auc_score(y_real_test, model_trtr.predict_proba(X_real_test)[:, 1])

    gap = abs(trtr_auc - tstr_auc)
    passes = gap <= TSTR_GAP_THRESHOLD

    result = {
        "tstr_auc": round(tstr_auc, 4),
        "trtr_auc": round(trtr_auc, 4),
        "gap": round(gap, 4),
        "target_gap": TSTR_GAP_THRESHOLD,
        "passes": passes,
    }
    logger.info(f"TSTR AUC: {tstr_auc:.4f} | TRTR AUC: {trtr_auc:.4f} | Gap: {gap:.4f} | "
                f"{'PASS ✓' if passes else 'FAIL ✗ — retune CTGAN'}")
    return result


# ── 4. Visualisation ──────────────────────────────────────────────────────────

def plot_ks_distributions(
    synthetic_df: pd.DataFrame,
    mimic_df: pd.DataFrame,
    ks_results: pd.DataFrame,
    top_n: int = 6,
    save_path: Path = None,
):
    """Paper Figure 2: side-by-side distribution plots with KS p-values annotated."""
    features = ks_results.head(top_n)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(synthetic_df[feat].dropna(), bins=30, alpha=0.6, label="Synthetic", color="#1f77b4", density=True)
        ax.hist(mimic_df[feat].dropna(),     bins=30, alpha=0.6, label="MIMIC-IV",  color="#ff7f0e", density=True)

        row = ks_results[ks_results["feature"] == feat].iloc[0]
        pval_str = f"p={row['p_value']:.3f}" if row["p_value"] >= 0.001 else "p<0.001"
        ax.set_title(f"{feat}\nKS stat={row['ks_statistic']:.3f}, {pval_str}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Synthetic vs MIMIC-IV: Feature Distribution Comparison (Figure 2)", fontsize=11)
    plt.tight_layout()

    save_path = save_path or FIGURES_DIR / "fig2_fidelity_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(str(save_path).replace(".png", ".pdf"), bbox_inches="tight")
    logger.info(f"Figure 2 saved: {save_path}")
    plt.close()


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_fidelity_validation():
    """Full M1.4 pipeline."""
    # Load synthetic data
    parquet_path = SYNTHETIC_DIR / "all_facilities.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Run generator.py first. Missing: {parquet_path}")
    synthetic_df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded synthetic data: {len(synthetic_df)} rows")

    # Load MIMIC-IV (requires credentialed access)
    mimic_path = MIMIC_DIR / "mimic_elderly_subset.parquet"
    if not mimic_path.exists():
        logger.warning(
            f"MIMIC-IV subset not found at {mimic_path}.\n"
            "CONTINGENCY: Using MIMIC-IV demo dataset or synthetic-vs-synthetic fidelity check.\n"
            "Apply for access at https://physionet.org/"
        )
        # Fallback: use a random 20% holdout of synthetic data as a proxy
        mimic_df = synthetic_df.sample(frac=0.2, random_state=cfg["project"]["seed"])
        logger.warning("Using synthetic holdout as MIMIC-IV proxy (for pipeline testing only).")
    else:
        mimic_df = pd.read_parquet(mimic_path)
        logger.info(f"Loaded MIMIC-IV elderly subset: {len(mimic_df)} rows")

    # 1. KS-tests
    ks_results = run_ks_tests(synthetic_df, mimic_df)
    ks_results.to_csv(RESULTS_DIR / "fidelity_ks_test.csv", index=False)

    # 2. Frobenius norm
    frobenius_results = frobenius_norm_comparison(synthetic_df, mimic_df)
    with open(RESULTS_DIR / "fidelity_frobenius.json", "w") as f:
        json.dump(frobenius_results, f, indent=2)

    # 3. TSTR experiment
    tstr_results = tstr_experiment(synthetic_df, mimic_df)
    with open(RESULTS_DIR / "fidelity_tstr.json", "w") as f:
        json.dump(tstr_results, f, indent=2)

    # 4. Visualisation
    plot_ks_distributions(synthetic_df, mimic_df, ks_results)

    # 5. Summary
    print("\n── Fidelity Validation Summary (C2) ─────────────────────")
    print(f"  KS-test: {ks_results['passes_alpha'].sum()}/{len(ks_results)} features pass α={KS_ALPHA}")
    print(f"  Frobenius norm (synthetic vs MIMIC-IV): {frobenius_results['frobenius_norm']:.4f}")
    if "frobenius_norm_baseline" in frobenius_results:
        print(f"  Frobenius norm (random baseline):       {frobenius_results['frobenius_norm_baseline']:.4f}")
    print(f"  TSTR AUC: {tstr_results['tstr_auc']:.4f} | TRTR AUC: {tstr_results['trtr_auc']:.4f} | "
          f"Gap: {tstr_results['gap']:.4f} | Target: <{TSTR_GAP_THRESHOLD:.0%}")
    status = "PASS ✓ — C2 Contribution validated" if tstr_results["passes"] else \
             "FAIL ✗ — Retune CTGAN (increase epochs or try GaussianCopula)"
    print(f"  TSTR result: {status}")


if __name__ == "__main__":
    run_fidelity_validation()
