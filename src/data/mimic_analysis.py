"""
FedAcuity -- MIMIC-IV Cohort Analysis (Contribution C2, secondary use)

fidelity.py validates feature-level distributional similarity (KS-test,
Frobenius norm, TSTR) on the 3 features mappable between MIMIC-IV hospital
EHR and the LTC MDS schema. That comparison is necessarily weak: a hospital
admission and an LTC facility-day are different units of observation.

This module uses MIMIC-IV differently -- as an external population-level
calibration anchor rather than a feature-fidelity benchmark. Within MIMIC-IV
itself (no synthetic data involved), it compares the cohort of elderly
patients discharged to post-acute/LTC care (the "staffing_mismatch=1" proxy
cohort) against those discharged elsewhere, and checks whether MIMIC-IV's
real-world discharge-to-post-acute rate is consistent with the mismatch
target rates the synthetic generator was calibrated to produce.

This is NOT fidelity validation. It is a face-validity / calibration check:
does the real-world base rate of "needs higher level of care" look like the
target rates encoded in the synthetic data design? It cannot prove the
synthetic data is realistic, but a gross mismatch would have been a red flag.

Usage:
    python -m src.data.mimic_analysis

Output:
    results/tables/mimic_cohort_analysis.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data.schema import NON_IID_SPEC
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIMIC_DIR   = Path(cfg["paths"]["data"]["mimic_iv"])
RESULTS_DIR = Path(cfg["paths"]["results"]["tables"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COHORT_FEATURES = ["medication_count", "rug_category", "resident_census"]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-variance Cohen's d for two independent samples."""
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def compare_cohorts(df: pd.DataFrame) -> dict:
    """LTC-bound (staffing_mismatch=1) vs non-LTC (staffing_mismatch=0) within MIMIC-IV."""
    ltc_bound = df[df["staffing_mismatch"] == 1]
    non_ltc   = df[df["staffing_mismatch"] == 0]

    logger.info(f"LTC-bound cohort: {len(ltc_bound):,} ({len(ltc_bound) / len(df):.1%})")
    logger.info(f"Non-LTC cohort:   {len(non_ltc):,} ({len(non_ltc) / len(df):.1%})")

    feature_tests = []
    for feat in COHORT_FEATURES:
        a = ltc_bound[feat].dropna().values.astype(float)
        b = non_ltc[feat].dropna().values.astype(float)
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        d = _cohens_d(a, b)
        feature_tests.append({
            "feature": feat,
            "ltc_bound_mean": round(float(a.mean()), 3),
            "ltc_bound_std": round(float(a.std()), 3),
            "non_ltc_mean": round(float(b.mean()), 3),
            "non_ltc_std": round(float(b.std()), 3),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6) if p_val >= 1e-6 else float(p_val),
            "cohens_d": round(d, 4),
            "direction_expected": bool(a.mean() > b.mean()),  # higher acuity proxy in LTC-bound cohort
        })
        logger.info(
            f"  {feat:<18} LTC-bound={a.mean():.2f} vs non-LTC={b.mean():.2f} "
            f"(d={d:.3f}, p={p_val:.2e})"
        )

    return {
        "n_total": int(len(df)),
        "n_ltc_bound": int(len(ltc_bound)),
        "n_non_ltc": int(len(non_ltc)),
        "ltc_bound_rate": round(float(len(ltc_bound) / len(df)), 4),
        "feature_comparisons": feature_tests,
    }


def calibration_check(mimic_rate: float) -> dict:
    """
    Compare MIMIC-IV's real discharge-to-post-acute rate against the
    mismatch target rates the synthetic generator was calibrated to hit.
    SNF is the closest real-world analogue to "discharged needing skilled
    nursing-level care", so it is the primary comparison point.
    """
    targets = {ct: spec["mismatch_rate"] for ct, spec in NON_IID_SPEC.items()}
    snf_target = targets["SNF"]
    gap_vs_snf = abs(mimic_rate - snf_target)

    closest_ct = min(targets, key=lambda ct: abs(targets[ct] - mimic_rate))

    result = {
        "mimic_discharge_to_postacute_rate": round(mimic_rate, 4),
        "synthetic_targets": {ct: round(r, 4) for ct, r in targets.items()},
        "closest_synthetic_care_type": closest_ct,
        "gap_vs_snf_target": round(gap_vs_snf, 4),
        "interpretation": (
            f"MIMIC-IV's real-world discharge-to-post-acute rate "
            f"({mimic_rate:.1%}) sits closest to the synthetic {closest_ct} "
            f"mismatch target ({targets[closest_ct]:.0%}), within "
            f"{gap_vs_snf:.1%} of the SNF target ({snf_target:.0%}). "
            "This is a face-validity signal, not proof of fidelity: the two "
            "rates measure related but distinct events (hospital discharge "
            "disposition vs. daily staffing-acuity ratio)."
        ),
    }
    logger.info(result["interpretation"])
    return result


def run_cohort_analysis():
    mimic_path = MIMIC_DIR / "mimic_elderly_subset.parquet"
    if not mimic_path.exists():
        raise FileNotFoundError(
            f"Missing {mimic_path}. Run src/data/mimic_preprocessor.py first."
        )
    df = pd.read_parquet(mimic_path)
    logger.info(f"Loaded MIMIC-IV elderly subset: {len(df):,} rows")

    cohort_results = compare_cohorts(df)
    calib_results = calibration_check(cohort_results["ltc_bound_rate"])

    output = {
        "cohort_comparison": cohort_results,
        "calibration_check": calib_results,
    }

    out_path = RESULTS_DIR / "mimic_cohort_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_path}")

    print("\n-- MIMIC-IV Cohort Analysis (C2, secondary use of MIMIC-IV) -----------")
    print(f"  LTC-bound cohort: {cohort_results['n_ltc_bound']:,} / {cohort_results['n_total']:,} "
          f"({cohort_results['ltc_bound_rate']:.1%})")
    print(f"  {'Feature':<18} {'LTC-bound':>10} {'Non-LTC':>10} {'Cohen d':>9} {'p-value':>10}")
    for ft in cohort_results["feature_comparisons"]:
        print(f"  {ft['feature']:<18} {ft['ltc_bound_mean']:>10.2f} {ft['non_ltc_mean']:>10.2f} "
              f"{ft['cohens_d']:>9.3f} {ft['p_value']:>10.2e}")
    print(f"\n  Calibration: {calib_results['interpretation']}")

    return output


if __name__ == "__main__":
    run_cohort_analysis()
