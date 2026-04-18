"""
FedAcuity — M1.1 Data Schema
Defines the synthetic LTC data schema, care-type non-IID distributions,
and the binary staffing-acuity mismatch label definition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

# ── Care Types ────────────────────────────────────────────────────────────────

CARE_TYPES = ["MC", "SNF", "IL"]

CARE_TYPE_LABELS = {
    "MC":  "Memory Care",
    "SNF": "Skilled Nursing Facility",
    "IL":  "Independent Living",
}

# ── Feature Definitions ───────────────────────────────────────────────────────

@dataclass
class FeatureSpec:
    name: str
    dtype: str                  # "float", "int", "category"
    min_val: float = 0.0
    max_val: float = 1.0
    description: str = ""


FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("adl_eating",            "float", 0, 6,   "ADL self-performance score: eating (MDS 3.0)"),
    FeatureSpec("adl_mobility",          "float", 0, 6,   "ADL self-performance score: bed mobility (MDS 3.0)"),
    FeatureSpec("adl_toileting",         "float", 0, 6,   "ADL self-performance score: toileting (MDS 3.0)"),
    FeatureSpec("adl_cognition",         "float", 0, 6,   "Cognitive function scale (MDS 3.0 BIMS proxy)"),
    FeatureSpec("mds_adl_summary",       "float", 0, 28,  "MDS 3.0 ADL summary score (sum of 4 ADL items × scale)"),
    FeatureSpec("rug_category",          "int",   1, 8,   "RUG-IV Resource Utilization Group (1=lowest, 8=highest)"),
    FeatureSpec("nursing_hours_rn",      "float", 0, 12,  "Registered Nurse hours available per 24h per resident"),
    FeatureSpec("nursing_hours_lpn",     "float", 0, 12,  "Licensed Practical Nurse hours per 24h per resident"),
    FeatureSpec("nursing_hours_cna",     "float", 0, 12,  "Certified Nursing Assistant hours per 24h per resident"),
    FeatureSpec("medication_count",      "int",   0, 20,  "Average medications per resident (Landi INCUR: ~8.5)"),
    FeatureSpec("fall_risk_score",       "float", 0, 10,  "Standardised fall risk score (0=low, 10=extreme)"),
    FeatureSpec("pain_assessment_score", "float", 0, 10,  "Pain assessment score (MDS Section J proxy)"),
    FeatureSpec("resident_census",       "int",   10, 120,"Daily resident count"),
    FeatureSpec("incident_count",        "int",   0, 15,  "Daily incident log count (falls, behavioural, medical)"),
]

FEATURE_NAMES: List[str] = [f.name for f in FEATURE_SPECS]
CONTINUOUS_FEATURES: List[str] = [f.name for f in FEATURE_SPECS if f.dtype in ("float", "int")]
LABEL_COL = "staffing_mismatch"

# ── Non-IID Distribution Specifications ──────────────────────────────────────
# Each care type has deliberately engineered distributional differences.
# These are sampled from when seeding CTGAN or post-processing generated data.

NON_IID_SPEC: Dict[str, Dict] = {
    "MC": {
        # Memory Care: high cognitive decline, high acuity, heavy nursing load
        "adl_cognition":         {"mean": 4.5, "std": 1.0, "clip": (0, 6)},
        "adl_mobility":          {"mean": 4.0, "std": 1.2, "clip": (0, 6)},
        "adl_eating":            {"mean": 3.5, "std": 1.3, "clip": (0, 6)},
        "medication_count":      {"mean": 11,  "std": 2.5, "clip": (5, 20)},
        "nursing_hours_rn":      {"mean": 2.5, "std": 0.5, "clip": (0.5, 8)},
        "nursing_hours_cna":     {"mean": 4.0, "std": 0.8, "clip": (1, 10)},
        "mismatch_rate":          0.40,  # High mismatch — understaffing crisis
        "rug_category_mode":     6,
    },
    "SNF": {
        # Skilled Nursing: post-acute rehab, high acuity variance, frequent discharge
        "adl_cognition":         {"mean": 2.5, "std": 1.5, "clip": (0, 6)},
        "adl_mobility":          {"mean": 3.0, "std": 1.5, "clip": (0, 6)},
        "medication_count":      {"mean": 9,   "std": 3.0, "clip": (3, 20)},
        "nursing_hours_rn":      {"mean": 3.0, "std": 0.7, "clip": (1, 10)},
        "nursing_hours_cna":     {"mean": 3.5, "std": 0.9, "clip": (1, 10)},
        "mismatch_rate":          0.28,  # Moderate — high variance
        "rug_category_mode":     5,
    },
    "IL": {
        # Independent Living: low acuity, wellness focus, stable patterns
        "adl_cognition":         {"mean": 1.0, "std": 0.8, "clip": (0, 6)},
        "adl_mobility":          {"mean": 1.2, "std": 0.9, "clip": (0, 6)},
        "medication_count":      {"mean": 5,   "std": 2.0, "clip": (0, 12)},
        "nursing_hours_rn":      {"mean": 1.0, "std": 0.3, "clip": (0.2, 4)},
        "nursing_hours_cna":     {"mean": 1.5, "std": 0.5, "clip": (0.3, 5)},
        "mismatch_rate":          0.12,  # Low — adequate coverage typical
        "rug_category_mode":     2,
    },
}

# ── Facility Assignments ──────────────────────────────────────────────────────

FACILITY_CARE_TYPES = {
    0: "MC",  1: "MC",  2: "MC",
    3: "SNF", 4: "SNF", 5: "SNF", 6: "SNF",
    7: "IL",  8: "IL",  9: "IL",
}

# Held-out for final evaluation (not seen during FL training)
HELD_OUT_FACILITIES = [8, 9]  # IL + IL

CLUSTER_ASSIGNMENTS = {
    "MC":  [0, 1, 2],
    "SNF": [3, 4, 5, 6],
    "IL":  [7, 8, 9],
}

# ── Label Definition ──────────────────────────────────────────────────────────

def compute_mismatch_label(
    adl_demand_score: np.ndarray,
    resident_census: np.ndarray,
    nursing_hours_available: np.ndarray,
    threshold: float = 0.85,
) -> np.ndarray:
    """
    Binary staffing-acuity mismatch label.

    Mismatch = 1 when:
        (ADL demand score × census) / total nursing hours > threshold

    Args:
        adl_demand_score:      weighted ADL dependency per resident (0–1 normalised)
        resident_census:       daily resident count
        nursing_hours_available: total nursing hours (RN + LPN + CNA)
        threshold:             demand/supply ratio above which mismatch is flagged

    Returns:
        Binary array: 1 = mismatch, 0 = adequate
    """
    demand = adl_demand_score * resident_census
    supply = np.maximum(nursing_hours_available, 0.1)  # avoid div-by-zero
    ratio = demand / supply
    return (ratio > threshold).astype(int)


def adl_demand_score(
    adl_eating: np.ndarray,
    adl_mobility: np.ndarray,
    adl_toileting: np.ndarray,
    adl_cognition: np.ndarray,
) -> np.ndarray:
    """
    Weighted composite ADL demand score (0–1 normalised).
    Weights informed by CMS staffing guidelines and LTC nursing literature.
    """
    weighted = (
        0.20 * adl_eating
        + 0.30 * adl_mobility
        + 0.25 * adl_toileting
        + 0.25 * adl_cognition
    )
    return weighted / 6.0   # normalise to [0, 1]


if __name__ == "__main__":
    print("Feature schema:")
    for f in FEATURE_SPECS:
        print(f"  {f.name:<28} {f.dtype:<10} [{f.min_val}, {f.max_val}]  — {f.description}")
    print(f"\nLabel: {LABEL_COL}")
    print(f"Facilities: {len(FACILITY_CARE_TYPES)}")
    print(f"Held-out: {HELD_OUT_FACILITIES}")
