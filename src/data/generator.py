"""
FedAcuity — M1.2 CTGAN Synthetic Data Generator
Generates 10 simulated LTC facilities × 3 years of daily records.
Applies deliberate non-IID distributions per care type.

Usage:
    python src/data/generator.py
"""

import os
import json
import logging
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from src.data.schema import (
    FEATURE_SPECS, LABEL_COL, FACILITY_CARE_TYPES,
    NON_IID_SPEC, adl_demand_score, compute_mismatch_label,
)
from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = cfg["project"]["seed"]
SYNTHETIC_DIR = Path(cfg["paths"]["data"]["synthetic"])
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)


# ── Seed Dataset Builder ──────────────────────────────────────────────────────

def build_seed_dataset(care_type: str, n_rows: int = 500) -> pd.DataFrame:
    """
    Build a small schema-conformant seed DataFrame for CTGAN training.
    Uses care-type-specific means/stds from NON_IID_SPEC.
    """
    rng = np.random.default_rng(SEED)
    spec = NON_IID_SPEC[care_type]
    n_days = n_rows

    rows = {}
    for feat in FEATURE_SPECS:
        if feat.name in spec:
            s = spec[feat.name]
            vals = rng.normal(s["mean"], s["std"], n_days)
            vals = np.clip(vals, s["clip"][0], s["clip"][1])
        else:
            # Default to mid-range with moderate variance
            mid = (feat.min_val + feat.max_val) / 2
            spread = (feat.max_val - feat.min_val) * 0.15
            vals = rng.normal(mid, spread, n_days)
            vals = np.clip(vals, feat.min_val, feat.max_val)

        if feat.dtype == "int":
            vals = np.round(vals).astype(int)

        rows[feat.name] = vals

    df = pd.DataFrame(rows)
    df["care_type"] = care_type

    # Compute label
    demand = adl_demand_score(
        df["adl_eating"].values,
        df["adl_mobility"].values,
        df["adl_toileting"].values,
        df["adl_cognition"].values,
    )
    total_nursing = (
        df["nursing_hours_rn"].values
        + df["nursing_hours_lpn"].values
        + df["nursing_hours_cna"].values
    )
    threshold = _calibrate_threshold(demand, df["resident_census"].values, total_nursing,
                                     spec["mismatch_rate"])
    df[LABEL_COL] = compute_mismatch_label(demand, df["resident_census"].values, total_nursing, threshold)

    logger.info(f"Seed dataset [{care_type}]: {len(df)} rows, "
                f"mismatch_rate={df[LABEL_COL].mean():.2%}")
    return df


def _calibrate_threshold(demand, census, supply, target_rate, n_steps=100):
    """Binary-search threshold to hit the target mismatch rate."""
    lo, hi = 0.01, 5.0
    for _ in range(n_steps):
        mid = (lo + hi) / 2
        rate = compute_mismatch_label(demand, census, supply, mid).mean()
        if rate > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── CTGAN Training ────────────────────────────────────────────────────────────

def train_ctgan(care_type: str, seed_df: pd.DataFrame) -> CTGANSynthesizer:
    """Train one CTGANSynthesizer per care type."""
    logger.info(f"Training CTGAN for care type: {care_type}")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_df)
    # Override detected types for categorical columns
    metadata.update_column("care_type", sdtype="categorical")
    metadata.update_column(LABEL_COL,   sdtype="categorical")

    ctgan_cfg = cfg["data"]["ctgan"]
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=ctgan_cfg["epochs"],
        batch_size=ctgan_cfg["batch_size"],
        generator_dim=tuple(ctgan_cfg["generator_dim"]),
        discriminator_dim=tuple(ctgan_cfg["discriminator_dim"]),
        verbose=True,
    )
    synthesizer.fit(seed_df)
    return synthesizer


# ── Facility Generation ───────────────────────────────────────────────────────

def generate_facility(
    facility_id: int,
    care_type: str,
    synthesizer: CTGANSynthesizer,
    years: int = 3,
) -> pd.DataFrame:
    """Generate one facility: 3 years × daily records."""
    n_rows = years * 365

    logger.info(f"Generating facility {facility_id} [{care_type}]: {n_rows} rows")
    df = synthesizer.sample(num_rows=n_rows)

    # Enforce care type
    df["care_type"] = care_type
    df["facility_id"] = facility_id

    # Add date index
    start_date = date(2021, 1, 1)
    df["date"] = [start_date + timedelta(days=i) for i in range(len(df))]

    # Apply facility-level noise to break intra-cluster homogeneity
    rng = np.random.default_rng(SEED + facility_id)
    spec = NON_IID_SPEC[care_type]
    for feat_name in ["nursing_hours_rn", "nursing_hours_lpn", "nursing_hours_cna",
                      "medication_count", "resident_census"]:
        feat_spec = next(f for f in FEATURE_SPECS if f.name == feat_name)
        noise = rng.normal(0, 0.05 * (feat_spec.max_val - feat_spec.min_val), len(df))
        df[feat_name] = np.clip(df[feat_name] + noise, feat_spec.min_val, feat_spec.max_val)

    # Recompute label after noise
    demand = adl_demand_score(
        df["adl_eating"].values,
        df["adl_mobility"].values,
        df["adl_toileting"].values,
        df["adl_cognition"].values,
    )
    total_nursing = (
        df["nursing_hours_rn"].values
        + df["nursing_hours_lpn"].values
        + df["nursing_hours_cna"].values
    )
    threshold = _calibrate_threshold(demand, df["resident_census"].values, total_nursing,
                                     spec["mismatch_rate"])
    df[LABEL_COL] = compute_mismatch_label(demand, df["resident_census"].values, total_nursing, threshold)

    # Reorder columns
    ordered_cols = ["facility_id", "date", "care_type"] + \
                   [f.name for f in FEATURE_SPECS] + [LABEL_COL]
    df = df[[c for c in ordered_cols if c in df.columns]]

    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_generation():
    """Full M1.2 pipeline: train CTGANs → generate 10 facilities → save."""
    n_cfg = cfg["data"]
    years = n_cfg["years_per_facility"]

    # Step 1: Train one CTGAN per care type
    synthesizers = {}
    for care_type in ["MC", "SNF", "IL"]:
        seed_df = build_seed_dataset(care_type)
        synthesizers[care_type] = train_ctgan(care_type, seed_df)
        model_path = SYNTHETIC_DIR / f"ctgan_{care_type}.pkl"
        synthesizers[care_type].save(str(model_path))
        logger.info(f"Saved CTGAN model: {model_path}")

    # Step 2: Generate each facility
    all_dfs = []
    facility_metadata = {}
    for fid, care_type in FACILITY_CARE_TYPES.items():
        df = generate_facility(fid, care_type, synthesizers[care_type], years=years)

        # Save per-facility CSV
        csv_path = SYNTHETIC_DIR / f"facility_{fid:02d}_{care_type}.csv"
        df.to_csv(csv_path, index=False)

        facility_metadata[fid] = {
            "care_type": care_type,
            "n_rows": len(df),
            "mismatch_rate": float(df[LABEL_COL].mean()),
            "path": str(csv_path),
        }
        all_dfs.append(df)
        logger.info(f"Saved: {csv_path} — mismatch_rate={df[LABEL_COL].mean():.2%}")

    # Step 3: Save combined parquet
    combined = pd.concat(all_dfs, ignore_index=True)
    parquet_path = SYNTHETIC_DIR / "all_facilities.parquet"
    combined.to_parquet(parquet_path, index=False)
    logger.info(f"Combined dataset: {len(combined)} rows → {parquet_path}")

    # Step 4: Save metadata JSON
    meta_path = SYNTHETIC_DIR / "dataset_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(facility_metadata, f, indent=2)
    logger.info(f"Metadata: {meta_path}")

    # Step 5: Quick summary
    print("\n── Dataset Generation Summary ────────────────────────────")
    for fid, meta in facility_metadata.items():
        print(f"  Facility {fid:02d} [{meta['care_type']}]: "
              f"{meta['n_rows']} rows, mismatch={meta['mismatch_rate']:.1%}")
    print(f"\n  Total: {len(combined)} rows across {len(facility_metadata)} facilities")
    print(f"  Saved to: {SYNTHETIC_DIR}")


if __name__ == "__main__":
    run_generation()
