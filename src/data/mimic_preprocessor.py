"""
FedAcuity -- MIMIC-IV preprocessing for fidelity validation (C2)
Extracts an elderly patient subset from raw MIMIC-IV v3.1 hosp tables and
maps available features to the FedAcuity schema.

Feature mapping:
  medication_count  <- prescriptions: distinct drugs per admission
  rug_category      <- diagnoses_icd: diagnosis count binned to 1-8 scale (proxy)
  resident_census   <- admissions: daily concurrent patient census
  staffing_mismatch <- discharge_location: post-acute/LTC discharge = 1, else 0

Features NOT mappable from MIMIC-IV (hospital EHR != LTC MDS):
  adl_*, mds_adl_summary, nursing_hours_*, fall_risk_score,
  pain_assessment_score, incident_count

fidelity.py only validates features present in BOTH datasets, so partial
overlap is correct and scientifically honest.

Usage:
    python src/data/mimic_preprocessor.py

Output:
    data/mimic_iv/mimic_elderly_subset.parquet
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HOSP_DIR = Path("data/mimic_iv")   # files placed flat in this directory
OUT_PATH  = Path("data/mimic_iv/mimic_elderly_subset.parquet")

AGE_MIN = 65

# discharge locations indicating high post-acute care need (proxy for staffing mismatch=1)
HIGH_ACUITY_DISCHARGE = {
    "SKILLED NURSING FACILITY",
    "CHRONIC/LONG TERM ACUTE CARE",
    "REHAB",
    "LONG TERM CARE HOSPITAL",
    "HOSPICE",
    "ASSISTED LIVING",
}


def _load(name: str) -> pd.DataFrame:
    path = HOSP_DIR / name
    if not path.exists():
        sys.exit(f"Missing: {path}  --  run src/data/download_mimic.py first.")
    log.info(f"Loading {name} ...")
    return pd.read_csv(path, compression="gzip", low_memory=False)


def build_subset() -> pd.DataFrame:
    # ---- patients: filter to elderly ----------------------------------------
    patients = _load("patients.csv.gz")[["subject_id", "anchor_age", "gender"]]
    elderly  = patients[patients["anchor_age"] >= AGE_MIN].copy()
    log.info(f"Elderly patients (age >= {AGE_MIN}): {len(elderly):,}")

    # ---- admissions: join + derive census + label ---------------------------
    admissions = _load("admissions.csv.gz")[[
        "subject_id", "hadm_id", "admittime", "dischtime", "discharge_location"
    ]]
    admissions = admissions.merge(elderly[["subject_id", "anchor_age"]], on="subject_id", how="inner")
    admissions["admittime"]  = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"]  = pd.to_datetime(admissions["dischtime"])

    log.info(f"Elderly admissions: {len(admissions):,}")

    # Proxy label: discharged to post-acute / LTC setting = staffing_mismatch=1
    admissions["discharge_location_clean"] = (
        admissions["discharge_location"].fillna("").str.strip().str.upper()
    )
    admissions["staffing_mismatch"] = (
        admissions["discharge_location_clean"].isin(HIGH_ACUITY_DISCHARGE)
    ).astype(int)
    mismatch_rate = admissions["staffing_mismatch"].mean()
    log.info(f"Proxy mismatch rate (discharged to post-acute): {mismatch_rate:.1%}")

    # Daily resident census: concurrent admissions per calendar date.
    # Vectorised O(n) sweep instead of O(n_dates x n_admissions) loop.
    log.info("Computing daily census (vectorised) ...")
    valid = admissions[["hadm_id", "admittime", "dischtime"]].dropna().copy()
    admit_dates_only = valid["admittime"].dt.normalize()
    # day after discharge: a patient leaves that day so census drops the next day
    discharge_plus1 = (valid["dischtime"] + pd.Timedelta(days=1)).dt.normalize()

    events = pd.concat([
        pd.Series(1,  index=admit_dates_only),
        pd.Series(-1, index=discharge_plus1),
    ]).groupby(level=0).sum().sort_index()

    census_series = events.cumsum()
    # forward-fill gaps (dates with no event keep previous census)
    full_range = pd.date_range(census_series.index.min(), census_series.index.max(), freq="D")
    census_series = census_series.reindex(full_range).ffill().fillna(0)

    admissions["admit_date"] = admit_dates_only.values
    admissions["resident_census"] = admissions["admit_date"].map(census_series).fillna(0).astype(int)

    # ---- prescriptions: medication count per admission ----------------------
    prescriptions = _load("prescriptions.csv.gz")[["hadm_id", "drug"]]
    prescriptions = prescriptions[prescriptions["hadm_id"].isin(admissions["hadm_id"])]
    med_count = (
        prescriptions.groupby("hadm_id")["drug"]
        .nunique()
        .reset_index()
        .rename(columns={"drug": "medication_count"})
    )
    log.info(f"Median medication count: {med_count['medication_count'].median():.1f}")

    # ---- diagnoses: fallback diagnosis count (used only if drgcodes missing) --
    diagnoses = _load("diagnoses_icd.csv.gz")[["hadm_id", "icd_code"]]
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(admissions["hadm_id"])]
    diag_count = (
        diagnoses.groupby("hadm_id")["icd_code"]
        .count()
        .reset_index()
        .rename(columns={"icd_code": "diag_count"})
    )

    # ---- drgcodes: better RUG proxy (DRG = Diagnosis Related Group) ---------
    # DRG weight directly encodes resource utilisation — closer to RUG concept.
    drg_path = HOSP_DIR / "drgcodes.csv.gz"
    if drg_path.exists():
        drgcodes = _load("drgcodes.csv.gz")[["hadm_id", "drg_type", "drg_severity"]]
        drgcodes = drgcodes[drgcodes["hadm_id"].isin(admissions["hadm_id"])]
        # Use max severity per admission (scale 1-4 in MIMIC; stretch to 1-8)
        drg_severity = (
            drgcodes.groupby("hadm_id")["drg_severity"]
            .max()
            .reset_index()
            .rename(columns={"drg_severity": "drg_sev_max"})
        )
        drg_severity["drg_sev_max"] = pd.to_numeric(drg_severity["drg_sev_max"], errors="coerce").fillna(1)
        # Map DRG severity 1-4 → RUG 1-8 (double to match scale)
        drg_severity["rug_category"] = (drg_severity["drg_sev_max"] * 2).clip(1, 8).astype(int)
        diag_count = diag_count.merge(drg_severity[["hadm_id", "rug_category"]], on="hadm_id", how="left")
        log.info("rug_category: using DRG severity (preferred)")
    else:
        # Fallback: bin raw diagnosis count to 1-8
        diag_count["rug_category"] = pd.qcut(
            diag_count["diag_count"], q=8, labels=range(1, 9), duplicates="drop"
        ).astype(float).astype(int)
        log.info("rug_category: using diagnosis count quantile bins (drgcodes.csv.gz not found)")

    # ---- assemble final DataFrame -------------------------------------------
    df = admissions[["hadm_id", "subject_id", "anchor_age", "resident_census",
                      "staffing_mismatch"]].copy()
    df = df.merge(med_count, on="hadm_id", how="left")
    df = df.merge(diag_count[["hadm_id", "rug_category"]], on="hadm_id", how="left")

    df["medication_count"] = df["medication_count"].fillna(0).astype(int)
    df["rug_category"]     = df["rug_category"].fillna(1).astype(int)

    # Clip to schema ranges
    df["medication_count"] = df["medication_count"].clip(0, 20)
    df["resident_census"]  = df["resident_census"].clip(10, 1000)  # hospital census >> 120
    df["rug_category"]     = df["rug_category"].clip(1, 8)

    # Drop ID columns before saving
    df = df.drop(columns=["hadm_id", "subject_id", "anchor_age"])

    log.info(f"Final subset: {len(df):,} rows x {df.shape[1]} columns")
    log.info(f"  Columns: {list(df.columns)}")
    log.info(f"  medication_count  -- mean={df['medication_count'].mean():.1f}, "
             f"std={df['medication_count'].std():.1f}")
    log.info(f"  rug_category      -- mean={df['rug_category'].mean():.1f}")
    log.info(f"  resident_census   -- mean={df['resident_census'].mean():.0f}")
    log.info(f"  staffing_mismatch -- rate={df['staffing_mismatch'].mean():.1%}")

    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset = build_subset()
    subset.to_parquet(OUT_PATH, index=False)
    log.info(f"Saved: {OUT_PATH}")
    print("\nMIMIC-IV elderly subset ready.")
    print(f"  Rows: {len(subset):,}")
    print(f"  Features mapped to schema: medication_count, rug_category, resident_census")
    print(f"  Label proxy: discharge to post-acute/LTC facility = staffing_mismatch")
    print(f"  Features NOT mappable (LTC-specific): adl_*, nursing_hours_*, "
          f"fall_risk_score, pain_assessment_score, incident_count, mds_adl_summary")
    print(f"\nNext: python -m src.data.fidelity")
