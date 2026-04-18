"""
FedAcuity — Data Loaders
Per-facility train/val/test splits with consistent seeding.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.schema import FEATURE_NAMES, LABEL_COL, FACILITY_CARE_TYPES, HELD_OUT_FACILITIES
from src.config import cfg

SYNTHETIC_DIR = Path(cfg["paths"]["data"]["synthetic"])
SEED = cfg["project"]["seed"]
SPLITS = cfg["data"]["splits"]


def load_facility(facility_id: int) -> pd.DataFrame:
    """Load a single facility's dataset."""
    care_type = FACILITY_CARE_TYPES[facility_id]
    path = SYNTHETIC_DIR / f"facility_{facility_id:02d}_{care_type}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run generator.py first. Missing: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def get_facility_splits(
    facility_id: int,
    df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (X_train, X_val, X_test), (y_train, y_val, y_test) for a facility."""
    if df is None:
        df = load_facility(facility_id)

    features = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[features].fillna(0)
    y = df[LABEL_COL]

    # First split: train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=SPLITS["test"],
        random_state=SEED + facility_id,
        stratify=y,
    )

    # Second split: train / val
    val_ratio = SPLITS["val"] / (SPLITS["train"] + SPLITS["val"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=SEED + facility_id,
        stratify=y_trainval,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_all_facilities(exclude_held_out: bool = True) -> Dict[int, Dict]:
    """
    Load all facilities into a dict:
      { facility_id: { "train": (X, y), "val": (X, y), "test": (X, y), "care_type": str } }
    """
    result = {}
    for fid in FACILITY_CARE_TYPES:
        if exclude_held_out and fid in HELD_OUT_FACILITIES:
            continue
        df = load_facility(fid)
        train, val, test = get_facility_splits(fid, df)
        result[fid] = {
            "train": train,
            "val":   val,
            "test":  test,
            "care_type": FACILITY_CARE_TYPES[fid],
        }
    return result


def load_held_out() -> Dict[int, Dict]:
    """Load the held-out facilities for final evaluation."""
    result = {}
    for fid in HELD_OUT_FACILITIES:
        df = load_facility(fid)
        _, _, test = get_facility_splits(fid, df)
        result[fid] = {"test": test, "care_type": FACILITY_CARE_TYPES[fid]}
    return result


def pool_all_data(exclude_held_out: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Pool all facility data for centralised oracle training."""
    facilities = load_all_facilities(exclude_held_out)
    X_parts, y_parts = [], []
    for fid, splits in facilities.items():
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_parts.extend([X_train, X_val])
        y_parts.extend([y_train, y_val])
    return pd.concat(X_parts), pd.concat(y_parts)
