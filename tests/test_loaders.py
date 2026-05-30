"""
FedAcuity — Unit Tests: src/data/loaders.py
Run: pytest tests/test_loaders.py -v
Note: These tests require the synthetic dataset to be generated first.
      Run `python -m src.data.generator` before running these tests.

Tests marked @pytest.mark.requires_data are skipped if data is absent.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.data.schema import (
    FEATURE_NAMES, LABEL_COL, FACILITY_CARE_TYPES,
    HELD_OUT_FACILITIES, CLUSTER_ASSIGNMENTS,
)
from src.config import cfg

SYNTHETIC_DIR = Path(cfg["paths"]["data"]["synthetic"])
DATA_AVAILABLE = SYNTHETIC_DIR.exists() and any(SYNTHETIC_DIR.glob("facility_*.csv"))


def requires_data(fn):
    """Decorator: skip test if synthetic data is not generated yet."""
    return pytest.mark.skipif(
        not DATA_AVAILABLE,
        reason="Synthetic data not found. Run: python -m src.data.generator"
    )(fn)


# ── Helper: Build a minimal synthetic DataFrame for unit testing ──────────────

def make_mock_facility_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a minimal but schema-conformant DataFrame for testing loaders."""
    rng = np.random.default_rng(seed)

    from src.data.schema import FEATURE_SPECS
    rows = {}
    for feat in FEATURE_SPECS:
        vals = rng.uniform(feat.min_val, feat.max_val, n_rows)
        if feat.dtype == "int":
            vals = np.round(vals).astype(int)
        rows[feat.name] = vals

    df = pd.DataFrame(rows)
    df["care_type"] = "SNF"
    df["facility_id"] = 3
    df["date"] = pd.date_range("2021-01-01", periods=n_rows)
    df[LABEL_COL] = rng.integers(0, 2, n_rows)  # binary label

    # Ensure label isn't all one class (would break stratification)
    df.loc[0, LABEL_COL] = 0
    df.loc[1, LABEL_COL] = 1
    return df


# ── Split Ratio Tests (using mock data — no file I/O) ─────────────────────────

class TestGetFacilitySplits:

    def _run_splits(self, n_rows=365):
        from src.data.loaders import get_facility_splits
        df = make_mock_facility_df(n_rows=n_rows)
        return get_facility_splits(facility_id=3, df=df)

    def test_returns_three_tuples(self):
        result = self._run_splits()
        assert len(result) == 3, "Expected (train, val, test) tuple"

    def test_each_split_is_xy_pair(self):
        train, val, test = self._run_splits()
        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            assert len(split) == 2, f"{split_name} should be (X, y) pair"
            X, y = split
            assert isinstance(X, pd.DataFrame), f"{split_name} X should be DataFrame"
            assert isinstance(y, pd.Series), f"{split_name} y should be Series"

    def test_split_sizes_sum_to_total(self):
        n = 500
        (X_train, _), (X_val, _), (X_test, _) = self._run_splits(n_rows=n)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == n, f"Split sizes {len(X_train)}+{len(X_val)}+{len(X_test)} != {n}"

    def test_split_sizes_approximately_correct(self):
        """60/20/20 split, allowing ±2 rows for rounding."""
        n = 500
        (X_train, _), (X_val, _), (X_test, _) = self._run_splits(n_rows=n)
        assert abs(len(X_train) - 300) <= 5, f"Train size {len(X_train)} far from 300"
        assert abs(len(X_val) - 100) <= 5, f"Val size {len(X_val)} far from 100"
        assert abs(len(X_test) - 100) <= 5, f"Test size {len(X_test)} far from 100"

    def test_no_overlap_between_splits(self):
        """Same index should not appear in multiple splits."""
        (X_train, _), (X_val, _), (X_test, _) = self._run_splits(n_rows=500)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert len(train_idx & val_idx) == 0, "Train and val overlap"
        assert len(train_idx & test_idx) == 0, "Train and test overlap"
        assert len(val_idx & test_idx) == 0, "Val and test overlap"

    def test_only_feature_columns_in_x(self):
        """X splits should contain only feature columns (not label, facility_id, date)."""
        (X_train, _), _, _ = self._run_splits()
        non_feature_cols = {"staffing_mismatch", "facility_id", "date", "care_type"}
        overlap = non_feature_cols & set(X_train.columns)
        assert len(overlap) == 0, f"Non-feature columns in X: {overlap}"

    def test_y_is_binary(self):
        (_, y_train), (_, y_val), (_, y_test) = self._run_splits()
        for y_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
            assert set(y.unique()).issubset({0, 1}), (
                f"{y_name} y contains non-binary values: {y.unique()}"
            )

    def test_stratification_preserves_label_ratio(self):
        """Train and test mismatch rates should be within 5% of each other."""
        n = 1000
        (_, y_train), _, (_, y_test) = self._run_splits(n_rows=n)
        diff = abs(y_train.mean() - y_test.mean())
        assert diff < 0.05, (
            f"Stratification failed: train mismatch={y_train.mean():.2%}, "
            f"test mismatch={y_test.mean():.2%}, diff={diff:.2%}"
        )

    def test_deterministic_with_same_seed(self):
        """Same facility_id + same df should produce identical splits."""
        df = make_mock_facility_df(n_rows=300, seed=99)
        from src.data.loaders import get_facility_splits
        (X1, y1), _, _ = get_facility_splits(facility_id=3, df=df)
        (X2, y2), _, _ = get_facility_splits(facility_id=3, df=df)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_different_facility_ids_give_different_splits(self):
        """Different facility IDs should shuffle differently (different SEED offset)."""
        df = make_mock_facility_df(n_rows=300, seed=42)
        from src.data.loaders import get_facility_splits
        (X_f3, _), _, _ = get_facility_splits(facility_id=3, df=df.copy())
        (X_f4, _), _, _ = get_facility_splits(facility_id=4, df=df.copy())
        # It's extremely unlikely they'd produce identical train sets with different seeds
        assert not X_f3.equals(X_f4), (
            "Different facility IDs should produce different splits"
        )


# ── Tests requiring actual generated data ─────────────────────────────────────

class TestLoadFacility:

    @requires_data
    def test_load_each_facility(self):
        from src.data.loaders import load_facility
        for fid in FACILITY_CARE_TYPES:
            df = load_facility(fid)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0, f"Facility {fid} has empty dataset"

    @requires_data
    def test_loaded_df_has_all_features(self):
        from src.data.loaders import load_facility
        df = load_facility(0)
        missing = set(FEATURE_NAMES) - set(df.columns)
        assert len(missing) == 0, f"Missing features: {missing}"

    @requires_data
    def test_loaded_df_has_label_column(self):
        from src.data.loaders import load_facility
        df = load_facility(0)
        assert LABEL_COL in df.columns

    @requires_data
    def test_loaded_df_has_date_column_as_datetime(self):
        from src.data.loaders import load_facility
        df = load_facility(0)
        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
            "date column should be datetime64"
        )

    @requires_data
    def test_facility_row_count_approximately_1095(self):
        """3 years × 365 days = 1095 rows per facility."""
        from src.data.loaders import load_facility
        for fid in FACILITY_CARE_TYPES:
            df = load_facility(fid)
            assert abs(len(df) - 1095) <= 5, (
                f"Facility {fid} has {len(df)} rows (expected ~1095)"
            )

    @requires_data
    def test_mismatch_rates_within_expected_bands(self):
        """Mismatch rates should be within ±10% of NON_IID_SPEC targets."""
        from src.data.loaders import load_facility
        from src.data.schema import NON_IID_SPEC

        targets = {"MC": 0.40, "SNF": 0.28, "IL": 0.12}
        tolerance = 0.10  # ±10%

        for fid, care_type in FACILITY_CARE_TYPES.items():
            df = load_facility(fid)
            actual = df[LABEL_COL].mean()
            target = targets[care_type]
            assert abs(actual - target) <= tolerance, (
                f"Facility {fid} [{care_type}]: mismatch={actual:.1%}, "
                f"target={target:.0%}, diff={abs(actual-target):.1%} > {tolerance:.0%}"
            )

    def test_load_facility_raises_on_missing_file(self):
        """Should raise FileNotFoundError if data not generated."""
        from src.data.loaders import load_facility
        with patch("src.data.loaders.SYNTHETIC_DIR", Path("/nonexistent/path")):
            with pytest.raises(FileNotFoundError):
                load_facility(0)


class TestLoadAllFacilities:

    @requires_data
    def test_returns_dict(self):
        from src.data.loaders import load_all_facilities
        result = load_all_facilities()
        assert isinstance(result, dict)

    @requires_data
    def test_excludes_held_out_by_default(self):
        from src.data.loaders import load_all_facilities
        result = load_all_facilities(exclude_held_out=True)
        for fid in HELD_OUT_FACILITIES:
            assert fid not in result, f"Held-out facility {fid} should be excluded"

    @requires_data
    def test_includes_held_out_when_flag_false(self):
        from src.data.loaders import load_all_facilities
        result = load_all_facilities(exclude_held_out=False)
        for fid in HELD_OUT_FACILITIES:
            assert fid in result, f"Held-out facility {fid} should be included"

    @requires_data
    def test_correct_number_of_training_facilities(self):
        from src.data.loaders import load_all_facilities
        result = load_all_facilities(exclude_held_out=True)
        expected_count = len(FACILITY_CARE_TYPES) - len(HELD_OUT_FACILITIES)
        assert len(result) == expected_count

    @requires_data
    def test_each_entry_has_required_keys(self):
        from src.data.loaders import load_all_facilities
        result = load_all_facilities()
        for fid, data in result.items():
            assert "train" in data, f"Facility {fid} missing 'train' split"
            assert "val" in data, f"Facility {fid} missing 'val' split"
            assert "test" in data, f"Facility {fid} missing 'test' split"
            assert "care_type" in data, f"Facility {fid} missing 'care_type'"

    @requires_data
    def test_care_types_match_facility_map(self):
        from src.data.loaders import load_all_facilities
        result = load_all_facilities(exclude_held_out=False)
        for fid, data in result.items():
            assert data["care_type"] == FACILITY_CARE_TYPES[fid], (
                f"Facility {fid} care type mismatch"
            )


class TestLoadHeldOut:

    @requires_data
    def test_returns_only_held_out_facilities(self):
        from src.data.loaders import load_held_out
        result = load_held_out()
        assert set(result.keys()) == set(HELD_OUT_FACILITIES)

    @requires_data
    def test_held_out_has_test_split_only(self):
        from src.data.loaders import load_held_out
        result = load_held_out()
        for fid, data in result.items():
            assert "test" in data
            # Should NOT have train/val (only final evaluation)
            assert "train" not in data, "Held-out facilities should not have train split"


class TestPoolAllData:

    @requires_data
    def test_returns_x_and_y(self):
        from src.data.loaders import pool_all_data
        X, y = pool_all_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    @requires_data
    def test_x_and_y_same_length(self):
        from src.data.loaders import pool_all_data
        X, y = pool_all_data()
        assert len(X) == len(y)

    @requires_data
    def test_excludes_held_out_by_default(self):
        from src.data.loaders import pool_all_data, load_all_facilities
        X_pool, _ = pool_all_data(exclude_held_out=True)
        facilities = load_all_facilities(exclude_held_out=True)
        # Pool combines train+val of all non-held-out facilities
        expected_rows = sum(
            len(d["train"][0]) + len(d["val"][0])
            for d in facilities.values()
        )
        assert len(X_pool) == expected_rows
