"""
FedAcuity — Unit Tests: src/data/schema.py
Run: pytest tests/test_schema.py -v
"""

import numpy as np
import pytest

from src.data.schema import (
    FEATURE_SPECS,
    FEATURE_NAMES,
    LABEL_COL,
    CARE_TYPES,
    FACILITY_CARE_TYPES,
    HELD_OUT_FACILITIES,
    CLUSTER_ASSIGNMENTS,
    NON_IID_SPEC,
    adl_demand_score,
    compute_mismatch_label,
)


# ── Feature Schema Tests ──────────────────────────────────────────────────────

class TestFeatureSchema:

    def test_feature_count(self):
        """We expect exactly 14 features per the schema design."""
        assert len(FEATURE_SPECS) == 14, (
            f"Expected 14 features, got {len(FEATURE_SPECS)}"
        )

    def test_all_feature_names_unique(self):
        names = [f.name for f in FEATURE_SPECS]
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_feature_names_list_matches_specs(self):
        assert set(FEATURE_NAMES) == {f.name for f in FEATURE_SPECS}

    def test_label_col_not_in_features(self):
        assert LABEL_COL not in FEATURE_NAMES, (
            "Label column should not be in feature list"
        )

    def test_label_col_is_staffing_mismatch(self):
        assert LABEL_COL == "staffing_mismatch"

    def test_all_features_have_valid_ranges(self):
        for f in FEATURE_SPECS:
            assert f.min_val <= f.max_val, (
                f"Feature {f.name}: min_val ({f.min_val}) > max_val ({f.max_val})"
            )

    def test_dtype_values_are_valid(self):
        valid_dtypes = {"float", "int", "category"}
        for f in FEATURE_SPECS:
            assert f.dtype in valid_dtypes, (
                f"Feature {f.name} has invalid dtype: {f.dtype}"
            )

    def test_adl_features_have_correct_range(self):
        adl_features = ["adl_eating", "adl_mobility", "adl_toileting", "adl_cognition"]
        for name in adl_features:
            spec = next(f for f in FEATURE_SPECS if f.name == name)
            assert spec.min_val == 0, f"{name} min should be 0"
            assert spec.max_val == 6, f"{name} max should be 6 (MDS 3.0 scale)"

    def test_nursing_hours_features_exist(self):
        nursing_feats = {"nursing_hours_rn", "nursing_hours_lpn", "nursing_hours_cna"}
        assert nursing_feats.issubset(set(FEATURE_NAMES))

    def test_resident_census_range(self):
        spec = next(f for f in FEATURE_SPECS if f.name == "resident_census")
        assert spec.min_val == 10
        assert spec.max_val == 120


# ── Care Type & Facility Tests ────────────────────────────────────────────────

class TestCareTypesAndFacilities:

    def test_care_types(self):
        assert set(CARE_TYPES) == {"MC", "SNF", "IL"}

    def test_facility_count(self):
        assert len(FACILITY_CARE_TYPES) == 10

    def test_facility_ids_are_0_to_9(self):
        assert set(FACILITY_CARE_TYPES.keys()) == set(range(10))

    def test_care_type_distribution(self):
        """MC:3, SNF:4, IL:3 per config."""
        from collections import Counter
        counts = Counter(FACILITY_CARE_TYPES.values())
        assert counts["MC"] == 3
        assert counts["SNF"] == 4
        assert counts["IL"] == 3

    def test_all_facilities_have_valid_care_type(self):
        for fid, ct in FACILITY_CARE_TYPES.items():
            assert ct in CARE_TYPES, f"Facility {fid} has invalid care type: {ct}"

    def test_held_out_facilities_exist_in_facility_map(self):
        for fid in HELD_OUT_FACILITIES:
            assert fid in FACILITY_CARE_TYPES

    def test_held_out_count(self):
        """Exactly 2 facilities are held out: one SNF (6), one IL (9). Facility 8 moved to IL training cluster."""
        assert len(HELD_OUT_FACILITIES) == 2

    def test_held_out_facilities_span_snf_and_il(self):
        """Held-out facilities should cover one SNF and one IL care type (not MC, and not both the same type)."""
        held_out_types = {FACILITY_CARE_TYPES[fid] for fid in HELD_OUT_FACILITIES}
        assert held_out_types == {"SNF", "IL"}, (
            f"Held-out care types should be {{SNF, IL}}, got {held_out_types}"
        )

    def test_each_cluster_retains_at_least_two_training_clients(self):
        """Every cluster must keep >=2 non-held-out facilities so FL aggregation is genuine, not single-client."""
        for care_type, facilities in CLUSTER_ASSIGNMENTS.items():
            training = [fid for fid in facilities if fid not in HELD_OUT_FACILITIES]
            assert len(training) >= 2, (
                f"{care_type} cluster has only {len(training)} training client(s): {training}"
            )


# ── Cluster Assignment Tests ──────────────────────────────────────────────────

class TestClusterAssignments:

    def test_all_clusters_present(self):
        assert set(CLUSTER_ASSIGNMENTS.keys()) == {"MC", "SNF", "IL"}

    def test_cluster_covers_all_facilities(self):
        all_in_clusters = set()
        for facilities in CLUSTER_ASSIGNMENTS.values():
            all_in_clusters.update(facilities)
        assert all_in_clusters == set(range(10))

    def test_no_facility_in_multiple_clusters(self):
        seen = []
        for facilities in CLUSTER_ASSIGNMENTS.values():
            for fid in facilities:
                assert fid not in seen, f"Facility {fid} appears in multiple clusters"
                seen.append(fid)

    def test_cluster_care_types_match_facility_care_types(self):
        """Every facility in a cluster should have the matching care type."""
        for cluster_ct, facilities in CLUSTER_ASSIGNMENTS.items():
            for fid in facilities:
                assert FACILITY_CARE_TYPES[fid] == cluster_ct, (
                    f"Facility {fid} is in {cluster_ct} cluster "
                    f"but has care type {FACILITY_CARE_TYPES[fid]}"
                )

    def test_mc_cluster_size(self):
        assert len(CLUSTER_ASSIGNMENTS["MC"]) == 3

    def test_snf_cluster_size(self):
        assert len(CLUSTER_ASSIGNMENTS["SNF"]) == 4

    def test_il_cluster_size(self):
        assert len(CLUSTER_ASSIGNMENTS["IL"]) == 3


# ── Non-IID Spec Tests ────────────────────────────────────────────────────────

class TestNonIIDSpec:

    def test_all_care_types_in_spec(self):
        assert set(NON_IID_SPEC.keys()) == {"MC", "SNF", "IL"}

    def test_mismatch_rates_in_spec(self):
        for ct, spec in NON_IID_SPEC.items():
            assert "mismatch_rate" in spec, f"{ct} missing mismatch_rate"
            rate = spec["mismatch_rate"]
            assert 0.0 < rate < 1.0, f"{ct} mismatch_rate {rate} out of range"

    def test_mismatch_rates_ordering(self):
        """MC should have highest mismatch, IL lowest."""
        assert NON_IID_SPEC["MC"]["mismatch_rate"] > NON_IID_SPEC["SNF"]["mismatch_rate"]
        assert NON_IID_SPEC["SNF"]["mismatch_rate"] > NON_IID_SPEC["IL"]["mismatch_rate"]

    def test_mc_has_highest_cognition_mean(self):
        mc_cog = NON_IID_SPEC["MC"]["adl_cognition"]["mean"]
        il_cog = NON_IID_SPEC["IL"]["adl_cognition"]["mean"]
        assert mc_cog > il_cog, "MC should have higher cognitive impairment than IL"

    def test_il_has_lowest_medication_count(self):
        mc_med = NON_IID_SPEC["MC"]["medication_count"]["mean"]
        il_med = NON_IID_SPEC["IL"]["medication_count"]["mean"]
        assert mc_med > il_med, "MC should have higher medication count than IL"


# ── ADL Demand Score Tests ────────────────────────────────────────────────────

class TestADLDemandScore:

    def test_output_shape(self):
        n = 100
        score = adl_demand_score(
            np.ones(n) * 3,
            np.ones(n) * 3,
            np.ones(n) * 3,
            np.ones(n) * 3,
        )
        assert score.shape == (n,)

    def test_all_zeros_gives_zero(self):
        score = adl_demand_score(
            np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
        )
        np.testing.assert_array_equal(score, 0.0)

    def test_all_max_gives_one(self):
        """All ADL scores at max (6) should give demand score of 1.0."""
        score = adl_demand_score(
            np.ones(10) * 6,
            np.ones(10) * 6,
            np.ones(10) * 6,
            np.ones(10) * 6,
        )
        np.testing.assert_array_almost_equal(score, 1.0)

    def test_output_range(self):
        """Output should always be in [0, 1] for valid ADL inputs."""
        rng = np.random.default_rng(42)
        n = 1000
        score = adl_demand_score(
            rng.uniform(0, 6, n),
            rng.uniform(0, 6, n),
            rng.uniform(0, 6, n),
            rng.uniform(0, 6, n),
        )
        assert score.min() >= 0.0
        assert score.max() <= 1.0

    def test_weights_sum_to_one(self):
        """The weighted combination (0.20+0.30+0.25+0.25=1.0) / 6 = 1/6 per unit."""
        # All ADL at value 1.0: expected score = (0.20+0.30+0.25+0.25)*1/6 = 1/6
        score = adl_demand_score(
            np.ones(1), np.ones(1), np.ones(1), np.ones(1)
        )
        expected = 1.0 / 6.0
        np.testing.assert_almost_equal(score[0], expected, decimal=6)

    def test_cognition_has_same_weight_as_toileting(self):
        """adl_cognition and adl_toileting both have weight 0.25."""
        # Set only toileting to 6
        s1 = adl_demand_score(np.zeros(1), np.zeros(1), np.ones(1)*6, np.zeros(1))
        # Set only cognition to 6
        s2 = adl_demand_score(np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1)*6)
        np.testing.assert_almost_equal(s1[0], s2[0], decimal=6)

    def test_mobility_has_higher_weight_than_eating(self):
        """adl_mobility (0.30) should have higher impact than adl_eating (0.20)."""
        s_mobility = adl_demand_score(np.zeros(1), np.ones(1)*6, np.zeros(1), np.zeros(1))
        s_eating = adl_demand_score(np.ones(1)*6, np.zeros(1), np.zeros(1), np.zeros(1))
        assert s_mobility[0] > s_eating[0]


# ── Mismatch Label Tests ──────────────────────────────────────────────────────

class TestComputeMismatchLabel:

    def test_output_is_binary(self):
        rng = np.random.default_rng(42)
        n = 500
        label = compute_mismatch_label(
            demand=rng.uniform(0, 1, n),
            resident_census=rng.integers(20, 80, n),
            nursing_hours_available=rng.uniform(2, 10, n),
        )
        unique_vals = np.unique(label)
        assert set(unique_vals).issubset({0, 1}), f"Non-binary values: {unique_vals}"

    def test_high_demand_low_supply_is_mismatch(self):
        """Maximum demand and minimum supply should always be mismatch."""
        label = compute_mismatch_label(
            demand=np.ones(10),       # max normalised demand
            resident_census=np.ones(10) * 100,  # high census
            nursing_hours_available=np.ones(10) * 0.1,  # near-zero supply
        )
        assert label.all(), "High demand / low supply should be mismatch=1"

    def test_zero_demand_is_never_mismatch(self):
        """Zero ADL demand should never trigger mismatch."""
        label = compute_mismatch_label(
            demand=np.zeros(10),
            resident_census=np.ones(10) * 50,
            nursing_hours_available=np.ones(10) * 5,
        )
        assert not label.any(), "Zero demand should be mismatch=0"

    def test_output_shape_matches_input(self):
        n = 200
        label = compute_mismatch_label(
            demand=np.ones(n) * 0.5,
            resident_census=np.ones(n) * 50,
            nursing_hours_available=np.ones(n) * 5,
        )
        assert label.shape == (n,)

    def test_threshold_controls_rate(self):
        """Higher threshold → lower mismatch rate."""
        rng = np.random.default_rng(42)
        n = 1000
        demand = rng.uniform(0.1, 0.9, n)
        census = rng.integers(20, 80, n).astype(float)
        supply = rng.uniform(1, 8, n)

        rate_low_threshold = compute_mismatch_label(demand, census, supply, threshold=0.1).mean()
        rate_high_threshold = compute_mismatch_label(demand, census, supply, threshold=5.0).mean()

        assert rate_low_threshold > rate_high_threshold, (
            "Lower threshold should produce higher mismatch rate"
        )

    def test_division_by_zero_is_handled(self):
        """Zero nursing hours should not raise ZeroDivisionError."""
        label = compute_mismatch_label(
            demand=np.ones(5) * 0.5,
            resident_census=np.ones(5) * 50,
            nursing_hours_available=np.zeros(5),  # zero supply
        )
        assert label.shape == (5,)

    def test_dtype_is_int(self):
        label = compute_mismatch_label(
            demand=np.array([0.5]),
            resident_census=np.array([50.0]),
            nursing_hours_available=np.array([5.0]),
        )
        assert label.dtype in [np.int32, np.int64, int], (
            f"Expected int dtype, got {label.dtype}"
        )
