from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "2d-gaussian-splatting"))

from planes.global_plane_support import (  # noqa: E402
    append_plane_coverage_audit,
    filter_global_planes_by_view_support,
)


def test_single_view_global_planes_cannot_rewrite_chart_depth():
    active, audit = filter_global_planes_by_view_support(
        {
            3: [(0, 8)],
            5: [(1, 2), (1, 9)],
            7: [(1, 4), (2, 6)],
        },
        minimum_distinct_views=2,
    )

    assert set(active) == {7}
    assert audit["accepted_global_plane_count"] == 1
    assert audit["rejected_global_plane_count"] == 2
    assert audit["rejected_local_membership_count"] == 3
    records = {row["global_plane_id"]: row for row in audit["records"]}
    assert records[5]["distinct_view_count"] == 1
    assert records[5]["reason"] == "insufficient_cross_view_support"
    assert records[7]["accepted"] is True


def test_plane_support_threshold_is_validated():
    with pytest.raises(ValueError, match="at least one"):
        filter_global_planes_by_view_support({}, minimum_distinct_views=0)


def test_generated_views_cannot_supply_cross_view_geometry_evidence():
    active, audit = filter_global_planes_by_view_support(
        {
            # View 40 is generated when the real Chart list has IDs [0, 40).
            1: [(3, 2), (40, 7)],
            2: [(3, 2), (4, 7), (40, 8)],
        },
        minimum_distinct_views=2,
        real_view_upper_bound=40,
    )

    assert set(active) == {2}
    records = {row["global_plane_id"]: row for row in audit["records"]}
    assert records[1]["distinct_view_count"] == 2
    assert records[1]["distinct_real_view_count"] == 1
    assert records[1]["accepted"] is False
    assert records[2]["real_view_ids"] == [3, 4]


def test_cross_view_coverage_audit_does_not_treat_missing_plane_as_failure():
    active, audit = filter_global_planes_by_view_support(
        {9: [(0, 1), (1, 2)]},
        minimum_distinct_views=2,
    )
    append_plane_coverage_audit(
        audit,
        active_global_plane_members=active,
        plane_masks=[
            np.array([[1, 1], [0, 3]], dtype=np.int32),
            np.array([[2, 0], [0, 0]], dtype=np.int32),
            np.zeros((2, 2), dtype=np.int32),
        ],
        visibility_masks=[
            np.array([[1, 0], [1, 1]], dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
        ],
    )

    coverage = audit["eligible_cross_view_coverage"]
    assert coverage["charts_without_eligible_cross_view_plane"] == 1
    assert coverage["all_active_global_planes_have_min_real_support"] is True
    assert coverage["per_chart"][0]["eligible_cross_view_plane_pixel_fraction"] == 0.5
    assert coverage["per_chart"][0]["eligible_visible_pixel_fraction"] == 0.25
