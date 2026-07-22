import numpy as np

from scripts import select_chart_views


def test_pose_near_reserves_restore_each_critical_primary_without_counting_as_third_baseline(
    monkeypatch,
):
    """A temporal mate may replace a failed primary but is not a primary support."""
    names = ["primary_a.png", "primary_b.png", "reserve_a.png", "reserve_b.png"]
    centers = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.98, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    directions = np.zeros_like(centers)
    monkeypatch.setattr(
        select_chart_views,
        "build_pose_geometry",
        lambda image_names, poses: (centers, directions, 1.0),
    )
    monkeypatch.setattr(
        select_chart_views,
        "_farthest_point_indices",
        lambda features, count: [0],
    )

    result = select_chart_views.select_gate_replacement_reserves(
        image_names=names,
        candidate_image_names=names,
        poses={},
        selected_indices=[0, 1],
        n_view_clusters=1,
        post_gate_min_views_per_cluster=2,
        min_baseline_ratio=0.5,
        max_borrowed_support_distance=2.0,
        reserve_max_pose_distance=0.05,
        reserves_per_vulnerable_support=1,
    )

    assert result["certified_for_one_primary_gate_rejection"] is True
    assert result["reserve_image_idx"] == [2, 3]
    assert result["alignment_image_idx"] == [0, 1, 2, 3]
    substitutions = {record["primary_image_idx"]: record for record in result["substitutions"]}
    assert substitutions[0]["reserve_image_idx"] == [2]
    assert substitutions[1]["reserve_image_idx"] == [3]
    assert all(record["passed"] for record in substitutions.values())


def test_reserve_audit_is_not_certified_when_clean_nearby_substitute_is_missing(monkeypatch):
    names = ["primary_a.png", "primary_b.png", "far_clean.png"]
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    directions = np.zeros_like(centers)
    monkeypatch.setattr(
        select_chart_views,
        "build_pose_geometry",
        lambda image_names, poses: (centers, directions, 1.0),
    )
    monkeypatch.setattr(
        select_chart_views,
        "_farthest_point_indices",
        lambda features, count: [0],
    )

    result = select_chart_views.select_gate_replacement_reserves(
        image_names=names,
        candidate_image_names=names,
        poses={},
        selected_indices=[0, 1],
        n_view_clusters=1,
        post_gate_min_views_per_cluster=2,
        min_baseline_ratio=0.5,
        max_borrowed_support_distance=2.0,
        reserve_max_pose_distance=0.05,
        reserves_per_vulnerable_support=1,
    )

    assert result["certified_for_one_primary_gate_rejection"] is False
    assert result["reserve_image_idx"] == []
    assert all(not record["passed"] for record in result["substitutions"])


def test_joint_reserves_cover_correlated_gate_failures_that_single_failure_audit_misses(
    monkeypatch,
):
    """Four primaries can pass a one-at-a-time audit yet lose three together."""
    names = [
        "primary_0.png",
        "primary_1.png",
        "primary_2.png",
        "primary_3.png",
        "joint_reserve.png",
    ]
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    directions = np.zeros_like(centers)
    monkeypatch.setattr(
        select_chart_views,
        "build_pose_geometry",
        lambda image_names, poses: (centers, directions, 1.0),
    )
    monkeypatch.setattr(
        select_chart_views,
        "_farthest_point_indices",
        lambda features, count: [0],
    )

    legacy = select_chart_views.select_gate_replacement_reserves(
        image_names=names,
        candidate_image_names=names,
        poses={},
        selected_indices=[0, 1, 2, 3],
        n_view_clusters=1,
        post_gate_min_views_per_cluster=2,
        min_baseline_ratio=0.5,
        max_borrowed_support_distance=10.0,
        reserve_max_pose_distance=10.0,
        reserves_per_vulnerable_support=1,
        gate_failure_budget=1,
    )
    assert legacy["certified_for_gate_failure_budget"] is True
    assert legacy["reserve_image_idx"] == []

    joint = select_chart_views.select_gate_replacement_reserves(
        image_names=names,
        candidate_image_names=names,
        poses={},
        selected_indices=[0, 1, 2, 3],
        n_view_clusters=1,
        post_gate_min_views_per_cluster=2,
        min_baseline_ratio=0.5,
        max_borrowed_support_distance=10.0,
        reserve_max_pose_distance=10.0,
        reserves_per_vulnerable_support=1,
        gate_failure_budget=3,
    )
    assert joint["method"] == "baseline_primary_plus_joint_gate_resilience"
    assert joint["certified_for_gate_failure_budget"] is True
    assert joint["reserve_image_idx"] == [4]
    assert joint["clusters"][0]["joint_failure_budget_audit"]["failure_scenario_count"] == 10
