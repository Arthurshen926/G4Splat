import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_gated_chart_coverage import audit_coverage
from scripts.select_chart_views import (
    candidate_reliability_scores,
    select_clustered_coverage_indices,
    static_support_ratios,
)


class _MaskLookup:
    def __init__(self, invalid):
        self.invalid = invalid

    def invalid_ratio_for_indices(self, image_name, mask_indices):
        assert mask_indices == [0, 1, 2, 3]
        return self.invalid[image_name]


def _poses(names):
    """Identity cameras laid out as three short independent trajectories."""
    centers = {
        "seq1__frame00000.png": 0.00,
        "seq1__frame00001.png": 0.10,
        "seq2__frame00000.png": 1.00,
        "seq2__frame00001.png": 1.10,
        "seq3__frame00000.png": 2.00,
        "seq3__frame00001.png": 2.10,
    }
    return {
        name: (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([-centers[name], 0.0, 0.0]),
        )
        for name in names
    }


def test_static_support_is_continuous_and_tree_is_only_a_weak_quality_prior():
    names = ["seq1__frame00000.png", "seq1__frame00001.png"]
    support = static_support_ratios(
        names,
        _MaskLookup({names[0]: 0.08, names[1]: 0.48}),
        [0, 1, 2, 3],
    )

    assert support == {names[0]: pytest.approx(0.92), names[1]: pytest.approx(0.52)}
    reliability = candidate_reliability_scores(
        names,
        quality_scores={
            name: {
                "sharpness": 1.0,
                "extreme_ratio": 0.0,
                "shadow_ratio": 0.0,
                "lower_mean_intensity": 1.0,
            }
            for name in names
        },
        static_support=support,
        tree_invalid_ratios={names[0]: 0.10, names[1]: 0.50},
    )
    assert reliability[names[0]] > reliability[names[1]]


def test_sequence_seed_reserves_two_baselined_views_per_trajectory():
    names = [
        "seq1__frame00000.png",
        "seq1__frame00001.png",
        "seq2__frame00000.png",
        "seq2__frame00001.png",
        "seq3__frame00000.png",
        "seq3__frame00001.png",
    ]
    selected, diagnostics = select_clustered_coverage_indices(
        names,
        names,
        _poses(names),
        n_images=6,
        n_view_clusters=1,
        min_views_per_cluster=2,
        min_baseline_ratio=0.02,
        min_global_pose_distance=0.001,
        max_borrowed_support_distance=1.0,
        min_views_per_sequence=2,
        sequence_coverage_mode="strict",
    )

    assert len(selected) == 6
    assert all(record["passed"] for record in diagnostics["sequences"])
    assert {record["sequence"] for record in diagnostics["sequences"]} == {
        "seq1", "seq2", "seq3"
    }


def test_sequence_seed_survives_one_hard_gate_rejection_per_trajectory():
    names = [
        f"seq{sequence}__frame{frame:05d}.png"
        for sequence in range(1, 4)
        for frame in range(3)
    ]
    centers = {
        name: float((sequence - 1) + 0.10 * frame)
        for sequence in range(1, 4)
        for frame, name in enumerate(
            [f"seq{sequence}__frame{index:05d}.png" for index in range(3)]
        )
    }
    poses = {
        name: (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([-centers[name], 0.0, 0.0]),
        )
        for name in names
    }

    selected, diagnostics = select_clustered_coverage_indices(
        names,
        names,
        poses,
        n_images=9,
        n_view_clusters=1,
        min_views_per_cluster=2,
        min_baseline_ratio=0.02,
        min_global_pose_distance=0.001,
        max_borrowed_support_distance=1.0,
        min_views_per_sequence=3,
        post_gate_min_views_per_sequence=2,
        sequence_gate_failure_budget=1,
        sequence_coverage_mode="strict",
    )

    assert len(selected) == 9
    for record in diagnostics["sequences"]:
        assert record["selected_count"] == 3
        assert record["gate_resilience"]["failure_scenario_count"] == 3
        assert record["gate_resilience"]["passed"] is True
        assert record["passed"] is True


def test_sequence_resilience_requires_pre_gate_capacity_for_post_gate_failures():
    names = [
        "seq1__frame00000.png",
        "seq1__frame00001.png",
        "seq2__frame00000.png",
        "seq2__frame00001.png",
        "seq3__frame00000.png",
        "seq3__frame00001.png",
    ]

    with pytest.raises(ValueError, match="post-gate support plus"):
        select_clustered_coverage_indices(
            names,
            names,
            _poses(names),
            n_images=6,
            n_view_clusters=1,
            min_views_per_cluster=2,
            min_baseline_ratio=0.02,
            min_global_pose_distance=0.001,
            max_borrowed_support_distance=1.0,
            min_views_per_sequence=2,
            post_gate_min_views_per_sequence=2,
            sequence_gate_failure_budget=1,
            sequence_coverage_mode="strict",
        )


def test_strict_sequence_coverage_rejects_a_missing_temporal_baseline():
    names = [
        "seq1__frame00000.png",
        "seq1__frame00001.png",
        "seq2__frame00000.png",
        "seq2__frame00001.png",
        "seq3__frame00000.png",
        "seq3__frame00001.png",
    ]
    candidates = [name for name in names if name != "seq2__frame00001.png"]

    with pytest.raises(RuntimeError, match="seq2"):
        select_clustered_coverage_indices(
            names,
            candidates,
            _poses(names),
            n_images=5,
            n_view_clusters=1,
            min_views_per_cluster=2,
            min_baseline_ratio=0.02,
            min_global_pose_distance=0.001,
            max_borrowed_support_distance=1.0,
            min_views_per_sequence=2,
            sequence_coverage_mode="strict",
        )


def test_post_gate_audit_does_not_borrow_another_sequence_as_geometry_support(tmp_path: Path):
    names = [
        "seq1__frame00000.png",
        "seq1__frame00001.png",
        "seq2__frame00000.png",
        "seq2__frame00001.png",
        "seq3__frame00000.png",
        "seq3__frame00001.png",
    ]
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    for name in names:
        (scene / "images" / name).touch()
    poses = _poses(names)
    lines = []
    for index, name in enumerate(names, start=1):
        qvec, tvec = poses[name]
        lines.extend([
            f"{index} {' '.join(str(v) for v in qvec)} {' '.join(str(v) for v in tvec)} 1 {name}",
            "",
        ])
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "images.txt").write_text("\n".join(lines))

    selection = {
        "coverage": {
            "view_clusters": 1,
            "min_views_per_cluster": 1,
            "min_baseline_ratio": 0.02,
            "max_borrowed_support_distance": 1.0,
            "sequence_coverage_mode": "strict",
            "min_views_per_sequence": 2,
        }
    }
    gate = {
        "records": [
            {"image_name": name, "rejected": name == "seq1__frame00001.png"}
            for name in names
        ]
    }
    result = audit_coverage(
        selection,
        gate,
        scene,
        minimum_active=1,
        max_target_pose_distance=10.0,
        max_sequence_p95=10.0,
    )

    assert result["checks"]["all_clusters_supported"] is True
    assert result["checks"]["same_sequence_static_support"] is False
    seq1 = next(item for item in result["same_sequence_support"] if item["sequence"] == "seq1")
    assert seq1["active_count"] == 1
    assert seq1["passed"] is False
    assert result["passed"] is False


def test_post_gate_audit_uses_the_surviving_sequence_requirement(tmp_path: Path):
    names = [
        f"seq{sequence}__frame{frame:05d}.png"
        for sequence in range(1, 3)
        for frame in range(3)
    ]
    centers = {
        name: float((sequence - 1) + 0.10 * frame)
        for sequence in range(1, 3)
        for frame, name in enumerate(
            [f"seq{sequence}__frame{index:05d}.png" for index in range(3)]
        )
    }
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    lines = []
    for index, name in enumerate(names, start=1):
        (scene / "images" / name).touch()
        lines.extend([
            f"{index} 1 0 0 0 {-centers[name]} 0 0 1 {name}",
            "",
        ])
    (sparse / "images.txt").write_text("\n".join(lines))
    selection = {
        "coverage": {
            "view_clusters": 1,
            "min_views_per_cluster": 1,
            "min_baseline_ratio": 0.02,
            "max_borrowed_support_distance": 1.0,
            "sequence_coverage_mode": "strict",
            "min_views_per_sequence": 3,
            "post_gate_min_views_per_sequence": 2,
            "sequence_gate_failure_budget": 1,
        }
    }
    gate = {
        "records": [
            {
                "image_name": name,
                "rejected": name == "seq1__frame00002.png",
            }
            for name in names
        ]
    }

    result = audit_coverage(
        selection,
        gate,
        scene,
        minimum_active=1,
        max_target_pose_distance=10.0,
        max_sequence_p95=10.0,
    )

    seq1 = next(item for item in result["same_sequence_support"] if item["sequence"] == "seq1")
    assert seq1["required_count"] == 2
    assert seq1["active_count"] == 2
    assert seq1["passed"] is True
    assert result["checks"]["same_sequence_static_support"] is True
    assert result["passed"] is True
