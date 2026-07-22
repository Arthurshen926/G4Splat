from pathlib import Path

import numpy as np

from scripts.select_joint_chart_set import build_joint_selection, correlated_failure_corridors
from scripts.select_quality_aware_charts import JointSelector, select_overlap_neighbours


def test_joint_selection_blocks_all_feedback_and_does_not_pin_accepted(tmp_path: Path):
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    names = [f"seq1__frame{i:05d}.png" for i in range(18)]
    for name in names:
        (scene / "images" / name).touch()
    lines = ["# Image list"]
    for index, name in enumerate(names, start=1):
        lines.extend([f"{index} 1 0 0 0 {-float(index)} 0 0 1 {name}", ""])
    (sparse / "images.txt").write_text("\n".join(lines))
    selection = {
        "scene_path": str(scene),
        "n_images": 4,
        "image_names": [names[0], names[5], names[10], names[15]],
        "candidate_pool_names": names,
        "coverage": {
            "view_clusters": 1,
            "min_views_per_cluster": 1,
            "min_baseline_ratio": 0.0,
            "min_global_pose_distance": 0.01,
            "max_borrowed_support_distance": 1.0,
        },
    }
    reports = [{"records": [{"image_name": names[5], "rejected": True}]}]

    result = build_joint_selection(
        selection,
        reports,
        scene,
        n_images=4,
        rejection_neighbor_radius=1,
        explicit_blocked_names=[names[10]],
    )

    assert names[4] not in result["image_names"]
    assert names[5] not in result["image_names"]
    assert names[6] not in result["image_names"]
    assert names[10] not in result["image_names"]
    assert len(result["image_names"]) == 4
    assert result["alignment_feedback_history"][-1]["mode"] == "joint_full_set_reselection"


def test_joint_reselection_carries_the_same_sequence_support_contract(tmp_path: Path):
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    names = [
        *[f"seq1__frame{i:05d}.png" for i in range(4)],
        *[f"seq2__frame{i:05d}.png" for i in range(4)],
    ]
    for name in names:
        (scene / "images" / name).touch()
    lines = ["# Image list"]
    for index, name in enumerate(names, start=1):
        lines.extend([f"{index} 1 0 0 0 {-float(index)} 0 0 1 {name}", ""])
    (sparse / "images.txt").write_text("\n".join(lines))
    selection = {
        "scene_path": str(scene),
        "n_images": 4,
        "image_names": names[:4],
        "candidate_pool_names": names,
        "candidate_reliability": {
            "weight": 0.03,
            "scores": {name: 0.5 for name in names},
            "selected_scores": {name: 0.5 for name in names[:4]},
        },
        "coverage": {
            "view_clusters": 1,
            "min_views_per_cluster": 1,
            "min_baseline_ratio": 0.01,
            "min_global_pose_distance": 0.0,
            "max_borrowed_support_distance": 1.0,
            "candidate_reliability_weight": 0.03,
            "sequence_coverage_mode": "strict",
            "min_views_per_sequence": 2,
            "post_gate_min_views_per_sequence": 1,
            "sequence_gate_failure_budget": 1,
        },
    }
    reports = [{"records": [{"image_name": names[0], "rejected": True}]}]

    result = build_joint_selection(selection, reports, scene, n_images=4)

    assert names[0] not in result["image_names"]
    sequence_records = result["coverage"]["sequences"]
    assert {record["sequence"] for record in sequence_records} == {"seq1", "seq2"}
    assert all(record["selected_count"] >= 2 for record in sequence_records)
    assert all(record["baseline_satisfied"] for record in sequence_records)


def test_joint_selection_preserves_exact_reference_before_cross_sequence_pose_match(tmp_path: Path):
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    sparse = scene / "sparse" / "0"
    sparse.mkdir(parents=True)
    names = [
        "seq4__frame00225.png",
        "seq12__frame00118.png",
        "seq4__frame00307.png",
    ]
    for name in names:
        (scene / "images" / name).touch()
    # seq12 is deliberately closest in translation to the reference.
    centers = [10.0, 10.001, 12.0]
    lines = ["# Image list"]
    for index, (name, center) in enumerate(zip(names, centers), start=1):
        lines.extend([f"{index} 1 0 0 0 {-center} 0 0 1 {name}", ""])
    (sparse / "images.txt").write_text("\n".join(lines))
    selection = {
        "scene_path": str(scene),
        "image_names": names[:2],
        "candidate_pool_names": names,
        "coverage": {
            "view_clusters": 1,
            "min_views_per_cluster": 1,
            "min_baseline_ratio": 0.0,
            "min_global_pose_distance": 0.0,
            "max_borrowed_support_distance": 1.0,
        },
    }

    result = build_joint_selection(
        selection,
        [],
        scene,
        n_images=2,
        reference_names=[names[0]],
        reference_features={names[0]: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.35])},
        reference_max_pose_distance=10.0,
        preserve_reference_names=True,
    )

    assert names[0] in result["image_names"]
    assert result["joint_selection"]["reference_anchor_assignments"][0]["selected_name"] == names[0]


def test_contiguous_gate_failures_escalate_only_the_local_corridor():
    names = [f"seq4__frame{i:05d}.png" for i in range(120, 140)]
    rejected = {f"seq4__frame{i:05d}.png" for i in range(129, 133)}

    corridors = correlated_failure_corridors(
        names,
        rejected,
        min_run_length=3,
        padding=2,
    )

    assert len(corridors) == 1
    assert corridors[0]["rejected_frames"] == [129, 130, 131, 132]
    assert corridors[0]["blocked_names"] == [
        f"seq4__frame{i:05d}.png" for i in range(127, 135)
    ]


def test_failed_frame_at_quarantine_edge_bridges_the_corridor():
    names = [f"seq4__frame{i:05d}.png" for i in range(120, 140)]
    # 126 is separated from the 129--132 run by precisely the two frames
    # already held out for re-audit.  Both endpoints now have direct evidence.
    rejected = {
        "seq4__frame00126.png",
        "seq4__frame00129.png",
        "seq4__frame00130.png",
        "seq4__frame00131.png",
        "seq4__frame00132.png",
    }

    corridors = correlated_failure_corridors(
        names,
        rejected,
        min_run_length=3,
        padding=2,
        max_bridge_gap=2,
    )

    assert len(corridors) == 1
    assert corridors[0]["rejected_frames"] == [126, 129, 130, 131, 132]
    assert corridors[0]["blocked_names"] == [
        f"seq4__frame{i:05d}.png" for i in range(124, 135)
    ]


def test_quality_selection_requires_redundant_reference_support_when_available():
    """A sparse anchor stays feasible, while a redundant one cannot collapse to one view."""
    selector = JointSelector(
        reliabilities=np.array([0.9, 0.8, 0.7]),
        cells=[np.array([0]), np.array([1]), np.array([2])],
        cell_weights=np.ones(3),
        target_distances=np.zeros((1, 3)),
        centers=np.zeros((3, 3)),
        support_matrix=np.zeros((0, 3), dtype=bool),
        pair_scores=np.zeros((3, 3)),
        reference_matrix=np.array(
            [
                [True, True, False],  # two valid views: require both
                [False, False, True],  # one valid view: cap the requirement at one
            ]
        ),
        reference_names=["redundant", "sparse"],
        min_support=1,
        min_baseline=0.01,
        min_reference_support=2,
    )

    valid, diagnostics = selector.constraints([0, 2])
    assert not valid
    assert diagnostics["references"][0]["available_support_count"] == 2
    assert diagnostics["references"][0]["required_support_count"] == 2
    assert diagnostics["references"][0]["support_count"] == 1
    assert diagnostics["references"][1]["required_support_count"] == 1
    assert diagnostics["references"][1]["passed"]

    valid, diagnostics = selector.constraints([0, 1, 2])
    assert valid
    assert all(item["passed"] for item in diagnostics["references"])


def test_quality_overlap_peer_selection_does_not_backfill_pose_neighbours():
    candidates = np.array([4, 1, 9, 3], dtype=np.int64)
    overlaps = np.array([0.10, 0.55, 0.55, 0.03], dtype=np.float64)

    selected = select_overlap_neighbours(candidates, overlaps, count=3, min_overlap=0.05)

    assert np.array_equal(selected, np.array([1, 9, 4], dtype=np.int64))
