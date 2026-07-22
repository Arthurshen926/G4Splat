from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts.run_cambridge_g4splat import (
    CambridgeG4Config,
    assert_audit_covers_full_dataset,
    assert_qc_retains_full_dataset,
    assert_valid_chart_selection,
    audit_command,
    joint_selection_command,
    joint_selection_output_path,
    scene_paths,
    selection_command,
    selection_input_provenance,
    train_fit_metric_command,
    train_fit_render_command,
    train_command,
    write_run_manifest,
)
from scripts.refine_free_gaussians import apply_screen_manifest_dense_defaults


def _config(tmp_path: Path) -> CambridgeG4Config:
    return CambridgeG4Config(
        datasets_root=tmp_path / "datasets",
        mask_root=tmp_path / "masks",
        output_root=tmp_path / "output",
        depth_checkpoint_dir=tmp_path / "depth",
    )


def _values_after(command: list[str], flag: str) -> list[str]:
    start = command.index(flag) + 1
    values = []
    for value in command[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def _write_full_train_qc_fixture(tmp_path: Path, names=("a.png", "b.png")):
    """Create the smallest full-train QC scene with real identity checks."""
    config = replace(
        _config(tmp_path),
        requested_charts=1,
        minimum_charts=1,
        view_clusters=1,
        candidate_min_views_per_cluster=1,
        joint_min_views_per_cluster=1,
        min_chart_views_per_sequence=0,
        post_gate_min_chart_views_per_sequence=0,
        chart_sequence_gate_failure_budget=0,
        chart_sequence_coverage_mode="off",
    )
    paths = scene_paths("StMarysChurch", config)
    source_images = paths.full_dataset / "images"
    qc_images = paths.qc_dataset / "images"
    source_images.mkdir(parents=True)
    qc_images.mkdir(parents=True)
    sparse = paths.qc_dataset / "sparse" / "0"
    sparse.mkdir(parents=True)
    source_mapping = {}
    pose_lines = []
    for index, name in enumerate(names, start=1):
        (source_images / name).touch()
        (qc_images / name).touch()
        source_mapping[name] = f"seq/{name}"
        pose_lines.extend([f"{index} 1 0 0 0 0 0 0 1 {name}", ""])
    (paths.full_dataset / "name_mapping.json").write_text(json.dumps(source_mapping))
    (paths.qc_dataset / "name_mapping.json").write_text(json.dumps(source_mapping))
    (sparse / "images.txt").write_text("\n".join(pose_lines))
    (sparse / "images.bin").write_bytes(b"test-poses")
    paths.mask_pickle.parent.mkdir(parents=True, exist_ok=True)
    paths.mask_pickle.write_bytes(b"mask")
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "dataset": str(paths.full_dataset),
        "records": [{"image_name": name, "status": "clean"} for name in names],
    }
    paths.audit_json.write_text(json.dumps(audit))
    paths.candidate_allowlist.write_text("\n".join(names) + "\n")
    (paths.qc_dataset / "qc_manifest.json").write_text(json.dumps({
        "source": str(paths.full_dataset),
        "reconstruction_image_policy": "all_audited_train_images_retained",
        "input_images": len(names),
        "output_images": len(names),
    }))
    return config, paths


def test_selection_uses_static_support_and_same_sequence_coverage(tmp_path):
    config = _config(tmp_path)
    command = selection_command(scene_paths("StMarysChurch", config), config, 24)

    assert _values_after(command, "--semantic-mask-indices") == ["0"]
    assert command[command.index("--max-semantic-invalid-ratio") + 1] == "0.1"
    assert _values_after(command, "--static-support-mask-indices") == ["0", "1", "2"]
    assert command[command.index("--min-static-support-ratio") + 1] == "0.1"
    assert command[command.index("--static-support-ratio-cache") + 1] == str(
        scene_paths("StMarysChurch", config).static_support_ratio_cache
    )
    assert _values_after(command, "--quality-mask-indices") == ["0", "1", "2"]
    assert command[command.index("--min-views-per-cluster") + 1] == "2"
    assert command[command.index("--post-gate-min-views-per-cluster") + 1] == "2"
    assert command[command.index("--reserve-max-pose-distance") + 1] == "0.05"
    assert command[command.index("--reserves-per-vulnerable-support") + 1] == "1"
    assert "--require-gate-replacement-reserves" in command
    assert command[command.index("--quality-score-cache") + 1] == str(
        scene_paths("StMarysChurch", config).quality_score_cache
    )
    assert command[command.index("--quality-score-mask-pickle") + 1] == str(
        scene_paths("StMarysChurch", config).mask_pickle
    )
    assert command[command.index("--quality-candidate-policy") + 1] == "soft_penalty"
    assert command[command.index("--coverage-objective") + 1] == "target_kcenter"
    assert command[command.index("--candidate-reliability-weight") + 1] == "0.03"
    assert command[command.index("--min-views-per-sequence") + 1] == "3"
    assert command[command.index("--post-gate-min-views-per-sequence") + 1] == "2"
    assert command[command.index("--sequence-gate-failure-budget") + 1] == "1"
    assert command[command.index("--sequence-coverage-mode") + 1] == "strict"
    assert "--occlusion-mask-indices" not in command
    assert command[command.index("--allowed-name-file") + 1] == str(
        scene_paths("StMarysChurch", config).candidate_allowlist
    )


def test_selection_uses_tree_as_soft_candidate_penalty_when_independent_mask_exists(tmp_path):
    config = _config(tmp_path)
    tree_mask = config.mask_root / "StMarysChurch" / "processed" / "masks_with_tree.pkl"
    tree_mask.parent.mkdir(parents=True)
    tree_mask.touch()
    paths = scene_paths("StMarysChurch", config)

    command = selection_command(paths, config, 24)

    assert command[command.index("--tree-mask-pickle") + 1] == str(tree_mask)
    assert command[command.index("--tree-mask-index") + 1] == "3"
    assert command[command.index("--max-tree-invalid-ratio") + 1] == "0.15"
    assert command[command.index("--tree-candidate-policy") + 1] == "soft_penalty"
    assert _values_after(command, "--static-support-mask-indices") == ["0", "1", "2", "3"]
    assert command[command.index("--quality-score-mask-pickle") + 1] == str(tree_mask)


def test_audit_command_only_passes_supported_mask_policy_arguments(tmp_path):
    config = _config(tmp_path)
    command = audit_command(scene_paths("StMarysChurch", config), config)

    assert "--quality-score-cache" not in command
    assert _values_after(command, "--quality-mask-indices") == ["0", "1", "2"]
    assert command[command.index("--tree-mask-index") + 1] == "-1"


def test_run_paths_isolate_plane_only_from_pseudo_ablation(tmp_path):
    plane_config = _config(tmp_path)
    pseudo_config = replace(
        plane_config,
        pseudo_initialization_mode="inpaint_only",
        pseudo_geometry_mask_mode="inpaint_only",
        pseudo_rgb_weight=0.01,
    )

    plane_paths = scene_paths("StMarysChurch", plane_config)
    pseudo_paths = scene_paths("StMarysChurch", pseudo_config)

    assert "planeonly" in plane_paths.full_output.name
    assert "pseudo" in pseudo_paths.full_output.name
    assert plane_paths.full_output != pseudo_paths.full_output
    assert plane_paths.selection_json.name == "chart_selection_n24_clusters8_support2.json"


def test_full_train_uses_explicit_charts_all_qc_dense_views_and_30k_final(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)
    selection = {"image_idx": [2, 7, 11]}

    command = train_command(paths, config, selection, screen_only=False)

    assert _values_after(command, "--image_idx") == ["2", "7", "11"]
    assert command[command.index("--dense_data_path") + 1] == str(paths.qc_dataset)
    assert command[command.index("--dense_regul") + 1] == "strong_decay"
    assert command[command.index("--final_free_gaussians_config") + 1] == "long"
    assert command[command.index("--final_free_gaussians_iterations") + 1] == "30000"
    assert command[command.index("--final_non_position_lr_decay_from") + 1] == "30000"
    assert command[command.index("--final_non_position_lr_final_mult") + 1] == "0.1"
    assert "--gate_aligned_chart_conflicts" in command
    assert command[command.index("--aligned_chart_min_valid_fraction") + 1] == "0.5"
    assert "--gate_plane_refinement" in command
    assert command[command.index("--min_global_plane_views") + 1] == "2"
    assert _values_after(command, "--cambridge_mask_indices") == ["0", "1", "2"]
    assert _values_after(command, "--cambridge_geometry_mask_indices") == ["0", "1", "2"]
    assert _values_after(command, "--cambridge_alpha_mask_indices") == ["1"]
    assert command[command.index("--semantic_alpha_weight") + 1] == "0.01"
    assert command[command.index("--rgb-supervision-profile") + 1] == "g4_tree_masked"
    assert command[command.index("--rgb-sampling-policy") + 1] == "all_train_importance"
    assert command[command.index("--chart-geometry-prior-weight") + 1] == "1.0"
    assert command[command.index("--warp_downsample_pixel_grid_size") + 1] == "2"
    assert "--downweight_input_view_color_loss" not in command
    assert "--scene_aligned_see3d_cameras" in command
    assert "--preserve_visible_see3d_render" in command
    assert command[command.index("--pseudo_initialization_mode") + 1] == "none"
    assert command[command.index("--pseudo_geometry_mask_mode") + 1] == "none"
    assert command[command.index("--pseudo_rgb_weight") + 1] == "0.0"
    assert command[command.index("--pseudo_geometry_weight") + 1] == "0.0"
    assert command[command.index("--pseudo_geometry_final_weight") + 1] == "0.0"
    assert "--white_background" in command
    assert "--dense_final_only" not in command
    assert "--use_color_correction" in command
    assert command[command.index("--color_correction_lr") + 1] == "0.001"
    assert command[command.index("--color_correction_reg") + 1] == "0.01"
    assert "--stop_after_initial_refinement" not in command


def test_strict_calibrated_pose_mode_is_an_explicit_frontend_variable(tmp_path):
    legacy = _config(tmp_path)
    strict = replace(legacy, strict_calibrated_poses=True)
    selection = {"image_idx": [2, 7]}

    legacy_command = train_command(
        scene_paths("StMarysChurch", legacy), legacy, selection, screen_only=True
    )
    strict_command = train_command(
        scene_paths("StMarysChurch", strict), strict, selection, screen_only=True
    )

    assert "--strict_calibrated_poses" not in legacy_command
    assert "--strict_calibrated_poses" in strict_command
    assert scene_paths("StMarysChurch", legacy).screen_output != scene_paths(
        "StMarysChurch", strict
    ).screen_output


def test_per_view_intrinsics_is_an_explicit_frontend_variable(tmp_path):
    strict = replace(_config(tmp_path), strict_calibrated_poses=True)
    per_view = replace(strict, per_view_calibrated_intrinsics=True)
    selection = {"image_idx": [2, 7]}

    strict_command = train_command(
        scene_paths("StMarysChurch", strict), strict, selection, screen_only=True
    )
    per_view_command = train_command(
        scene_paths("StMarysChurch", per_view), per_view, selection, screen_only=True
    )

    assert "--per_view_calibrated_intrinsics" not in strict_command
    assert "--per_view_calibrated_intrinsics" in per_view_command
    assert scene_paths("StMarysChurch", strict).screen_output != scene_paths(
        "StMarysChurch", per_view
    ).screen_output


def test_runner_forwards_fixed_chart_alignment_seed(tmp_path):
    config = replace(_config(tmp_path), chart_alignment_seed=17)
    command = train_command(
        scene_paths("StMarysChurch", config), config, {"image_idx": [2, 7]}, screen_only=True
    )

    index = command.index("--chart-alignment-seed")
    assert command[index + 1] == "17"


def test_frontend_alignment_appends_only_verified_gate_replacement_reserves(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)
    selection = {
        "image_idx": [2, 7, 11],
        "gate_replacement_reserves": {"alignment_image_idx": [2, 5, 7, 11]},
    }

    command = train_command(paths, config, selection, screen_only=True)

    assert _values_after(command, "--image_idx") == ["2", "5", "7", "11"]


def test_chart_only_screen_is_an_explicit_single_variable_ablation(tmp_path):
    config = replace(_config(tmp_path), dense_final_only=True)
    paths = scene_paths("StMarysChurch", config)

    command = train_command(paths, config, {"image_idx": [2, 7]}, screen_only=True)

    assert "--dense_final_only" in command
    assert command[command.index("--dense_data_path") + 1] == str(paths.qc_dataset)


def test_ulfloc_rgb_profile_changes_only_the_explicit_rgb_protocol_flag(tmp_path):
    config = replace(_config(tmp_path), rgb_supervision_profile="ulfloc_legacy")
    paths = scene_paths("StMarysChurch", config)

    command = train_command(paths, config, {"image_idx": [2, 7]}, screen_only=True)

    assert command[command.index("--rgb-supervision-profile") + 1] == "ulfloc_legacy"
    assert _values_after(command, "--cambridge_geometry_mask_indices") == ["0", "1", "2"]


def test_full_rgb_profile_changes_only_the_explicit_rgb_protocol_flag(tmp_path):
    config = replace(_config(tmp_path), rgb_supervision_profile="full_rgb")
    paths = scene_paths("StMarysChurch", config)

    command = train_command(paths, config, {"image_idx": [2, 7]}, screen_only=True)

    assert command[command.index("--rgb-supervision-profile") + 1] == "full_rgb"
    assert _values_after(command, "--cambridge_geometry_mask_indices") == ["0", "1", "2"]
    assert command[command.index("--semantic_alpha_weight") + 1] == "0.01"


def test_rgb_sampling_policy_is_an_explicit_single_variable(tmp_path):
    config = replace(_config(tmp_path), rgb_sampling_policy="legacy_interleaved")
    command = train_command(
        scene_paths("StMarysChurch", config), config, {"image_idx": [2, 7]}, screen_only=True
    )

    assert command[command.index("--rgb-sampling-policy") + 1] == "legacy_interleaved"


def test_post_gate_joint_selection_is_explicit_and_changes_frontend_identity(tmp_path):
    base_config = replace(_config(tmp_path), requested_charts=40, minimum_charts=24)
    config = replace(base_config, joint_chart_count=24)
    paths = scene_paths("StMarysChurch", config)
    base_paths = scene_paths("StMarysChurch", base_config)

    command = joint_selection_command(paths, config, output=paths.screen_output)
    output_path = joint_selection_output_path(paths.screen_output, config)

    assert paths.screen_output != base_paths.screen_output
    assert command[command.index("--counts") + 1] == "24"
    assert command[command.index("--min-pose-support") + 1] == "2"
    assert command[command.index("--min-sequence-support") + 1] == "2"
    assert command[command.index("--neighbor-selection") + 1] == "overlap"
    assert command[command.index("--selection") + 1] == str(paths.selection_json)
    assert command[command.index("--target-scene-path") + 1] == str(paths.qc_dataset)
    assert _values_after(command, "--mask-indices") == ["0", "1", "2"]
    assert output_path.name == "chart_selection_quality_n24.json"

    tree_mask = config.mask_root / "StMarysChurch" / "processed" / "masks_with_tree.pkl"
    tree_mask.parent.mkdir(parents=True)
    tree_mask.touch()
    tree_paths = scene_paths("StMarysChurch", config)
    tree_command = joint_selection_command(tree_paths, config, output=tree_paths.screen_output)
    assert tree_command[tree_command.index("--mask-pickle") + 1] == str(tree_mask)
    assert _values_after(tree_command, "--mask-indices") == ["0", "1", "2", "3"]


def test_candidate_reserve_and_post_gate_support_are_independent(tmp_path):
    config = replace(
        _config(tmp_path),
        requested_charts=48,
        minimum_charts=32,
        candidate_min_views_per_cluster=4,
        joint_min_views_per_cluster=2,
        joint_chart_count=24,
    )
    paths = scene_paths("StMarysChurch", config)

    candidate_command = selection_command(paths, config, 48)
    joint_command = joint_selection_command(paths, config, output=paths.screen_output)

    assert candidate_command[candidate_command.index("--min-views-per-cluster") + 1] == "4"
    assert candidate_command[candidate_command.index("--post-gate-min-views-per-cluster") + 1] == "2"
    assert candidate_command[candidate_command.index("--reserves-per-vulnerable-support") + 1] == "1"
    assert joint_command[joint_command.index("--min-pose-support") + 1] == "2"
    assert joint_command[joint_command.index("--min-sequence-support") + 1] == "2"
    assert paths.selection_json.name == "chart_selection_n48_clusters8_support4.json"


def test_full_train_audit_rejects_a_silent_subset(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)
    image_dir = paths.full_dataset / "images"
    image_dir.mkdir(parents=True)
    for name in ("seq1__frame00001.png", "seq1__frame00002.png"):
        (image_dir / name).touch()

    audit = {
        "dataset": str(paths.full_dataset),
        "records": [{"image_name": "seq1__frame00001.png", "status": "clean"}],
    }
    with pytest.raises(RuntimeError, match="does not cover"):
        assert_audit_covers_full_dataset(paths, audit)


def test_qc_rejects_same_count_but_replaced_camera_identity(tmp_path):
    config, paths = _write_full_train_qc_fixture(tmp_path)
    qc_images = paths.qc_dataset / "images"
    (qc_images / "b.png").unlink()
    (qc_images / "replacement.png").touch()

    manifest = json.loads((paths.qc_dataset / "qc_manifest.json").read_text())
    with pytest.raises(RuntimeError, match="missing_qc"):
        assert_qc_retains_full_dataset(paths, manifest)


def test_chart_selection_provenance_binds_qc_pose_candidate_and_coverage(tmp_path):
    config, paths = _write_full_train_qc_fixture(tmp_path)
    provenance = selection_input_provenance(paths, config)
    selection = {
        "image_idx": [0],
        "image_names": ["a.png"],
        "n_images": 1,
        "actual_n_images": 1,
        "coverage": {
            "coverage_objective": "target_kcenter",
            "min_views_per_cluster": 1,
            "candidate_reliability_weight": config.chart_candidate_reliability_weight,
            "min_views_per_sequence": config.min_chart_views_per_sequence,
            "post_gate_min_views_per_sequence": config.post_gate_min_chart_views_per_sequence,
            "sequence_gate_failure_budget": config.chart_sequence_gate_failure_budget,
            "sequence_coverage_mode": config.chart_sequence_coverage_mode,
            "sequences": [],
            "clusters": [{"cluster": 0, "coverage_satisfied": True}],
        },
        "static_support_filter": {
            "mask_indices": [0, 1, 2],
            "min_support_ratio": config.min_chart_static_support_ratio,
        },
        "quality_filter": {"candidate_policy": config.chart_quality_candidate_policy},
        "allowed_name_filter": {"selected_all_allowed": True},
        "input_provenance": provenance,
    }

    assert_valid_chart_selection(
        paths,
        config,
        selection,
        expected_provenance=provenance,
    )

    (paths.qc_dataset / "sparse" / "0" / "images.bin").write_bytes(b"changed-poses")
    changed_provenance = selection_input_provenance(paths, config)
    with pytest.raises(RuntimeError, match="input provenance"):
        assert_valid_chart_selection(
            paths,
            config,
            selection,
            expected_provenance=changed_provenance,
        )


def test_feedback_selection_accepts_cluster_adaptive_joint_gate_certificate(tmp_path):
    """A feedback gate may expose correlated failures in just one pose cluster."""
    config, paths = _write_full_train_qc_fixture(tmp_path, names=("a.png", "b.png", "c.png"))
    config = replace(
        config,
        requested_charts=2,
        minimum_charts=2,
        joint_min_views_per_cluster=2,
    )
    provenance = selection_input_provenance(paths, config)
    selection = {
        "image_idx": [0, 1],
        "image_names": ["a.png", "b.png"],
        "n_images": 2,
        "actual_n_images": 2,
        "coverage": {
            "coverage_objective": "target_kcenter",
            "min_views_per_cluster": 1,
            "candidate_reliability_weight": config.chart_candidate_reliability_weight,
            "min_views_per_sequence": config.min_chart_views_per_sequence,
            "post_gate_min_views_per_sequence": config.post_gate_min_chart_views_per_sequence,
            "sequence_gate_failure_budget": config.chart_sequence_gate_failure_budget,
            "sequence_coverage_mode": config.chart_sequence_coverage_mode,
            "sequences": [],
            "clusters": [{"cluster": 0, "coverage_satisfied": True}],
        },
        "static_support_filter": {
            "mask_indices": [0, 1, 2],
            "min_support_ratio": config.min_chart_static_support_ratio,
        },
        "quality_filter": {"candidate_policy": config.chart_quality_candidate_policy},
        "allowed_name_filter": {"selected_all_allowed": True},
        "candidate_pool_names": ["a.png", "b.png", "c.png"],
        "gate_replacement_reserves": {
            "method": "baseline_primary_plus_joint_gate_resilience",
            "post_gate_min_views_per_cluster": 2,
            "min_baseline_ratio": config.min_baseline_ratio,
            "reserve_max_pose_distance": config.min_global_pose_distance,
            "gate_failure_budget": 1,
            "gate_failure_budget_by_cluster": {"0": 2},
            "certified_for_gate_failure_budget": True,
            "reserve_image_idx": [2],
            "reserve_image_names": ["c.png"],
            "alignment_image_idx": [0, 1, 2],
            "alignment_image_names": ["a.png", "b.png", "c.png"],
        },
        "input_provenance": provenance,
    }

    assert_valid_chart_selection(
        paths,
        config,
        selection,
        expected_provenance=provenance,
    )


def test_final_schedule_is_explicit_and_changes_run_identity(tmp_path):
    default_config = _config(tmp_path)
    config = replace(
        default_config,
        final_iterations=80_000,
        final_non_position_lr_decay_from=30_000,
        final_non_position_lr_final_mult=0.1,
    )
    paths = scene_paths("StMarysChurch", config)
    default_paths = scene_paths("StMarysChurch", default_config)
    command = train_command(paths, config, {"image_idx": [2, 7]}, screen_only=False)

    assert "full80k" in paths.full_output.name
    assert paths.full_output != default_paths.full_output
    assert paths.screen_output == default_paths.screen_output
    assert command[command.index("--final_free_gaussians_iterations") + 1] == "80000"
    assert command[command.index("--final_non_position_lr_decay_from") + 1] == "30000"
    assert command[command.index("--final_non_position_lr_final_mult") + 1] == "0.1"

    render_command = train_fit_render_command(paths, config, screen_only=False)
    metric_command = train_fit_metric_command(paths, screen_only=False, config=config)
    assert render_command[render_command.index("--iteration") + 1] == "80000"
    assert "ours_80000" in metric_command[metric_command.index("--output") + 1]


def test_inpaint_only_pseudo_ablation_is_forwarded_explicitly(tmp_path):
    config = replace(
        _config(tmp_path),
        pseudo_initialization_mode="inpaint_only",
        pseudo_geometry_mask_mode="inpaint_only",
        pseudo_rgb_weight=0.01,
        pseudo_geometry_weight=0.05,
        pseudo_geometry_final_weight=0.005,
    )
    paths = scene_paths("StMarysChurch", config)

    command = train_command(paths, config, {"image_idx": [2, 7]}, screen_only=True)

    assert command[command.index("--pseudo_initialization_mode") + 1] == "inpaint_only"
    assert command[command.index("--pseudo_geometry_mask_mode") + 1] == "inpaint_only"
    assert command[command.index("--pseudo_rgb_weight") + 1] == "0.01"
    assert command[command.index("--pseudo_geometry_weight") + 1] == "0.05"
    assert command[command.index("--pseudo_geometry_final_weight") + 1] == "0.005"


def test_run_manifest_records_protected_pseudo_policy(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)

    manifest_path = write_run_manifest(
        "StMarysChurch",
        paths,
        config,
        {"image_idx": [2, 7]},
        ["python", "train.py"],
        screen_only=True,
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["mask_policy"]["preserve_visible_see3d_render"] is True
    assert manifest["mask_policy"]["pseudo_initialization_mode"] == "none"
    assert manifest["mask_policy"]["pseudo_geometry_mask_mode"] == "none"
    assert manifest["mask_policy"]["pseudo_rgb_weight"] == 0.0


def test_screen_train_stops_after_initial_7k_pass(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("GreatCourt", config)

    command = train_command(paths, config, {"image_idx": [0]}, screen_only=True)

    assert "--stop_after_initial_refinement" in command
    assert command[command.index("--free_gaussians_config") + 1] == "default"


def test_screen_schedule_config_is_forwarded_and_changes_screen_identity(tmp_path):
    default_config = _config(tmp_path)
    long_screen_config = replace(
        default_config,
        screen_free_gaussians_config="screen_long7k",
    )
    default_paths = scene_paths("GreatCourt", default_config)
    long_paths = scene_paths("GreatCourt", long_screen_config)

    command = train_command(
        long_paths,
        long_screen_config,
        {"image_idx": [0]},
        screen_only=True,
    )

    assert long_paths.screen_output != default_paths.screen_output
    assert command[command.index("--free_gaussians_config") + 1] == "screen_long7k"


def test_frontend_command_stops_after_alignment_gate(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("GreatCourt", config)

    command = train_command(
        paths,
        config,
        {"image_idx": [0, 2]},
        screen_only=True,
        stop_after_alignment_gate=True,
    )

    assert "--stop_after_alignment_gate" in command


def test_full_train_can_continue_without_repeating_frontend(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)

    command = train_command(
        paths,
        config,
        {"image_idx": [3, 8]},
        screen_only=False,
        continue_after_initial=True,
    )

    assert "--continue_after_initial_refinement" in command
    assert "--stop_after_initial_refinement" not in command


def test_full_train_can_resume_after_see3d_plane_stage(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)

    command = train_command(
        paths,
        config,
        {"image_idx": [3, 8]},
        screen_only=False,
        continue_after_see3d_plane_stage=1,
    )

    assert command[command.index("--continue_after_see3d_plane_stage") + 1] == "1"
    assert "--continue_after_initial_refinement" not in command


def test_screen_train_can_resume_after_completed_sfm(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)

    command = train_command(
        paths,
        config,
        {"image_idx": [3, 8]},
        screen_only=True,
        continue_after_sfm=True,
    )

    assert "--continue_after_sfm" in command
    assert "--stop_after_initial_refinement" in command


def test_screen_train_can_resume_after_completed_alignment(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)

    command = train_command(
        paths,
        config,
        {"image_idx": [3, 8]},
        screen_only=True,
        continue_after_alignment=True,
    )

    assert "--continue_after_alignment" in command
    assert "--continue_after_sfm" not in command
    assert "--stop_after_initial_refinement" in command


def test_screen_train_can_resume_after_plane_refinement(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)

    command = train_command(
        paths,
        config,
        {"image_idx": [3, 8]},
        screen_only=True,
        continue_after_plane=True,
    )

    assert "--continue_after_plane_refinement" in command
    assert "--continue_after_alignment" not in command
    assert "--continue_after_sfm" not in command
    assert "--stop_after_initial_refinement" in command


def test_train_fit_evaluation_uses_all_train_and_static_mask_policy(tmp_path):
    config = _config(tmp_path)
    paths = scene_paths("StMarysChurch", config)

    render_command = train_fit_render_command(paths, config, screen_only=True)
    metric_command = train_fit_metric_command(paths, screen_only=True)

    assert render_command[render_command.index("--iteration") + 1] == "7000"
    assert render_command[render_command.index("-s") + 1] == str(paths.qc_dataset)
    assert render_command[render_command.index("-m") + 1].endswith("/free_gaussians")
    assert "--skip_mesh" in render_command
    assert "--rgb_only" in render_command
    assert "--white_background" in render_command
    assert metric_command[metric_command.index("--mask-pickle") + 1] == str(paths.mask_pickle)
    assert metric_command[metric_command.index("--dataset-path") + 1] == str(paths.qc_dataset)


def test_screen_manifest_recovers_dense_supervision_only_when_explicitly_enabled(tmp_path):
    from argparse import Namespace
    import json

    dense_dataset = tmp_path / "dense"
    dense_dataset.mkdir()
    output = tmp_path / "run" / "free_gaussians"
    output.mkdir(parents=True)
    (output.parent / "cambridge_g4_manifest.json").write_text(json.dumps({
        "mode": "screen7k",
        "dense_dataset": str(dense_dataset),
        "config": {"dense_final_only": False},
    }))
    args = Namespace(
        output_path=str(output),
        dense_data_path=None,
        dense_regul="none",
    )

    apply_screen_manifest_dense_defaults(args)

    assert args.dense_data_path == str(dense_dataset)
    assert args.dense_regul == "strong"


def test_screen_manifest_preserves_explicit_chart_only_policy(tmp_path):
    from argparse import Namespace
    import json

    dense_dataset = tmp_path / "dense"
    dense_dataset.mkdir()
    output = tmp_path / "run" / "free_gaussians"
    output.mkdir(parents=True)
    (output.parent / "cambridge_g4_manifest.json").write_text(json.dumps({
        "mode": "screen7k",
        "dense_dataset": str(dense_dataset),
        "config": {"dense_final_only": True},
    }))
    args = Namespace(
        output_path=str(output),
        dense_data_path=None,
        dense_regul="none",
    )

    apply_screen_manifest_dense_defaults(args)

    assert args.dense_data_path is None
    assert args.dense_regul == "none"


def test_screen_manifest_recovers_color_correction_for_running_parent(tmp_path):
    from argparse import Namespace
    import json

    dense_dataset = tmp_path / "dense"
    dense_dataset.mkdir()
    output = tmp_path / "run" / "free_gaussians"
    output.mkdir(parents=True)
    (output.parent / "cambridge_g4_manifest.json").write_text(json.dumps({
        "mode": "screen7k",
        "dense_dataset": str(dense_dataset),
        "config": {
            "use_color_correction": True,
            "color_correction_lr": 5e-4,
            "color_correction_reg": 2e-2,
        },
    }))
    args = Namespace(
        output_path=str(output),
        dense_data_path=str(dense_dataset),
        dense_regul="strong",
        use_color_correction=False,
        color_correction_lr=1e-3,
        color_correction_reg=1e-2,
    )

    apply_screen_manifest_dense_defaults(args)

    assert args.use_color_correction is True
    assert args.color_correction_lr == 5e-4
    assert args.color_correction_reg == 2e-2
