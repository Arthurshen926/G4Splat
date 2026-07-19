from dataclasses import replace
import json
from pathlib import Path

from scripts.run_cambridge_g4splat import (
    CambridgeG4Config,
    audit_command,
    scene_paths,
    selection_command,
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


def test_selection_is_target_kcenter_strict_mask0_without_tree_filter(tmp_path):
    config = _config(tmp_path)
    command = selection_command(scene_paths("StMarysChurch", config), config, 24)

    assert _values_after(command, "--semantic-mask-indices") == ["0"]
    assert _values_after(command, "--quality-mask-indices") == ["0", "1", "2"]
    assert command[command.index("--coverage-objective") + 1] == "target_kcenter"
    assert "--occlusion-mask-indices" not in command
    assert command[command.index("--allowed-name-file") + 1] == str(
        scene_paths("StMarysChurch", config).candidate_allowlist
    )


def test_selection_adds_tree15_filter_when_independent_mask_exists(tmp_path):
    config = _config(tmp_path)
    tree_mask = config.mask_root / "StMarysChurch" / "processed" / "masks_with_tree.pkl"
    tree_mask.parent.mkdir(parents=True)
    tree_mask.touch()
    paths = scene_paths("StMarysChurch", config)

    command = selection_command(paths, config, 24)

    assert command[command.index("--tree-mask-pickle") + 1] == str(tree_mask)
    assert command[command.index("--tree-mask-index") + 1] == "3"
    assert command[command.index("--max-tree-invalid-ratio") + 1] == "0.15"


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
    assert plane_paths.selection_json.name == "chart_selection_n24.json"


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
    assert "--dense_final_only" in command
    assert "--use_color_correction" in command
    assert command[command.index("--color_correction_lr") + 1] == "0.001"
    assert command[command.index("--color_correction_reg") + 1] == "0.01"
    assert "--stop_after_initial_refinement" not in command


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


def test_screen_manifest_recovers_dense_supervision_for_initial_7k(tmp_path):
    from argparse import Namespace
    import json

    dense_dataset = tmp_path / "dense"
    dense_dataset.mkdir()
    output = tmp_path / "run" / "free_gaussians"
    output.mkdir(parents=True)
    (output.parent / "cambridge_g4_manifest.json").write_text(json.dumps({
        "mode": "screen7k",
        "dense_dataset": str(dense_dataset),
    }))
    args = Namespace(
        output_path=str(output),
        dense_data_path=None,
        dense_regul="none",
    )

    apply_screen_manifest_dense_defaults(args)

    assert args.dense_data_path == str(dense_dataset)
    assert args.dense_regul == "strong"


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
