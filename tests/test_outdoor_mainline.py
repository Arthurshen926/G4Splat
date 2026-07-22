from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from outdoor.inverse_depth import fuse_inverse_depth_sources
from outdoor.structure_graph import build_structure_graph_from_samples, triangulation_angle_degrees
from outdoor.structural_selection import select_structural_charts, validate_structural_selection
from matcha.cambridge_training import build_spatial_camera_blocks, fused_inverse_depth_nll
from scripts.run_cambridge_g4splat import scene_paths
from scripts.run_cambridge_outdoor_mainline import (
    MAINLINE_POLICY_VERSION,
    _completed_sfm_can_resume_alignment,
    _mainline_runtime_env,
    _train_with_mainline_contract,
    _reuse_compatible_frontend,
    _serialized_config,
    build_mainline_config,
)


def test_static_structure_graph_requires_multiview_units_and_uses_true_angles():
    samples = [
        np.asarray([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]], dtype=np.float32),
        np.asarray([[10.0, 0.0, 5.0]], dtype=np.float32),
    ]
    confidences = [np.ones(len(points), dtype=np.float32) for points in samples]
    graph = build_structure_graph_from_samples(
        samples,
        confidences,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        bins=8,
        max_units=16,
        min_unit_views=2,
        block_bins=2,
    )

    assert graph["unit_centers"].shape[0] == 2
    assert graph["unit_view_count"].min() >= 2
    assert np.all(graph["unit_support"][2] == 0.0)
    angles = triangulation_angle_degrees(
        np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert float(angles[0]) > 1.0


def test_inverse_depth_fusion_retains_source_provenance_without_fixed_depth_cap():
    fused = fuse_inverse_depth_sources(
        plane_depth=np.asarray([[100.0, 0.0]], dtype=np.float32),
        plane_confidence=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chart_depth=np.asarray([[110.0, 80.0]], dtype=np.float32),
        chart_confidence=np.asarray([[1.0, 1.0]], dtype=np.float32),
        mono_depth=np.asarray([[90.0, 70.0]], dtype=np.float32),
    )

    assert fused["depth"][0, 0] > 50.0
    assert fused["source_bitmask"][0, 0] == 7
    assert fused["source_bitmask"][0, 1] == 6
    assert fused["confidence"][0, 0] > fused["confidence"][0, 1]
    assert np.isfinite(fused["rho_variance"][0, 0])


def test_variance_aware_inverse_depth_loss_downweights_mono_only_evidence():
    rendered = torch.tensor([[1.1, 5.0], [1.1, 1.1]], dtype=torch.float32)
    target = torch.ones((2, 2), dtype=torch.float32)
    variance = torch.ones((2, 2), dtype=torch.float32)
    all_multiview = fused_inverse_depth_nll(
        rendered,
        target,
        variance,
        confidence=torch.ones((2, 2)),
        source_bitmask=torch.tensor([[1, 1], [1, 1]], dtype=torch.uint8),
    )
    second_is_mono_only = fused_inverse_depth_nll(
        rendered,
        target,
        variance,
        confidence=torch.ones((2, 2)),
        source_bitmask=torch.tensor([[1, 4], [1, 1]], dtype=torch.uint8),
    )

    assert torch.isfinite(second_is_mono_only)
    assert second_is_mono_only < all_multiview


def test_spatial_camera_blocks_keep_all_real_views_but_balance_cycles():
    blocks = build_spatial_camera_blocks(
        torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ]),
        bins=2,
    )

    assert sorted(index for block in blocks for index in block) == [0, 1, 2, 3]
    assert len(blocks) == 2
    assert sorted(map(len, blocks)) == [2, 2]


def _write_target_scene(path: Path, names: list[str]) -> None:
    (path / "images").mkdir(parents=True)
    sparse = path / "sparse" / "0"
    sparse.mkdir(parents=True)
    lines = []
    for index, name in enumerate(names, start=1):
        (path / "images" / name).touch()
        lines.extend([f"{index} 1 0 0 0 {-index} 0 0 1 {name}", ""])
    (sparse / "images.txt").write_text("\n".join(lines))


def test_structural_selector_stops_on_coverage_not_fixed_chart_count(tmp_path):
    names = ["a.png", "b.png", "c.png", "d.png"]
    target_scene = tmp_path / "target"
    _write_target_scene(target_scene, names)
    graph_path = tmp_path / "graph.npz"
    support = np.ones((4, 3), dtype=np.float32)
    np.savez_compressed(
        graph_path,
        image_names=np.asarray(names),
        unit_support=support,
        unit_weight=np.asarray([0.4, 0.35, 0.25], dtype=np.float32),
        unit_centers=np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 6.0], [2.0, 0.0, 7.0]], dtype=np.float32),
        camera_centers=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        unit_block_ids=np.asarray([0, 1, 2], dtype=np.int32),
        block_weight=np.asarray([0.4, 0.35, 0.25], dtype=np.float32),
        valid_fractions=np.asarray([0.9, 0.8, 0.85, 0.75], dtype=np.float32),
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({"records": [
        {"image_name": name, "rejected": False, "valid_fraction": 0.9, "relative_p90": 0.1, "gt25_fraction": 0.05}
        for name in names
    ]}))
    output = tmp_path / "selection.json"

    selection = select_structural_charts(
        graph_path,
        gate_path,
        target_scene,
        output,
        min_charts=2,
        max_charts=4,
        target_structural_coverage=0.90,
        target_multiview_coverage=0.70,
        min_block_views=2,
        min_triangulation_angle_degrees=0.5,
        target_triangulation_angle_degrees=5.0,
    )

    assert selection["actual_n_images"] == 3
    assert validate_structural_selection(output, gate_path)["passed"]
    assert selection["structural_selection"]["diagnostics"]["structural_coverage"] >= 0.90


def test_mainline_config_freezes_cameras_and_real_view_topology(tmp_path):
    args = SimpleNamespace(
        datasets_root=tmp_path / "datasets",
        mask_root=tmp_path / "masks",
        output_root=tmp_path / "output",
        depth_checkpoint_dir=tmp_path / "depth",
        run_tag="test",
        broad_charts=56,
        min_charts=24,
        view_clusters=8,
        final_iterations=40_000,
        final_non_position_lr_decay_from=30_000,
        final_non_position_lr_final_mult=0.1,
    )
    config = build_mainline_config(args)

    assert config.strict_calibrated_poses is True
    assert config.per_view_calibrated_intrinsics is True
    assert config.densification_view_policy == "dense_only"
    assert config.min_global_plane_views == 3
    assert config.joint_chart_count is None
    assert config.scene_aligned_see3d_cameras is False
    assert config.preserve_visible_see3d_render is False
    assert config.disable_see3d is True


def test_mainline_reuses_pre_switch_frontend_without_reenabling_see3d(tmp_path):
    args = SimpleNamespace(
        datasets_root=tmp_path / "datasets",
        mask_root=tmp_path / "masks",
        output_root=tmp_path / "output",
        depth_checkpoint_dir=tmp_path / "depth",
        run_tag="test",
        broad_charts=56,
        min_charts=24,
        view_clusters=8,
        final_iterations=40_000,
        final_non_position_lr_decay_from=30_000,
        final_non_position_lr_final_mult=0.1,
    )
    config = build_mainline_config(args)
    default_paths = scene_paths("StMarysChurch", config)
    frontend = (
        config.output_root
        / "runs"
        / "StMarysChurch_g4_qc_n56_strictclean_planeonly_legacy_screen7k_v3"
    )
    frontend.mkdir(parents=True)
    recorded = _serialized_config(config)
    recorded["scene_aligned_see3d_cameras"] = True
    recorded["preserve_visible_see3d_render"] = True
    (frontend / "outdoor_mainline_manifest.json").write_text(json.dumps({
        "policy_version": MAINLINE_POLICY_VERSION,
        "config": recorded,
    }))

    reused = _reuse_compatible_frontend("StMarysChurch", config, default_paths)

    assert reused.screen_output == frontend
    assert config.disable_see3d is True
    assert config.scene_aligned_see3d_cameras is False


def test_mainline_restarts_alignment_from_completed_fixed_camera_sfm(tmp_path):
    screen_output = tmp_path / "screen"
    sparse = screen_output / "mast3r_sfm" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").touch()
    (screen_output / "mast3r_sfm" / "pointmaps").mkdir()

    assert _completed_sfm_can_resume_alignment(screen_output)
    (sparse / "images.bin").unlink()
    assert not _completed_sfm_can_resume_alignment(screen_output)


def test_mainline_runtime_does_not_inject_unsupported_allocator(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    env = _mainline_runtime_env(2)

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env


def test_mainline_contract_enables_block_balanced_real_view_sampling(tmp_path):
    command = _train_with_mainline_contract(
        ["python", "train.py"],
        semantic_manifest=tmp_path / "semantics.json",
    )

    assert command[command.index("--dense-view-sampling-policy") + 1] == "spatial_block_balanced"
    assert command[command.index("--dense-view-block-bins") + 1] == "4"
    assert command[command.index("--plane-min-size-ratio") + 1] == "0.004"
    assert command[command.index("--plane-min-pixels") + 1] == "64"
    assert command[command.index("--max-chart-abs-depth") + 1] == "0"
    assert command[command.index("--max-chart-abs-point") + 1] == "0"
