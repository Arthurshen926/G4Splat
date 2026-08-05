import json
from pathlib import Path
import subprocess
import sys

from outdoor.evidence_store import sha256_file
from scripts.run_cambridge_hybrid_teacher import (
    PIPELINE_VERSION,
    PROFILES,
    _evaluation_is_current,
    _teacher_command,
    _teacher_evaluation_mode,
    _teacher_result_is_current,
    _trainer_implementation_hashes,
)


def test_static_runner_evaluates_complete_canonical_map_not_empty_hybrid_branch():
    assert _teacher_evaluation_mode(
        training_profile="hybrid_handoff_quality",
        reconstruction_target="static",
    ) == "canonical"
    assert _teacher_evaluation_mode(
        training_profile="hybrid_rigid_stage1",
        reconstruction_target="static",
    ) == "rigid"
    assert _teacher_evaluation_mode(
        training_profile="hybrid_quality",
        reconstruction_target="sequence_conditioned_legacy",
    ) == "hybrid"


def test_scene_contract_entrypoint_imports_repo_from_any_working_directory(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build_cambridge_scene_manifest.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_quality_profile_decouples_foliage_coverage_from_ray_bandwidth():
    profile = PROFILES["quality"]

    assert (
        "native-rigid-depth-calibrated-continuous-all-camera-posterior"
        in PIPELINE_VERSION
    )
    assert profile["selected_foliage_views"] == 384
    assert profile["maximum_dense_rays_per_foliage_view"] == 2_048
    assert profile["maximum_dense_rays_total"] == 786_432
    assert profile["minimum_dense_rays_per_foliage_view"] == 512
    assert profile["maximum_bound_rays_per_foliage_view"] == 8_192
    assert profile["maximum_dynamic_births_per_foliage_view"] == 384
    assert (
        profile[
            "maximum_weak_continuous_dynamic_births_per_foliage_view"
        ]
        == 2_048
    )
    assert (
        profile["dynamic_birth_target_source_pixels_per_basis"]
        == 192.0
    )
    assert profile["use_temporal_dav2_witnesses"] is True
    assert profile["temporal_dav2_maximum_gap"] == 12
    assert profile["temporal_dav2_maximum_rays_per_view"] == 2_048
    assert profile["temporal_dav2_maximum_births_per_view"] == 2_048
    assert profile["use_rigid_depth_calibrated_foliage"] is True
    assert profile["rigid_calibration_resolution_scale"] == 0.125
    assert profile["foliage_voxel_size"] == 0.12
    assert profile["maximum_mast3r_pointmap_seeds"] == 240_000
    assert profile["mast3r_pointmap_seeds_per_view"] == 7_000
    assert (
        profile["mast3r_maximum_single_sequence_seed_fraction"]
        == 0.20
    )
    assert profile["maximum_chart_seeds"] == 180_000
    assert profile["chart_seeds_per_view"] == 6_000
    assert profile["maximum_dav2_rigid_seeds"] == 80_000
    assert profile["dav2_rigid_selected_views"] == 512
    assert profile["dav2_rigid_seeds_per_view"] == 384
    assert profile["dav2_rigid_cross_sequence_radius"] == 0.25
    assert profile["training_profile"] == "static_handoff_quality"
    assert profile["rigid_pretrain_iterations"] == 32_000
    assert profile["rigid_pretrain_surface_gaussians"] == 1_200_000
    assert profile["rigid_pretrain_surface_growth_per_event"] == 5_000
    assert profile["rigid_pretrain_densify_until_iteration"] == 16_000
    assert profile["rigid_geometry_gradient_ratio"] == 0.15
    assert profile["maximum_surface_gaussians"] == 1_400_000
    assert profile["maximum_volume_gaussians"] == 2_000_000
    assert profile["maximum_volume_splits"] == 20_000
    assert profile["volume_densify_until_fraction"] == 0.60
    assert profile["surface_densify_until_fraction"] == 0.40
    assert profile["mature_handoff_surface_policy"] == "atlas_residual"
    assert profile["maximum_rigid_completion_seeds"] == 20_000
    assert profile["volume_opacity_lr"] == 4.0e-3
    assert (
        profile[
            "surface_retirement_optical_mass_fraction_per_event"
        ]
        == 0.0025
    )
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_cambridge_hybrid_teacher.py"
    ).read_text(encoding="utf-8")
    assert "initialization_temporal_dav2" in runner
    assert "initialization_rigid_depth_calibrated_temporal_dav2" in runner
    assert "augment_temporal_dav2_foliage.py" in runner
    assert "TEMPORAL_DAV2_AUGMENTATION_VERSION" in runner
    assert "rebuild_foliage_with_rigid_depth.py" in runner
    assert "RIGID_CALIBRATED_INITIALIZATION_VERSION" in runner


def test_fast_profile_shortens_training_without_halving_foliage_evidence():
    fast = PROFILES["fast"]
    quality = PROFILES["quality"]

    assert fast["iterations"] < quality["iterations"]
    assert fast["selected_foliage_views"] == 384
    assert (
        fast["selected_foliage_views"]
        == quality["selected_foliage_views"]
    )
    assert (
        fast["maximum_dense_rays_total"]
        == quality["maximum_dense_rays_total"]
        == 786_432
    )
    assert fast["training_profile"] == "static_handoff_fast"
    assert fast["use_temporal_dav2_witnesses"] is True
    assert fast["temporal_dav2_maximum_rays_per_view"] == 1_024
    assert fast["temporal_dav2_maximum_births_per_view"] == 1_024
    assert (
        fast["foliage_voxel_size"]
        == quality["foliage_voxel_size"]
        == 0.12
    )


def test_staged_teacher_command_closes_scale_and_rigid_lr_contract(tmp_path):
    common = {
        "python": "python",
        "dataset": tmp_path / "dataset",
        "teacher": tmp_path / "teacher",
        "evidence": tmp_path / "evidence",
        "initialization": tmp_path / "initialization",
        "iterations": 24_000,
        "training_profile": "hybrid_rigid_stage1",
        "maximum_surface_gaussians": 400_000,
        "maximum_surface_growth_per_event": 2_500,
        "maximum_volume_gaussians": 800_000,
        "maximum_volume_splits": 1_500,
        "checkpoint_every": 500,
        "geometry_gradient_ratio": 0.25,
        "rgb_images": None,
        "volume_densify_until_iteration": 20_000,
        "surface_densify_until_iteration": 15_000,
    }
    command = _teacher_command(**common)

    assert command[command.index("--maximum-surface-scale") + 1] == "0.5"
    assert (
        command[command.index("--maximum-surface-radius-pixels") + 1]
        == "24"
    )
    assert command[command.index("--position_lr_init") + 1] == "1.6e-5"
    assert (
        command[command.index("--non_position_lr_decay_from") + 1]
        == "18000"
    )
    assert command[command.index("--reconstruction-target") + 1] == "static"
    assert (
        command[command.index("--geometry-gradient-ratio") + 1]
        == "0.25"
    )
    assert (
        command[command.index("--projected-rigid-depth-weight") + 1]
        == "0.0"
    )
    assert command[command.index("--volume-split-radius") + 1] == "2.0"
    assert (
        command[command.index("--maximum-volume-radius-pixels") + 1]
        == "24"
    )
    assert command[command.index("--maximum-skeleton-radius-pixels") + 1] == "12"
    assert command[command.index("--maximum-envelope-radius-pixels") + 1] == "24"
    assert (
        command[command.index("--maximum-static-detail-radius-pixels") + 1]
        == "12"
    )
    assert command[command.index("--view-cache-size") + 1] == "2048"
    assert command[command.index("--image-prefetch-workers") + 1] == "8"
    assert command[command.index("--image-prefetch-depth") + 1] == "32"
    assert (
        command[command.index("--volume-densify-until-iteration") + 1]
        == "20000"
    )
    assert (
        command[command.index("--densify_until_iter") + 1]
        == "15000"
    )
    assert (
        command[
            command.index(
                "--surface-retirement-optical-mass-fraction-per-event"
            )
            + 1
        ]
        == "0.0025"
    )


def test_completed_teacher_requires_current_inputs_code_and_state(tmp_path):
    initialization = tmp_path / "initialization"
    initialization.mkdir()
    surface = initialization / "surface.npz"
    foliage = initialization / "foliage.pth"
    surface.write_bytes(b"surface")
    foliage.write_bytes(b"foliage")
    manifest_path = initialization / "initialization_manifest.json"
    manifest = {
        "version": "test-init-v1",
        "surface_seed": str(surface),
        "foliage_seed": str(foliage),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    state = tmp_path / "teacher.pth"
    state.write_bytes(b"teacher-state")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "evidence_hash": "evidence",
                "iterations": 50,
                "training_profile": "hybrid_quality",
                "maximum_surface_gaussians": 400,
                "maximum_surface_growth_per_event": 25,
                "maximum_volume_gaussians": 800,
                "maximum_volume_splits_per_event": 10,
                "teacher_state": str(state),
                "teacher_state_sha256": sha256_file(state),
                "implementation_hashes": _trainer_implementation_hashes(),
                "training_contract": {
                    "schedule_horizon": 50,
                    "volume_densify_until_iteration": 40,
                    "densify_until_iter": 35,
                    "geometry_gradient_ratio": 0.15,
                    "mature_handoff_surface_policy": "joint",
                    "rigid_background_completion": {
                        "maximum_seeds": 20_000,
                    },
                    "volume_opacity_lr": 4.0e-3,
                    "surface_retirement_optical_mass_fraction_per_event": (
                        0.0025
                    ),
                    "initialization_version": "test-init-v1",
                    "initialization_manifest_sha256": sha256_file(
                        manifest_path
                    ),
                    "surface_seed_sha256": sha256_file(surface),
                    "foliage_seed_sha256": sha256_file(foliage),
                    "surface_warmstart_ply_sha256": None,
                    "surface_warmstart_manifest_sha256": None,
                },
            }
        ),
        encoding="utf-8",
    )
    arguments = {
        "evidence_hash": "evidence",
        "initialization": initialization,
        "iterations": 50,
        "training_profile": "hybrid_quality",
        "maximum_surface_gaussians": 400,
        "maximum_surface_growth_per_event": 25,
        "maximum_volume_gaussians": 800,
        "maximum_volume_splits": 10,
        "geometry_gradient_ratio": 0.15,
        "volume_densify_until_iteration": 40,
        "surface_densify_until_iteration": 35,
    }

    assert _teacher_result_is_current(result_path, **arguments)
    changed = json.loads(result_path.read_text())
    changed["training_contract"]["volume_densify_until_iteration"] = 39
    result_path.write_text(json.dumps(changed), encoding="utf-8")
    assert not _teacher_result_is_current(result_path, **arguments)
    changed["training_contract"]["volume_densify_until_iteration"] = 40
    result_path.write_text(json.dumps(changed), encoding="utf-8")
    state.write_bytes(b"different-state")
    assert not _teacher_result_is_current(result_path, **arguments)


def test_completed_teacher_binds_the_exact_rigid_handoff(tmp_path):
    initialization = tmp_path / "initialization"
    initialization.mkdir()
    surface = initialization / "surface.npz"
    foliage = initialization / "foliage.pth"
    surface.write_bytes(b"surface-seed")
    foliage.write_bytes(b"foliage-seed")
    manifest_path = initialization / "initialization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "test-init-v1",
                "surface_seed": str(surface),
                "foliage_seed": str(foliage),
            }
        ),
        encoding="utf-8",
    )
    warmstart = tmp_path / "rigid.ply"
    handoff = tmp_path / "rigid_surface_handoff.json"
    warmstart.write_bytes(b"rigid-surface")
    handoff.write_text('{"protocol":"native-rigid-surface-handoff-v1"}')
    state = tmp_path / "teacher.pth"
    state.write_bytes(b"teacher-state")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "evidence_hash": "evidence",
                "iterations": 30,
                "training_profile": "hybrid_handoff_quality",
                "maximum_surface_gaussians": 800,
                "maximum_surface_growth_per_event": 50,
                "maximum_volume_gaussians": 1200,
                "maximum_volume_splits_per_event": 30,
                "teacher_state": str(state),
                "teacher_state_sha256": sha256_file(state),
                "implementation_hashes": _trainer_implementation_hashes(),
                "training_contract": {
                    "schedule_horizon": 30,
                    "geometry_gradient_ratio": 0.15,
                    "mature_handoff_surface_policy": "appearance_only",
                    "rigid_background_completion": {
                        "maximum_seeds": 0,
                    },
                    "volume_opacity_lr": 4.0e-3,
                    "surface_retirement_optical_mass_fraction_per_event": (
                        0.0025
                    ),
                    "initialization_version": "test-init-v1",
                    "initialization_manifest_sha256": sha256_file(
                        manifest_path
                    ),
                    "surface_seed_sha256": sha256_file(surface),
                    "foliage_seed_sha256": sha256_file(foliage),
                    "surface_warmstart_ply_sha256": sha256_file(warmstart),
                    "surface_warmstart_manifest_sha256": sha256_file(handoff),
                },
            }
        ),
        encoding="utf-8",
    )
    arguments = {
        "evidence_hash": "evidence",
        "initialization": initialization,
        "iterations": 30,
        "training_profile": "hybrid_handoff_quality",
        "maximum_surface_gaussians": 800,
        "maximum_surface_growth_per_event": 50,
        "maximum_volume_gaussians": 1200,
        "maximum_volume_splits": 30,
        "geometry_gradient_ratio": 0.15,
        "surface_warmstart_ply": warmstart,
        "surface_warmstart_manifest": handoff,
        "mature_handoff_surface_policy": "appearance_only",
        "maximum_rigid_completion_seeds": 0,
        "volume_opacity_lr": 4.0e-3,
    }

    assert _teacher_result_is_current(result_path, **arguments)
    warmstart.write_bytes(b"different-rigid-surface")
    assert not _teacher_result_is_current(result_path, **arguments)


def test_evaluation_requires_exact_teacher_content_and_full_view_count(
    tmp_path,
):
    state = tmp_path / "teacher.pth"
    state.write_bytes(b"teacher-state")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "teacher_state": str(state.resolve()),
                "teacher_state_sha256": sha256_file(state),
                "evaluation_implementation_sha256": sha256_file(
                    Path("scripts/evaluate_hybrid_teacher.py")
                ),
                "evaluation_mode": "hybrid",
                "view_count": 1487,
                "metric_protocol": {
                    "rgb_source": {
                        "image_root": "/rgb",
                        "image_count": 1487,
                        "name_set_sha256": "names",
                        "content_mapping_sha256": "content",
                        "content_bytes": 123,
                        "producer_manifest_sha256": "producer",
                        "target_storage": "shared",
                        "canonical_image_size_wh": [640, 360],
                    },
                    "camera_geometry_sha256": "cameras",
                },
                "student_or_standard_renderer_used": False,
            }
        ),
        encoding="utf-8",
    )

    assert _evaluation_is_current(
        metrics,
        teacher_state=state,
        evaluation_mode="hybrid",
        expected_view_count=1487,
    )
    assert not _evaluation_is_current(
        metrics,
        teacher_state=state,
        evaluation_mode="hybrid",
        expected_view_count=3,
    )
    assert _evaluation_is_current(
        metrics,
        teacher_state=state,
        evaluation_mode="hybrid",
        expected_view_count=1487,
        expected_rgb_source={
            "image_root": "/rgb",
            "image_count": 1487,
            "name_set_sha256": "names",
            "content_mapping_sha256": "content",
            "content_bytes": 123,
            "producer_manifest_sha256": "producer",
            "target_storage": "shared",
            "canonical_image_size_wh": [640, 360],
        },
        expected_camera_geometry_sha256="cameras",
    )
    assert not _evaluation_is_current(
        metrics,
        teacher_state=state,
        evaluation_mode="hybrid",
        expected_view_count=1487,
        expected_rgb_source={
            "image_root": "/rgb",
            "image_count": 1487,
            "name_set_sha256": "names",
            "content_mapping_sha256": "different-content",
            "content_bytes": 123,
            "producer_manifest_sha256": "producer",
            "target_storage": "shared",
            "canonical_image_size_wh": [640, 360],
        },
        expected_camera_geometry_sha256="cameras",
    )
    state.write_bytes(b"new-state")
    assert not _evaluation_is_current(
        metrics,
        teacher_state=state,
        evaluation_mode="hybrid",
        expected_view_count=1487,
    )
