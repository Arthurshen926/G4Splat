import copy
import random
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from matcha.cambridge_training import PerImageAffineColorCorrection, opacity_reset_due
from scripts.train_standard_full_2dgs import (
    _affine_color_audit,
    _evaluate,
    _capture_rng_state,
    _contract_mismatches,
    _fixed_camera_schedule,
    _implementation_audit,
    _loaded_camera_filenames,
    _mask_resolution_audit,
    _optimization_audit,
    _scene_input_ply_audit,
    _source_control_manifest_path,
    _staged_names,
    _rng_state_byte_tensor_on_cpu,
    _restore_color_correction_state,
    _restore_rng_state,
    _training_state_contract,
    _ulfloc_clean_main_camera_schedule,
)


class _Lookup:
    def __init__(self, shape):
        self.masks = {
            "source.png": tuple(torch.ones(shape, dtype=torch.bool) for _ in range(3))
        }

    @staticmethod
    def source_name_for(_image_name):
        return "source.png"


def test_mask_resolution_audit_distinguishes_native_and_resampled_protocols():
    view = SimpleNamespace(
        image_name="staged.png",
        original_image=torch.zeros((3, 540, 960)),
    )

    native = _mask_resolution_audit(_Lookup((540, 960)), [view], mask_indices=(0, 1, 2))
    resampled = _mask_resolution_audit(_Lookup((360, 640)), [view], mask_indices=(0, 1, 2))

    assert native["native_pixel_geometry"]
    assert native["contract"] == "native_mask_pixels"
    assert not resampled["native_pixel_geometry"]
    assert resampled["views_requiring_mask_resampling"] == 1


def test_standard_control_exposes_an_explicit_ply_warm_start_contract():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    assert '"--init-ply"' in source
    assert '"initialization": (' in source
    assert 'else ("ply_warm_start" if init_ply is not None else "sparse_pcd")' in source
    assert '"init_ply_sha256": init_ply_digest' in source
    assert '"effective_initialization_ply_sha256"' in source
    assert '"cambridge_mask_pickle_sha256"' in source
    assert '"source_control_manifest_sha256"' in source
    assert 'input_audit["init_gaussians"] = int(gaussians.get_xyz.shape[0])' in source
    assert '"fresh_for_this_control"' in source
    assert 'gaussians.set_mip_filter(False)' in source


def test_training_state_fork_restores_all_python_numpy_and_torch_rngs():
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    state = _capture_rng_state()

    expected = (random.random(), np.random.random(), torch.rand(4))
    _restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(4))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_training_state_rng_bytes_are_normalized_after_cuda_mapped_load():
    state = torch.get_rng_state()
    normalized = _rng_state_byte_tensor_on_cpu(state, field="torch_cpu")

    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.uint8
    assert torch.equal(normalized, state)

    # ``torch.load(..., map_location='cuda')`` is what exposed this bug in a
    # real state-fork smoke run.  Exercise the same device transition when a
    # CUDA test worker is available while keeping the test runnable on CPU.
    if torch.cuda.is_available():
        remapped = state.cuda()
        restored = _rng_state_byte_tensor_on_cpu(remapped, field="torch_cpu")
        assert restored.device.type == "cpu"
        assert torch.equal(restored, state)


def test_affine_state_restore_includes_parameters_and_adam_moments():
    correction = PerImageAffineColorCorrection(["a", "b"])
    optimizer = torch.optim.Adam(correction.parameters(), lr=1e-3)
    correction(torch.ones((3, 1, 1)), "b").sum().backward()
    optimizer.step()

    payload = {
        "color_correction": {
            "model_state": copy.deepcopy(correction.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        }
    }
    restored = PerImageAffineColorCorrection(["a", "b"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    _restore_color_correction_state(
        payload,
        color_correction=restored,
        color_correction_optimizer=restored_optimizer,
    )

    assert torch.equal(restored.log_scales, correction.log_scales)
    assert torch.equal(restored.biases, correction.biases)
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_affine_color_audit_records_channelwise_bounds():
    correction = PerImageAffineColorCorrection(["a", "b"])
    with torch.no_grad():
        correction.log_scales[1, :, 0, 0] = torch.log(torch.tensor([2.0, 3.0, 4.0]))
        correction.biases[1, :, 0, 0] = torch.tensor([0.1, 0.2, 0.3])

    audit = _affine_color_audit(correction)

    assert audit["camera_count"] == 2
    assert np.allclose(audit["scale_min_rgb"], [1.0, 1.0, 1.0])
    assert np.allclose(audit["scale_max_rgb"], [2.0, 3.0, 4.0])
    assert np.allclose(audit["bias_max_rgb"], [0.1, 0.2, 0.3])
    assert audit["identity_regularization"] > 0.0


def test_training_state_contract_locks_data_camera_stream_and_implementation_but_not_branch_schedule():
    input_audit = {
        "source_path": "/dataset",
        "image_count": 1487,
        "colmap_camera_count": 1487,
        "loaded_camera_count": 1487,
        "source_split_count": 1487,
        "camera_set_sha256_input": "camera-set",
        "name_mapping_sha256": "mapping",
        "source_control_manifest_sha256": "adapter",
        "scene_sparse_ply_sha256": "ply",
        "scene_sparse_ply_bytes": 123,
        "supervision_profile": "ulfloc_masked",
        "cambridge_mask_pickle_sha256": "masks",
        "cambridge_mask_dataset_path": "/dataset",
        "white_background": True,
        "renderer": "diff_surfel_rasterization",
        "mip_filter": "disabled",
        "implementation": {"files": {"trainer.py": {"sha256": "code"}}},
    }
    schedule = {
        "protocol": "clean_ulfloc_main_scene_shuffle_randint_pop_v1",
        "seed": 0,
        "scheduled_iterations": 40000,
        "training_camera_count": 1487,
        "sequence_sha256": "sequence",
    }
    contract = _training_state_contract(
        input_audit, schedule, declared_total_iterations=40000
    )

    assert _contract_mismatches(contract, copy.deepcopy(contract)) == []
    changed_stream = copy.deepcopy(contract)
    changed_stream["camera_schedule"]["sequence_sha256"] = "other"
    assert any("sequence_sha256" in message for message in _contract_mismatches(contract, changed_stream))

    # LR settings deliberately live outside the immutable continuation
    # contract so a branch may vary exactly that one post-checkpoint factor.
    changed_branch_hyperparameter = copy.deepcopy(input_audit)
    changed_branch_hyperparameter["optimization"] = {
        "non_position_schedule": {"decay_from_iter": 30000, "final_multiplier": 0.1}
    }
    assert _training_state_contract(
        changed_branch_hyperparameter, schedule, declared_total_iterations=40000
    ) == contract


def test_scene_sparse_initialization_is_fingerprinted_even_without_warm_start(tmp_path):
    input_ply = tmp_path / "input.ply"
    input_ply.write_bytes(b"sparse-ply")

    audit = _scene_input_ply_audit(tmp_path)

    assert audit["scene_sparse_ply"] == str(input_ply.resolve())
    assert audit["scene_sparse_ply_sha256"] == (
        "7d2dc0023aea4db9d5239ed0c244e26036f16a64b1ab55c093998d283980d66a"
    )
    assert audit["scene_sparse_ply_bytes"] == len(b"sparse-ply")


def test_external_adapter_input_manifest_is_the_preferred_provenance_file(tmp_path):
    legacy = tmp_path / "external_control_manifest.json"
    legacy.write_text("legacy")
    adapter = tmp_path / "input_manifest.json"
    adapter.write_text("adapter")

    assert _source_control_manifest_path(tmp_path) == adapter


def test_standard_control_records_the_effective_native_prune_threshold():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    # ULF-Loc/STDLoc configuration files declare 0.05 but their released 2D
    # loops pass literal 0.005.  The standard control must make the executed
    # reference threshold its default rather than silently inheriting the
    # misleading argument-file value.
    assert "parser.set_defaults(opacity_cull=0.005)" in source
    assert '"opacity_cull": float(opt.opacity_cull)' in source


def test_standard_control_audits_the_full_topology_threshold_contract():
    opt = SimpleNamespace(
        iterations=7000,
        position_lr_init=1.6e-5,
        position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.001,
        rotation_lr=0.001,
        non_position_lr_decay_from=-1,
        non_position_lr_final_mult=1.0,
        lambda_dssim=0.2,
        lambda_dist=0.0,
        lambda_normal=0.05,
        percent_dense=0.01,
        opacity_cull=0.005,
        densify_from_iter=500,
        densify_until_iter=15000,
        densification_interval=100,
        densify_grad_threshold=0.0002,
        opacity_reset_interval=3000,
    )

    audit = _optimization_audit(opt)

    assert audit["topology"] == {
        "percent_dense": 0.01,
        "opacity_cull": 0.005,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densification_interval": 100,
        "densify_grad_threshold": 0.0002,
        "opacity_reset_interval": 3000,
    }
    assert audit["learning_rates"]["position_init"] == 1.6e-5


def test_standard_control_implementation_audit_covers_trainer_and_native_kernel():
    audit = _implementation_audit()

    assert audit["protocol"] == "standard_2dgs_implementation_contract_v1"
    assert "scripts/train_standard_full_2dgs.py" in audit["files"]
    assert "2d-gaussian-splatting/scene/gaussian_model.py" in audit["files"]
    assert any(path.endswith(".so") for path in audit["files"])


def test_causal_reset_continuation_decouples_reset_from_densification_horizon():
    native = [
        iteration
        for iteration in range(1, 7001)
        if opacity_reset_due(
            iteration,
            opacity_reset_interval=3000,
            densify_from_iter=500,
            densify_until_iter=3500,
            white_background=True,
            continue_after_densify=False,
        )
    ]
    held_reset_timeline = [
        iteration
        for iteration in range(1, 7001)
        if opacity_reset_due(
            iteration,
            opacity_reset_interval=3000,
            densify_from_iter=500,
            densify_until_iter=3500,
            white_background=True,
            continue_after_densify=True,
        )
    ]

    # The native semantics confound a short densification horizon with a
    # missing 6k reset. The opt-in causal mode restores exactly that reset.
    assert native == [500, 3000]
    assert held_reset_timeline == [500, 3000, 6000]


def test_standard_control_records_effective_reset_iterations_not_just_flags():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    assert 'input_audit["opacity_reset_iterations"]' in source
    assert "continue_after_densify=bool(args.continue_opacity_resets_after_densify)" in source


def test_standard_control_saves_before_topology_and_skips_unpersisted_terminal_mutation():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    # Clean ULF-Loc/STDLoc main calls scene.save before the densification and
    # opacity-reset block.  At a terminal save the later mutation is not
    # persisted, so the standard runner must also leave its final in-memory
    # state equal to its final checkpoint for fair rendering/evaluation.
    save_index = source.index("            if iteration in saves:")
    terminal_gate_index = source.index("            if iteration < opt.iterations:", save_index)
    densify_index = source.index("                if iteration < opt.densify_until_iter:", terminal_gate_index)
    reset_index = source.index("                if opacity_reset_due(", terminal_gate_index)
    assert save_index < terminal_gate_index < densify_index < reset_index


def test_exact_training_state_is_captured_after_the_optimizer_branch_not_at_ply_save():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    ply_save_index = source.index("            if iteration in saves:")
    optimizer_step_index = source.index("                gaussians.optimizer.step()", ply_save_index)
    state_save_index = source.index("            if iteration in state_checkpoint_iterations:", optimizer_step_index)

    assert ply_save_index < optimizer_step_index < state_save_index
    assert '"final_render": "intentionally_skipped_no_matching_ply_snapshot"' in source


def test_exact_training_state_preserves_the_observable_training_loop_ema():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()

    assert '"training_loop_state": dict(training_loop_state)' in source
    assert 'resume_ema = float(resume_state_payload["training_loop_state"]["ema"])' in source
    assert 'training_loop_state={"ema": float(ema)}' in source
    assert '"state_checkpoint_version": 2' in source


def test_ulfloc_compatible_runtime_flags_are_explicit_and_audited():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()
    assert '"--deterministic-cudnn"' in source
    assert "torch.backends.cudnn.deterministic = True" in source
    assert '"cudnn": {' in source


def test_ulfloc_reference_profile_rejects_an_accidental_black_background():
    source = Path("scripts/train_standard_full_2dgs.py").read_text()

    # The local 2DGS arguments file defaults to black, while the released
    # ULF-Loc/STDLoc Cambridge recipe defaults to white.  A strict reference
    # control must fail closed instead of silently accepting that mismatch.
    assert '"--allow-nonreference-background"' in source
    assert 'args.supervision_profile == "ulfloc_masked"' in source
    assert "not dataset.white_background" in source
    assert '"white_background": bool(dataset.white_background)' in source


def test_fixed_camera_schedule_is_epoch_balanced_and_rng_independent():
    views = [SimpleNamespace(image_name=f"view_{index:02d}") for index in range(5)]

    first, first_audit = _fixed_camera_schedule(views, iterations=13, seed=1729)
    second, second_audit = _fixed_camera_schedule(views, iterations=13, seed=1729)

    assert first == second
    assert first_audit == second_audit
    assert first_audit["protocol"] == "local_seeded_epoch_permutation_v1"
    assert first_audit["full_epochs"] == 2
    assert first_audit["tail_length"] == 3
    assert first_audit["unique_cameras_seen"] == 5
    assert first_audit["per_camera_count_min"] == 2
    assert first_audit["per_camera_count_max"] == 3
    assert sorted(first[:5]) == list(range(5))
    assert sorted(first[5:10]) == list(range(5))


def test_ulfloc_clean_main_schedule_reproduces_scene_shuffle_and_unused_draw():
    views = [SimpleNamespace(image_name=name) for name in ("a", "b", "c")]

    first, audit = _ulfloc_clean_main_camera_schedule(views, iterations=7, seed=0)
    second, repeat_audit = _ulfloc_clean_main_camera_schedule(views, iterations=7, seed=0)

    # This literal sequence is independently derived from Python's
    # ``Random(0).shuffle`` then the released randint-pop loop. It should not
    # collapse to the runner's per-epoch permutation schedule.
    assert first == [2, 1, 0, 2, 1, 0, 1]
    assert first == second
    assert audit == repeat_audit
    assert audit["protocol"] == "clean_ulfloc_main_scene_shuffle_randint_pop_v1"
    assert audit["preloop_unused_camera_draw"]
    assert audit["full_epochs"] == 2
    assert audit["tail_length"] == 1
    assert audit["unique_cameras_seen"] == 3
    assert audit["per_camera_count_min"] == 2
    assert audit["per_camera_count_max"] == 3


def test_staged_name_audit_ignores_external_mask_sidecars(tmp_path):
    (tmp_path / "frame.png").write_bytes(b"rgb")
    (tmp_path / "masks.pkl").write_bytes(b"not an RGB frame")

    assert _staged_names(tmp_path) == {"frame.png"}


def test_loaded_camera_audit_recovers_real_filename_extensions():
    cameras = [SimpleNamespace(image_name="first"), SimpleNamespace(image_name="second")]

    assert _loaded_camera_filenames({"first.jpg", "second.png"}, cameras) == {
        "first.jpg",
        "second.png",
    }


def test_empty_diagnostic_evaluation_still_creates_metrics_directory(tmp_path):
    scene = SimpleNamespace(getTrainCameras=lambda: [])

    result = _evaluate(
        scene=scene,
        gaussians=None,
        pipe=None,
        background=None,
        output_dir=tmp_path / "evaluation",
        target_stems=set(),
        evaluate_all=False,
    )

    assert result == {"evaluated_views": 0, "per_view": {}}
    assert (tmp_path / "evaluation").is_dir()
