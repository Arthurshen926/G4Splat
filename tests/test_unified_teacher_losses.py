from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts.train_unified_outdoor_teacher import (
    SURFACE_OWNERSHIP_REPAIR_TARGET,
    TRAINING_PROFILES,
    _adapt_volume,
    _activation_iteration,
    _accumulate_volume_stats,
    _adaptive_geometry_scale,
    _backward_conditioned_foliage,
    _dynamic_enabled,
    _foliage_enabled,
    _full_epoch_schedule,
    _dynamic_opacity_floor,
    _dynamic_observation_factor,
    _dav2_observation_patch_loss,
    _enforce_retirement_only_fallback_opacity_gradient,
    _evidence_biased_schedule,
    _balanced_instance_topk,
    _boundary_evidence_weight,
    _branch_isolated_rgb_losses,
    _bounded_surface_retirement_fractions,
    _canopy_topology_signal,
    _confirmed_contradiction_fraction,
    _continued_surface_iteration,
    _conditioned_base_gradient_gate,
    _conditioned_photo_losses,
    _evidence_adaptive_role_quotas,
    _geometry_losses,
    _inherit_surface_optimizer_contract,
    _load_dav2_observation_patches,
    _masked_clamp_max_,
    _masked_ssim_loss,
    _apply_mature_surface_gradient_policy,
    _phase,
    _resolved_phase_schedule,
    _retain_checkpoint_snapshot,
    _rigid_completion_seed_indices,
    _resume_training_contract_differences,
    _soft_surface_canopy_conflict,
    _surface_capture_to_device,
    _surface_topology_active,
    _volume_topology_ramp_scale,
    _validate_fixed_cameras,
    _validate_initialization_rgb_source,
    _validate_initialization_protocol,
    _validate_surface_warmstart,
    _volume_stats,
    _write_rigid_stage_surface_handoff,
)
from outdoor.hybrid_gaussian_renderer import (
    LAYER_DYNAMIC_LEAF,
    VolumetricFoliageModel,
)
from outdoor.training_evidence import (
    FoliageRayEvidence,
    OutdoorGeometryEvidence,
)

RIGID_SURFACE_OPTIMIZER = {
    "position_lr_init": 1.6e-5,
    "position_lr_final": 1.6e-6,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 20_000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
}


def test_dav2_observation_patch_contract_is_topology_independent(tmp_path):
    seed = tmp_path / "surface_seed.npz"
    np.savez_compressed(
        seed,
        source_type=np.asarray([4, 4, 1], dtype=np.int8),
        dav2_observation_view_name=np.asarray(
            ["view_a", "view_a", ""]
        ),
        dav2_observation_uv=np.asarray(
            [[0.25, 0.25], [0.75, 0.75], [np.nan, np.nan]],
            dtype=np.float32,
        ),
    )
    lookup = _load_dav2_observation_patches(
        seed, device=torch.device("cpu")
    )
    assert set(lookup) == {"view_a"}
    assert lookup["view_a"].shape == (2, 2)

    target = torch.zeros(3, 8, 8)
    prediction = torch.ones(3, 8, 8, requires_grad=True)
    loss, sampled = _dav2_observation_patch_loss(
        prediction,
        target,
        lookup["view_a"],
        radius=1,
    )
    assert sampled == 18
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0


def test_mature_handoff_appearance_policy_preserves_geometry_gradients():
    surface = SimpleNamespace(
        _xyz=torch.nn.Parameter(torch.ones(2, 3)),
        _scaling=torch.nn.Parameter(torch.ones(2, 2)),
        _rotation=torch.nn.Parameter(torch.ones(2, 4)),
        _opacity=torch.nn.Parameter(torch.ones(2, 1)),
        _features_dc=torch.nn.Parameter(torch.ones(2, 1, 3)),
        _features_rest=torch.nn.Parameter(torch.ones(2, 3, 3)),
    )
    for parameter in vars(surface).values():
        parameter.grad = torch.ones_like(parameter)

    audit = _apply_mature_surface_gradient_policy(
        surface, "appearance_only"
    )

    for parameter in (
        surface._xyz,
        surface._scaling,
        surface._rotation,
        surface._opacity,
    ):
        assert not bool(parameter.grad.any())
    assert bool(surface._features_dc.grad.any())
    assert bool(surface._features_rest.grad.any())
    assert audit["geometry_trainable"] is False
    assert audit["appearance_trainable"] is True


def test_surface_retirement_is_continuous_and_optical_mass_bounded():
    previous = torch.zeros(3)
    desired = torch.tensor([0.8, 0.4, 0.0])
    optical_depth = torch.ones(3)
    updated, audit = _bounded_surface_retirement_fractions(
        previous,
        desired,
        optical_depth,
        maximum_mass_fraction=0.05,
    )
    torch.testing.assert_close(
        updated, torch.tensor([0.1, 0.05, 0.0])
    )
    assert np.isclose(audit["requested_optical_mass_fraction"], 0.4)
    assert np.isclose(audit["realized_optical_mass_fraction"], 0.05)
    assert updated[0] > updated[1] > updated[2]


def test_volume_topology_ramp_is_continuous_and_horizon_bound():
    args = SimpleNamespace(
        volume_topology_ramp_iterations=100,
        training_profile="hybrid_handoff_quality",
        phase_schedule_horizon=1000,
    )
    activation = _activation_iteration(
        TRAINING_PROFILES["hybrid_handoff_quality"]["dynamic_start"],
        1000,
    )
    before = _volume_topology_ramp_scale(activation - 2, args)
    middle = _volume_topology_ramp_scale(activation + 48, args)
    after = _volume_topology_ramp_scale(activation + 198, args)

    assert before == 0.0
    assert 0.45 < middle < 0.55
    assert after == 1.0


def test_masked_clamp_updates_original_parameter_rows():
    values = torch.nn.Parameter(
        torch.tensor([[2.0], [-1.0], [3.0], [0.0]])
    )
    mask = torch.tensor([True, False, True, False])
    _masked_clamp_max_(values, mask, 0.5)
    torch.testing.assert_close(
        values.detach(),
        torch.tensor([[0.5], [-1.0], [0.5], [0.0]]),
    )


def test_retained_checkpoint_snapshot_survives_rolling_replace(tmp_path):
    rolling = tmp_path / "hybrid_teacher_checkpoint.pth"
    rolling.write_bytes(b"iteration-12000")
    snapshot = _retain_checkpoint_snapshot(rolling, 12_000)
    assert snapshot.read_bytes() == b"iteration-12000"
    assert snapshot.stat().st_ino == rolling.stat().st_ino
    replacement = tmp_path / "replacement.pth"
    replacement.write_bytes(b"iteration-13000")
    replacement.replace(rolling)
    assert rolling.read_bytes() == b"iteration-13000"
    assert snapshot.read_bytes() == b"iteration-12000"


def test_adaptive_geometry_scale_targets_bounded_gradient_ratio():
    assert _adaptive_geometry_scale(
        2.0, 4.0, target_ratio=0.25
    ) == 0.125
    assert _adaptive_geometry_scale(
        100.0, 0.01, target_ratio=0.25
    ) == 10.0
    assert _adaptive_geometry_scale(
        0.01, 100.0, target_ratio=0.25
    ) == 0.001
    assert _adaptive_geometry_scale(
        2.0, 4.0, target_ratio=0.0
    ) == 1.0


def test_mature_surface_handoff_continues_absolute_lr_iteration():
    assert _continued_surface_iteration(1, 24_000) == 24_001
    assert _continued_surface_iteration(1_000, 24_000) == 25_000
    with np.testing.assert_raises_regex(ValueError, "Invalid"):
        _continued_surface_iteration(0, 24_000)


def test_branch_isolated_rgb_never_routes_rigid_loss_through_mixed_image():
    structural = torch.full(
        (3, 16, 16), 0.4, requires_grad=True
    )
    mixed = torch.full((3, 16, 16), 0.2, requires_grad=True)
    target = torch.full((3, 16, 16), 0.7)
    rigid = torch.ones(16, 16)
    canopy = torch.zeros(16, 16)
    losses = _branch_isolated_rgb_losses(
        structural,
        mixed,
        target,
        background_weight=rigid,
        canopy_weight=canopy,
        rigid_weight=rigid,
        rigid_high_frequency_weight=rigid,
        lambda_dssim=0.2,
    )
    (
        losses["background"]
        + losses["rigid"]
        + losses["rigid_high_frequency"]
    ).backward()
    assert structural.grad is not None
    assert float(structural.grad.abs().sum()) > 0
    assert mixed.grad is None


def test_conditioned_context_photo_is_excluded_from_foliage_objective():
    prediction = torch.full(
        (3, 16, 16), 0.2, requires_grad=True
    )
    target = torch.full((3, 16, 16), 0.8)
    task = {
        "p_canopy": torch.zeros(16, 16),
        "p_sky": torch.ones(16, 16),
        "p_rigid": torch.zeros(16, 16),
    }
    losses = _conditioned_photo_losses(prediction, target, task)
    losses["canopy"].backward(retain_graph=True)
    assert prediction.grad is not None
    torch.testing.assert_close(
        prediction.grad, torch.zeros_like(prediction.grad)
    )
    prediction.grad.zero_()
    losses["combined"].backward()
    assert float(prediction.grad.abs().sum()) > 0
    assert float(losses["context"]) > 0


def test_boundary_evidence_is_continuous_and_never_hard_zero():
    boundary = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    weight = _boundary_evidence_weight(boundary)

    torch.testing.assert_close(
        weight,
        torch.tensor([1.0, 0.8125, 0.625, 0.4375, 0.25]),
    )
    assert bool((weight > 0).all())


def test_capacity_repair_resume_allows_only_exact_audited_contract():
    saved = {
        "maximum_volume_gaussians": 1_500_000,
        "maximum_volume_splits_per_event": 12_000,
        "geometry_weight": 0.12,
    }
    current = {
        **saved,
        "maximum_volume_gaussians": 2_000_000,
        "maximum_volume_splits_per_event": 20_000,
        "volume_densify_until_iteration": 10_000,
        "boundary_supervision_contract": (
            "continuous_semantic_boundary_evidence_floor025_no_binary_gate"
        ),
        "volume_reallocation_contract": (
            "role_matched_lineage_safe_retire_then_evidence_adaptive_split"
        ),
    }

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_volume_capacity_repair_migration=True,
    )
    bad = {**current, "maximum_volume_gaussians": 2_100_000}
    with np.testing.assert_raises_regex(
        RuntimeError, "target contract mismatch"
    ):
        _resume_training_contract_differences(
            saved,
            bad,
            allow_volume_capacity_repair_migration=True,
        )


def test_surface_ownership_resume_changes_only_the_audited_contract():
    saved = {
        "mature_handoff_surface_policy": "appearance_only",
        "surface_ownership_contract": (
            "surface_opacity_follows_mature_handoff_policy"
        ),
        "geometry_weight": 0.12,
    }
    current = {
        **saved,
        **SURFACE_OWNERSHIP_REPAIR_TARGET,
    }

    assert not _resume_training_contract_differences(
        saved,
        current,
        allow_surface_ownership_repair_migration=True,
    )
    bad = {**current, "geometry_weight": 0.13}
    assert _resume_training_contract_differences(
        saved,
        bad,
        allow_surface_ownership_repair_migration=True,
    ) == {"geometry_weight"}


def test_rigid_completion_births_only_independently_supported_deficits():
    seed = {
        "xyz": np.asarray(
            [[0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "scales": np.full((3, 2), 0.03, dtype=np.float32),
        "geometry_confidence": np.asarray(
            [1.0, 0.8, 1.0], dtype=np.float32
        ),
        "position_sigma": np.full(3, 0.04, dtype=np.float32),
        "pointmap_cross_sequence_supported": np.asarray(
            [True, True, False]
        ),
        "persistent_geometry_evidence": np.asarray(
            [False, False, False]
        ),
    }
    selected, audit = _rigid_completion_seed_indices(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        seed,
        maximum_seeds=4,
    )
    assert selected.tolist() == [1]
    assert audit["candidate_count"] == 1
    assert audit["cross_sequence_pointmap_selected"] == 1
    disabled, disabled_audit = _rigid_completion_seed_indices(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        seed,
        maximum_seeds=0,
    )
    assert disabled.tolist() == []
    assert not disabled_audit["enabled"]


def test_conditioned_base_gradient_gate_excludes_canonical_forward_owners():
    foliage = SimpleNamespace(
        dynamic_leaf_mask=torch.tensor(
            [False, False, True, True, True]
        ),
        xyz=torch.zeros(5, 3),
    )
    gate = _conditioned_base_gradient_gate(
        foliage,
        torch.tensor([1.0, 1.0, 1.0, 0.35, 0.0]),
    )
    torch.testing.assert_close(
        gate,
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_conditioned_base_gradient_gate_rejects_misaligned_topology():
    foliage = SimpleNamespace(
        dynamic_leaf_mask=torch.tensor([False, True]),
        xyz=torch.zeros(2, 3),
    )
    with np.testing.assert_raises_regex(ValueError, "one value"):
        _conditioned_base_gradient_gate(
            foliage, torch.ones(3)
        )


def test_evidence_adaptive_role_quotas_normalize_producer_density():
    quotas = _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 59,
            "canonical_crown": 32_781,
            "dynamic_leaf": 136_302,
        },
        {
            "static_skeleton": 403,
            "canonical_crown": 122_391,
            "dynamic_leaf": 938_283,
        },
        12_000,
    )
    assert quotas["static_skeleton"] == 59
    assert quotas["canonical_crown"] > quotas["dynamic_leaf"]
    assert sum(quotas.values()) == 12_000


def test_evidence_adaptive_role_quotas_ignore_duplicate_row_multiplier():
    eligible = {
        "static_skeleton": 100,
        "canonical_crown": 4_000,
        "dynamic_leaf": 8_000,
    }
    population = {
        "static_skeleton": 1_000,
        "canonical_crown": 40_000,
        "dynamic_leaf": 80_000,
    }
    baseline = _evidence_adaptive_role_quotas(
        eligible, population, 6_000
    )
    duplicated = _evidence_adaptive_role_quotas(
        {**eligible, "dynamic_leaf": 80_000},
        {**population, "dynamic_leaf": 800_000},
        6_000,
    )
    assert baseline == duplicated


def test_evidence_adaptive_role_quotas_use_observed_dynamic_population():
    quotas = _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 0,
            "canonical_crown": 4_000,
            "dynamic_leaf": 8_000,
        },
        {
            "static_skeleton": 1,
            "canonical_crown": 40_000,
            "dynamic_leaf": 800_000,
        },
        6_000,
        observable_counts={
            "static_skeleton": 0,
            "canonical_crown": 20_000,
            "dynamic_leaf": 10_000,
        },
    )
    # Unvisited owner-camera rows must not dilute the deficit measured on
    # exact dynamic contexts in this topology epoch.
    assert quotas["dynamic_leaf"] > quotas["canonical_crown"]
    assert sum(quotas.values()) == 6_000


def test_evidence_adaptive_role_quotas_never_invent_or_drop_capacity():
    assert _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 3,
            "canonical_crown": 0,
            "dynamic_leaf": 9,
        },
        {
            "static_skeleton": 30,
            "canonical_crown": 1,
            "dynamic_leaf": 90,
        },
        6,
    ) == {
        "static_skeleton": 3,
        "canonical_crown": 0,
        "dynamic_leaf": 3,
    }
    assert _evidence_adaptive_role_quotas(
        {
            "static_skeleton": 3,
            "canonical_crown": 0,
            "dynamic_leaf": 9,
        },
        {
            "static_skeleton": 30,
            "canonical_crown": 1,
            "dynamic_leaf": 90,
        },
        20,
    ) == {
        "static_skeleton": 3,
        "canonical_crown": 0,
        "dynamic_leaf": 9,
    }
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        _evidence_adaptive_role_quotas(
            {
                "static_skeleton": -1,
                "canonical_crown": 0,
                "dynamic_leaf": 1,
            },
            {
                "static_skeleton": 1,
                "canonical_crown": 1,
                "dynamic_leaf": 1,
            },
            1,
        )
    with np.testing.assert_raises_regex(ValueError, "exceeds"):
        _evidence_adaptive_role_quotas(
            {
                "static_skeleton": 2,
                "canonical_crown": 0,
                "dynamic_leaf": 0,
            },
            {
                "static_skeleton": 1,
                "canonical_crown": 1,
                "dynamic_leaf": 1,
            },
            1,
        )


def test_resume_contract_allows_only_camera_container_cache_digest_change():
    saved = {
        "camera_geometry_sha256": "same-geometry",
        "fixed_camera_validation": {"exact_image_identity": True},
        "camera_intrinsics_contract_sha256": "cache-four",
        "geometry_weight": 0.12,
    }
    current = {
        **saved,
        "camera_intrinsics_contract_sha256": "cache-all",
    }
    assert not _resume_training_contract_differences(saved, current)
    changed_geometry = {
        **current,
        "camera_geometry_sha256": "different-geometry",
    }
    assert _resume_training_contract_differences(
        saved, changed_geometry
    ) == {
        "camera_geometry_sha256",
        "camera_intrinsics_contract_sha256",
    }
    changed_loss = {**current, "geometry_weight": 0.20}
    assert _resume_training_contract_differences(
        saved, changed_loss
    ) == {
        "camera_intrinsics_contract_sha256",
        "geometry_weight",
    }


def test_fast_profile_overlaps_dynamic_foliage_with_topology():
    assert TRAINING_PROFILES["fast"]["iterations"] == 30_000
    assert _phase(2_399, 30_000, "fast") == "canonical_bootstrap"
    assert _phase(2_400, 30_000, "fast") == "topology"
    assert not _dynamic_enabled(2_399, 30_000, "fast")
    assert _dynamic_enabled(2_400, 30_000, "fast")
    assert _phase(11_999, 30_000, "fast") == "topology"
    assert _phase(12_000, 30_000, "fast") == "dynamic_appearance"
    assert _phase(21_899, 30_000, "fast") == "dynamic_appearance"
    assert _phase(21_900, 30_000, "fast") == "ownership_cleanup"
    assert _phase(26_999, 30_000, "fast") == "ownership_cleanup"
    assert _phase(27_000, 30_000, "fast") == "canonical_polish"


def test_quality_profile_keeps_final_benchmark_schedule():
    assert TRAINING_PROFILES["quality"]["iterations"] == 80_000
    assert _phase(4_799, 80_000) == "canonical_bootstrap"
    assert _phase(4_800, 80_000) == "topology"
    assert _dynamic_enabled(4_800, 80_000, "quality")
    assert _phase(46_399, 80_000) == "topology"
    assert _phase(46_400, 80_000) == "dynamic_appearance"


def test_handoff_profile_starts_foliage_without_rebuilding_rigid_surface():
    profile = TRAINING_PROFILES["hybrid_handoff_quality"]

    assert profile["iterations"] == 30_000
    assert profile["foliage_start"] == 0.0
    assert _phase(599, 30_000, "hybrid_handoff_quality") == (
        "canonical_bootstrap"
    )
    assert _phase(600, 30_000, "hybrid_handoff_quality") == "topology"
    assert _phase(7_499, 30_000, "hybrid_handoff_quality") == "topology"
    assert _phase(7_500, 30_000, "hybrid_handoff_quality") == (
        "static_foliage"
    )
    assert not _dynamic_enabled(
        599, 30_000, "hybrid_handoff_quality"
    )
    assert _dynamic_enabled(
        600, 30_000, "hybrid_handoff_quality"
    )


def test_absolute_schedule_makes_short_run_an_exact_method_prefix():
    profile = "hybrid_handoff_quality"
    horizon = 12_000
    assert _resolved_phase_schedule(profile, horizon) == (
        ("canonical_bootstrap", 240),
        ("topology", 3_000),
        ("static_foliage", 4_200),
        ("dynamic_appearance", 8_640),
        ("ownership_cleanup", 11_040),
        ("canonical_polish", 12_000),
    )
    assert _activation_iteration(0.02, horizon) == 241
    short = _full_epoch_schedule(1487, 4_000, 1702)
    complete = _full_epoch_schedule(1487, horizon, 1702)
    np.testing.assert_array_equal(short, complete[:4_000])
    for step in (0, 239, 240, 2_999, 3_000, 3_999):
        # The requested stop iteration is deliberately absent: phase and
        # branch identity depend only on the persisted method horizon.
        assert _phase(step, horizon, profile) == _phase(
            step, horizon, profile
        )
        assert _dynamic_enabled(step, horizon, profile) == (
            step >= 240
        )
        assert _foliage_enabled(step, horizon, profile)


def test_surface_topology_schedule_can_extend_past_named_topology_phase():
    args = SimpleNamespace(
        densify_from_iter=500,
        densify_until_iter=10_500,
    )
    assert _phase(8_999, 30_000, "hybrid_handoff_quality") == (
        "static_foliage"
    )
    assert _surface_topology_active(8_999, args)
    assert _surface_topology_active(10_499, args)
    assert not _surface_topology_active(10_500, args)


def test_known_invalid_free_space_initialization_cannot_train():
    with np.testing.assert_raises_regex(
        RuntimeError, "invalid free-space contradiction metadata"
    ):
        _validate_initialization_protocol(
            {
                "version": (
                    "outdoor-role-aware-initialization-v14-priority-"
                    "dav2-hole-fill"
                )
            }
        )
    _validate_initialization_protocol(
        {
            "version": (
                "outdoor-role-aware-initialization-v15-"
                "confirmed-free-space"
            )
        }
    )


def test_exact_dynamic_rgb_initialization_requires_matching_training_rgb():
    contract = {
        "image_count": 3,
        "name_set_sha256": "names",
        "content_mapping_sha256": "content",
        "producer_manifest_sha256": "producer",
        "target_storage": "uint8",
        "canonical_image_size_wh": [640, 360],
    }
    initialization = {
        "rgb_source": dict(contract),
        "foliage": {
            "dense_dynamic_rgb_source": "exact_training_target_raster"
        },
    }
    _validate_initialization_rgb_source(initialization, dict(contract))
    mismatched = dict(contract)
    mismatched["content_mapping_sha256"] = "different"
    with np.testing.assert_raises_regex(
        RuntimeError, "RGB source contract mismatch"
    ):
        _validate_initialization_rgb_source(initialization, mismatched)
    with np.testing.assert_raises_regex(
        RuntimeError, "no persisted RGB source contract"
    ):
        _validate_initialization_rgb_source(
            {"foliage": initialization["foliage"]}, contract
        )


def test_native_rigid_surface_handoff_requires_exact_digest_and_no_colmap(
    tmp_path,
):
    surface = tmp_path / "surface.ply"
    surface.write_bytes(b"rigid surface")
    digest = hashlib.sha256(surface.read_bytes()).hexdigest()
    manifest = tmp_path / "handoff.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "native-rigid-surface-handoff-v1",
                "eligible_for_hybrid_surface_handoff": True,
                "rejection_reasons": [],
                "surface_ply": str(surface.resolve()),
                "surface_ply_sha256": digest,
                "camera_geometry_sha256": "camera",
                "historical_gaussian_input_used": False,
                "colmap_points_or_tracks_used": False,
                "surface_ownership_contract": (
                    "rigid_pixels_tree_sky_transient_excluded"
                ),
                "surface_optimizer": RIGID_SURFACE_OPTIMIZER,
            }
        ),
        encoding="utf-8",
    )

    accepted = _validate_surface_warmstart(surface, manifest)

    assert accepted["surface_ply_sha256"] == digest
    assert accepted["handoff_manifest_sha256"]
    assert accepted["surface_optimizer"] == RIGID_SURFACE_OPTIMIZER
    opt = SimpleNamespace(position_lr_init=1.6e-4)
    inherited = _inherit_surface_optimizer_contract(opt, accepted)
    assert inherited == RIGID_SURFACE_OPTIMIZER
    assert opt.position_lr_init == 1.6e-5
    assert opt.position_lr_max_steps == 20_000
    surface.write_bytes(b"mutated")
    with np.testing.assert_raises_regex(RuntimeError, "digest mismatch"):
        _validate_surface_warmstart(surface, manifest)


def test_legacy_rigid_handoff_recovers_validated_producer_schedule(tmp_path):
    surface = tmp_path / "surface.ply"
    surface.write_bytes(b"legacy rigid surface")
    digest = hashlib.sha256(surface.read_bytes()).hexdigest()
    manifest = tmp_path / "rigid_surface_handoff.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "native-rigid-surface-handoff-v1",
                "eligible_for_hybrid_surface_handoff": True,
                "surface_ply": str(surface.resolve()),
                "surface_ply_sha256": digest,
                "surface_iteration": 24_000,
                "historical_gaussian_input_used": False,
                "colmap_points_or_tracks_used": False,
                "surface_ownership_contract": (
                    "rigid_pixels_tree_sky_transient_excluded"
                ),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "iterations": 24_000,
                "surface_ply": str(surface.resolve()),
                "training_contract": {
                    "surface_optimizer": RIGID_SURFACE_OPTIMIZER
                },
            }
        ),
        encoding="utf-8",
    )

    accepted = _validate_surface_warmstart(surface, manifest)

    assert accepted["surface_optimizer"] == RIGID_SURFACE_OPTIMIZER
    assert accepted["surface_optimizer_source"] == (
        "validated_legacy_producer_result"
    )


def test_geometry_trained_rigid_stage_writes_eligible_handoff(tmp_path):
    output = tmp_path / "rigid"
    point_cloud = output / "point_cloud" / "iteration_24000"
    point_cloud.mkdir(parents=True)
    surface = point_cloud / "point_cloud.ply"
    surface.write_bytes(b"geometry-trained rigid surface")
    initialization_directory = tmp_path / "initialization"
    initialization_directory.mkdir()
    initialization_manifest = (
        initialization_directory / "initialization_manifest.json"
    )
    initialization = {
        "historical_model_initialization": False,
        "surface": {
            "historical_trained_ply_used": False,
            "colmap_points_or_tracks_used": False,
            "canopy_surface_seed_count": 0,
        },
    }
    initialization_manifest.write_text(
        json.dumps(initialization), encoding="utf-8"
    )
    camera_contract = {
        "camera_geometry_sha256": "exact-camera-geometry"
    }
    (output / "camera_intrinsics_contract.json").write_text(
        json.dumps(camera_contract), encoding="utf-8"
    )

    destination = _write_rigid_stage_surface_handoff(
        output=output,
        iteration=24_000,
        surface_iteration=48_000,
        surface_point_count=321,
        initialization=initialization,
        initialization_directory=initialization_directory,
        evidence_store={
            "geometry_source": "mast3r_only",
            "colmap_points_or_tracks_used": False,
        },
        camera_runtime_contract=camera_contract,
        fixed_camera_validation={"camera_count": 1487},
        geometry_audit={
            "source_consumption_count": {
                "chart_native_factor": 12_000,
                "track_observation_factor": 12_000,
            }
        },
        surface_optimizer=RIGID_SURFACE_OPTIMIZER,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["eligible_for_hybrid_surface_handoff"]
    assert payload["producer"] == "hybrid_rigid_stage1"
    assert payload["surface_iteration"] == 48_000
    assert payload["surface_ownership_contract"] == (
        "rigid_pixels_tree_sky_transient_excluded"
    )
    accepted = _validate_surface_warmstart(surface, destination)
    assert accepted["camera_geometry_sha256"] == "exact-camera-geometry"
    assert accepted["surface_optimizer_source"] == "handoff_manifest"


def test_legacy_mast3r_only_store_without_colmap_boolean_is_eligible(
    tmp_path,
):
    """Legacy stores expressed the same guarantee without a boolean field."""
    output = tmp_path / "rigid"
    point_cloud = output / "point_cloud" / "iteration_24000"
    point_cloud.mkdir(parents=True)
    surface = point_cloud / "point_cloud.ply"
    surface.write_bytes(b"geometry-trained rigid surface")
    initialization_directory = tmp_path / "initialization"
    initialization_directory.mkdir()
    initialization = {
        "historical_model_initialization": False,
        "surface": {
            "historical_trained_ply_used": False,
            "colmap_points_or_tracks_used": False,
            "canopy_surface_seed_count": 0,
        },
    }
    (initialization_directory / "initialization_manifest.json").write_text(
        json.dumps(initialization), encoding="utf-8"
    )
    camera_contract = {
        "camera_geometry_sha256": "exact-camera-geometry"
    }
    (output / "camera_intrinsics_contract.json").write_text(
        json.dumps(camera_contract), encoding="utf-8"
    )

    destination = _write_rigid_stage_surface_handoff(
        output=output,
        iteration=24_000,
        surface_point_count=321,
        initialization=initialization,
        initialization_directory=initialization_directory,
        evidence_store={
            "geometry_source": "mast3r_only",
            "artifacts": [
                {
                    "name": "mast3r_track_graph",
                    "source_type": "mast3r_fixed_camera_track_graph",
                    "coordinate_frame": "COLMAP_world",
                }
            ],
        },
        camera_runtime_contract=camera_contract,
        fixed_camera_validation={"camera_count": 1487},
        geometry_audit={
            "source_consumption_count": {
                "chart_native_factor": 12_000,
                "track_observation_factor": 12_000,
            }
        },
        surface_optimizer=RIGID_SURFACE_OPTIMIZER,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["eligible_for_hybrid_surface_handoff"]
    assert not payload["colmap_points_or_tracks_used"]


def test_legacy_store_with_colmap_track_artifact_fails_closed(tmp_path):
    output = tmp_path / "rigid"
    point_cloud = output / "point_cloud" / "iteration_24000"
    point_cloud.mkdir(parents=True)
    (point_cloud / "point_cloud.ply").write_bytes(b"surface")
    initialization_directory = tmp_path / "initialization"
    initialization_directory.mkdir()
    initialization = {
        "historical_model_initialization": False,
        "surface": {
            "historical_trained_ply_used": False,
            "colmap_points_or_tracks_used": False,
            "canopy_surface_seed_count": 0,
        },
    }
    (initialization_directory / "initialization_manifest.json").write_text(
        json.dumps(initialization), encoding="utf-8"
    )
    camera_contract = {
        "camera_geometry_sha256": "exact-camera-geometry"
    }
    (output / "camera_intrinsics_contract.json").write_text(
        json.dumps(camera_contract), encoding="utf-8"
    )

    destination = _write_rigid_stage_surface_handoff(
        output=output,
        iteration=24_000,
        surface_point_count=1,
        initialization=initialization,
        initialization_directory=initialization_directory,
        evidence_store={
            "geometry_source": "mast3r_only",
            "artifacts": [
                {
                    "name": "colmap_tracks",
                    "source_type": "colmap_tracks",
                }
            ],
        },
        camera_runtime_contract=camera_contract,
        fixed_camera_validation={"camera_count": 1487},
        geometry_audit={
            "source_consumption_count": {
                "chart_native_factor": 1,
                "track_observation_factor": 1,
            }
        },
        surface_optimizer=RIGID_SURFACE_OPTIMIZER,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert not payload["eligible_for_hybrid_surface_handoff"]
    assert payload["colmap_points_or_tracks_used"]


def test_fixed_camera_contract_checks_exact_scaled_k_and_pose():
    view = SimpleNamespace(
        image_name="frame.png",
        image_width=640,
        image_height=360,
        focal_x=400.0,
        focal_y=420.0,
        cx=321.0,
        cy=179.0,
        R=np.eye(3),
        T=np.asarray([1.0, 2.0, 3.0]),
    )
    contract = {
        "records": [
            {
                "image_name": "frame.png",
                "camera": {
                    "width": 1280,
                    "height": 720,
                    "fx": 800.0,
                    "fy": 840.0,
                    "cx": 642.0,
                    "cy": 358.0,
                },
                "T_world_to_camera": [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ]
    }
    audit = _validate_fixed_cameras([view], contract)
    assert audit["camera_count"] == 1
    assert max(audit["maximum_absolute_error"].values()) == 0.0

    bad_view = SimpleNamespace(**vars(view))
    bad_view.focal_x += 0.1
    with np.testing.assert_raises_regex(
        RuntimeError, "camera calibration differs"
    ):
        _validate_fixed_cameras([bad_view], contract)


def test_contradiction_fraction_uses_all_confirmed_binary_evidence():
    positive = torch.tensor([4, 1, 0], dtype=torch.int16)
    confirmed_free = torch.tensor([3, 1, 2], dtype=torch.int16)

    ratio = _confirmed_contradiction_fraction(
        positive, confirmed_free
    )

    assert torch.allclose(
        ratio, torch.tensor([3.0 / 7.0, 0.5, 1.0])
    )
    assert ratio[0] < 0.55


def test_exact_dynamic_observation_survives_unsampled_topology_window():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.full((1, 3), 0.05),
        "colors": torch.full((1, 3), 0.4),
        "opacities": torch.full((1, 1), 0.001),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor(
            [LAYER_DYNAMIC_LEAF], dtype=torch.int8
        ),
        "support_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_camera_ids": torch.tensor(
            [[7]], dtype=torch.int32
        ),
        "observation_uv": torch.tensor([[[0.5, 0.5]]]),
        "observation_depth": torch.tensor([[2.0]]),
        "support_view_count": torch.tensor([1], dtype=torch.int16),
        "free_space_violation_count": torch.tensor(
            [10], dtype=torch.int16
        ),
        "occupancy_probability": torch.tensor([0.01]),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    stats = _volume_stats(foliage)

    event = _adapt_volume(
        SimpleNamespace(
            volume_split_radius=3.0,
            maximum_volume_gaussians=1,
            maximum_volume_splits=0,
        ),
        foliage,
        stats,
        volume_budget=1,
    )

    assert len(foliage) == 1
    assert event["pruned"] == 0


def test_read_only_temporal_fallback_cannot_drive_volume_topology():
    class FoliageStub:
        xyz = torch.zeros(2, 3)

        def __len__(self):
            return 2

    stats = _volume_stats(FoliageStub())
    means2d = torch.zeros(2, 3, requires_grad=True)
    means2d.grad = torch.tensor(
        [[2.0, 0.0, 0.0], [9.0, 0.0, 0.0]]
    )
    package = SimpleNamespace(
        structural_count=0,
        radii=torch.tensor([5.0, 7.0]),
        volume_means2d=means2d,
        responsibility=torch.tensor(
            [[3.0, 0.0, 2.0, 4.0], [8.0, 0.0, 6.0, 7.0]]
        ),
    )

    _accumulate_volume_stats(
        stats,
        package,
        ownership_gate=torch.tensor([1.0, 0.0]),
        exact_conditioned_dynamic=True,
    )

    assert stats["radius"].tolist() == [5.0, 0.0]
    assert stats["gradient"].tolist() == [2.0, 0.0]
    assert stats["gradient_count"].tolist() == [1.0, 0.0]
    assert stats["conditioned_radius"].tolist() == [5.0, 0.0]
    assert stats["conditioned_gradient"].tolist() == [2.0, 0.0]
    assert stats["conditioned_gradient_count"].tolist() == [1.0, 0.0]
    assert stats["contribution"].tolist() == [3.0, 0.0]
    assert stats["rigid"].tolist() == [2.0, 0.0]
    assert stats["residual"].tolist() == [4.0, 0.0]


def test_conditioned_backward_preserves_independent_screen_gradient_leaf():
    foliage = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        foliage.weight.fill_(2.0)
    volume_means2d = torch.zeros(1, 3, requires_grad=True)
    loss = (
        foliage(torch.ones(1, 1)).sum()
        + 3.0 * volume_means2d[:, :2].sum()
    )

    _backward_conditioned_foliage(loss, foliage, volume_means2d)

    assert foliage.weight.grad is not None
    assert volume_means2d.grad is not None
    assert volume_means2d.grad[:, :2].tolist() == [[3.0, 3.0]]


def test_fallback_opacity_can_retire_but_cannot_establish_occupancy():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [0.1, 0.0, 2.0],
                [0.2, 0.0, 2.0],
                [0.3, 0.0, 2.0],
            ]
        ),
        "scales": torch.full((4, 3), 0.05),
        "colors": torch.full((4, 3), 0.4),
        "opacities": torch.full((4, 1), 0.015),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 4
        ),
        "layer_role": torch.tensor([0, 2, 2, 2], dtype=torch.int8),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)
    foliage.opacity_logits.grad = torch.tensor(
        [[-1.0], [-2.0], [3.0], [-4.0]]
    )
    foliage.dynamic_opacity_basis.grad = torch.ones_like(
        foliage.dynamic_opacity_basis
    )

    audit = _enforce_retirement_only_fallback_opacity_gradient(
        foliage,
        visibility_gate=torch.tensor([1.0, 0.5, 0.4, 1.0]),
        exact_gradient_gate=torch.tensor([1.0, 0.0, 0.0, 1.0]),
    )

    # Canonical and exact-owner rows remain unconstrained. The negative
    # fallback gradient that would grow opacity is removed, while the
    # positive retirement gradient remains.
    assert foliage.opacity_logits.grad[:, 0].tolist() == [
        -1.0,
        0.0,
        3.0,
        -4.0,
    ]
    assert not bool(
        foliage.dynamic_opacity_basis.grad[1:3].any()
    )
    assert bool(foliage.dynamic_opacity_basis.grad[[0, 3]].all())
    assert audit["fallback_opacity_rows"] == 2
    assert audit["fallback_opacity_increase_rows_suppressed"] == 1
    assert (
        audit["fallback_opacity_increase_gradient_suppressed"] == 2.0
    )
    assert audit["fallback_temporal_opacity_rows_suppressed"] == 2


def test_dynamic_opacity_floor_uses_local_dense_ray_evidence():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]
        ),
        "scales": torch.full((2, 3), 0.05),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.015),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "layer_role": torch.full(
            (2,), LAYER_DYNAMIC_LEAF, dtype=torch.int8
        ),
        "initialization_source": torch.tensor(
            [4, 1], dtype=torch.int8
        ),
        "occupancy_probability": torch.tensor([0.9, 0.9]),
        "support_view_count": torch.tensor(
            [2, 2], dtype=torch.int16
        ),
        "unknown_view_count": torch.zeros(2, dtype=torch.int16),
        "ray_depth_nll": torch.zeros(2),
        "free_space_violation_count": torch.zeros(
            2, dtype=torch.int16
        ),
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    floor = _dynamic_opacity_floor(foliage)

    assert 0.04 < floor[0] <= 0.06
    assert floor[1] == 0.015


def test_masked_ssim_does_not_create_zero_boundary_error():
    image = torch.rand(3, 32, 32)
    weight = torch.zeros(32, 32)
    weight[8:24, 8:24] = 1
    assert _masked_ssim_loss(image, image, weight).item() < 1e-6


def test_dynamic_observation_factor_has_geometry_gradient_at_tiny_opacity():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.2, 0.0, 2.0]]),
        "scales": torch.tensor([[0.1, 0.1, 0.1]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[1e-6]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor([LAYER_DYNAMIC_LEAF], dtype=torch.int8),
        "support_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_uv": torch.tensor([[[0.5, 0.5]]]),
        "observation_depth": torch.tensor([[2.0]]),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    camera = SimpleNamespace(
        colmap_id=7,
        world_view_transform=torch.eye(4),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )
    loss, observed, audit = _dynamic_observation_factor(
        model, camera, torch.zeros(model.dynamic_rank)
    )
    assert audit["matched"] == 1
    assert observed.tolist() == [True]
    assert loss > 0
    loss.backward()
    assert model.xyz.grad is not None
    assert model.xyz.grad.norm() > 0
    assert audit["ray_hit"] > 0
    assert audit["hit_alpha_mean"] < 1e-4
    assert audit["ray_contract"] == (
        "exact_owner_complete_lineage_analytic_hit_and_"
        "prehit_transmittance"
    )
    assert model.opacity_logits.grad is not None
    # Gradient descent must increase exact-owner optical mass even from an
    # almost transparent initialization.
    assert model.opacity_logits.grad[0, 0] < 0


def test_dynamic_observation_ray_likelihood_rewards_owner_hit_mass():
    def model(opacity: float, z: float = 2.0):
        payload = {
            "version": "independent_sfm_semantic_canopy_volume_v1",
            "centers": torch.tensor([[0.0, 0.0, z]]),
            "scales": torch.tensor([[0.1, 0.1, 0.1]]),
            "colors": torch.tensor([[0.2, 0.4, 0.6]]),
            "opacities": torch.tensor([[opacity]]),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "layer_role": torch.tensor(
                [LAYER_DYNAMIC_LEAF], dtype=torch.int8
            ),
            "support_camera_ids": torch.tensor(
                [[7]], dtype=torch.int32
            ),
            "observation_camera_ids": torch.tensor(
                [[7]], dtype=torch.int32
            ),
            "observation_uv": torch.tensor([[[0.5, 0.5]]]),
            "observation_depth": torch.tensor([[2.0]]),
        }
        result = VolumetricFoliageModel(1, device="cpu")
        result.initialize_from_volume_state(payload)
        return result

    camera = SimpleNamespace(
        colmap_id=7,
        world_view_transform=torch.eye(4),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )
    low = model(0.02)
    high = model(0.80)
    before = model(0.80, z=1.0)

    low_loss, _, low_audit = _dynamic_observation_factor(
        low, camera, torch.zeros(low.dynamic_rank)
    )
    high_loss, _, high_audit = _dynamic_observation_factor(
        high, camera, torch.zeros(high.dynamic_rank)
    )
    _, _, before_audit = _dynamic_observation_factor(
        before, camera, torch.zeros(before.dynamic_rank)
    )

    assert high_audit["hit_alpha_mean"] > low_audit["hit_alpha_mean"]
    assert high_loss < low_loss
    assert before_audit["prehit_free"] > high_audit["prehit_free"]


def test_dynamic_observation_factor_aggregates_every_split_descendant():
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, 2.0]]),
        "scales": torch.tensor([[0.1, 0.1, 0.1]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.10]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_role": torch.tensor(
            [LAYER_DYNAMIC_LEAF], dtype=torch.int8
        ),
        "track_id": torch.tensor([41], dtype=torch.int64),
        "support_camera_ids": torch.tensor([[7]], dtype=torch.int32),
        "observation_camera_ids": torch.tensor(
            [[7]], dtype=torch.int32
        ),
        "observation_uv": torch.tensor([[[0.5, 0.5]]]),
        "observation_depth": torch.tensor([[2.0]]),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    event = model.split(torch.tensor([0]))
    assert event["children"] == 2
    assert model.track_id.tolist() == [41, 41]
    camera = SimpleNamespace(
        colmap_id=7,
        world_view_transform=torch.eye(4),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )

    _, _, audit = _dynamic_observation_factor(
        model,
        camera,
        torch.zeros(model.dynamic_rank),
        maximum_observations=1,
    )

    assert audit["available"] == 1
    assert audit["matched"] == 1
    assert audit["matched_descendants"] == 2


def test_dynamic_observation_factor_rotates_through_dense_view_births():
    count = 5
    centers = torch.tensor(
        [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0],
         [0.3, 0.0, 2.0], [0.4, 0.0, 2.0]]
    )
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": centers,
        "scales": torch.full((count, 3), 0.1),
        "colors": torch.full((count, 3), 0.4),
        "opacities": torch.full((count, 1), 0.015),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count
        ),
        "layer_role": torch.full(
            (count,), LAYER_DYNAMIC_LEAF, dtype=torch.int8
        ),
        "support_camera_ids": torch.full(
            (count, 1), 7, dtype=torch.int32
        ),
        "observation_camera_ids": torch.full(
            (count, 1), 7, dtype=torch.int32
        ),
        "observation_uv": torch.stack(
            [
                centers[:, 0] * 0.5 + 0.5,
                torch.full((count,), 0.5),
            ],
            dim=1,
        )[:, None],
        "observation_depth": torch.full((count, 1), 2.0),
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    camera = SimpleNamespace(
        colmap_id=7,
        world_view_transform=torch.eye(4),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )
    covered = torch.zeros(count, dtype=torch.bool)
    for update in range(3):
        _, observed, audit = _dynamic_observation_factor(
            model,
            camera,
            torch.zeros(model.dynamic_rank),
            maximum_observations=2,
            sample_update=update,
        )
        assert audit["available"] == count
        covered |= observed
    assert covered.all()


def test_surface_resume_capture_moves_parameters_buffers_and_metadata():
    capture = (
        0,
        torch.nn.Parameter(torch.ones(2, 3)),
        torch.nn.Parameter(torch.ones(2, 1, 3)),
        torch.nn.Parameter(torch.ones(2, 15, 3)),
        torch.nn.Parameter(torch.ones(2, 2)),
        torch.nn.Parameter(torch.ones(2, 4)),
        torch.nn.Parameter(torch.ones(2, 1)),
        torch.ones(2),
        torch.ones(2, 1),
        torch.ones(2, 1),
        {"state": {}, "param_groups": []},
        1.0,
        {
            "source_type": torch.ones(2, dtype=torch.int16),
            "label": "preserved",
        },
    )
    moved = _surface_capture_to_device(
        capture, torch.device("cpu")
    )

    assert all(
        moved[index].device.type == "cpu" for index in range(1, 10)
    )
    assert moved[12]["source_type"].device.type == "cpu"
    assert moved[12]["label"] == "preserved"
    assert all(
        isinstance(moved[index], torch.nn.Parameter)
        and moved[index].is_leaf
        for index in range(1, 7)
    )


def test_geometry_losses_do_not_propagate_invalid_evidence():
    depth = torch.tensor(
        [[[float("nan"), 0.0], [2.0, 4.0]]],
        requires_grad=True,
    )
    package = SimpleNamespace(
        depth=depth,
        normal_world=torch.tensor(
            [
                [[float("nan"), 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
    )
    evidence = {
        "chart_depth": torch.tensor(
            [[[float("nan"), 2.0], [2.5, 4.5]]]
        ),
        "chart_weight": torch.ones(1, 2, 2),
        "plane_depth": torch.tensor(
            [[[1.0, float("inf")], [2.5, 4.5]]]
        ),
        "plane_weight": torch.ones(1, 2, 2),
        "plane_normal_world": torch.tensor(
            [
                [[float("nan"), 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
        "rho_mean": torch.tensor(
            [[[float("nan"), 0.5], [0.5, 0.25]]]
        ),
        "rho_variance": torch.tensor(
            [[[float("nan"), 0.1], [0.1, 0.1]]]
        ),
        "support_view_count": torch.full((1, 2, 2), 3.0),
        "mono_depth": torch.tensor(
            [[[float("nan"), 1.0], [2.0, 3.0]]]
        ),
    }
    args = SimpleNamespace(
        geometry_weight=0.08,
        plane_weight=0.06,
        normal_weight=0.025,
        ordinal_weight=0.015,
    )
    loss, values = _geometry_losses(
        package, evidence, torch.ones(2, 2), args
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    loss.backward()
    assert torch.isfinite(depth.grad).all()


def test_chart_owner_uses_single_support_metric_inverse_depth():
    depth = torch.full((1, 2, 2), 4.0, requires_grad=True)
    package = SimpleNamespace(
        depth=depth,
        normal_world=torch.zeros(3, 2, 2),
    )
    evidence = {
        "chart_depth": torch.full((1, 2, 2), 2.0),
        "chart_weight": torch.ones(1, 2, 2),
        "rho_mean": torch.full((1, 2, 2), 0.5),
        "rho_variance": torch.full((1, 2, 2), 0.01),
        "source_bitmask": torch.full(
            (1, 2, 2), 2, dtype=torch.uint8
        ),
        # A calibrated Chart is a valid metric observation even when it has
        # no second independent camera in the cache.
        "support_view_count": torch.ones(1, 2, 2),
    }
    args = SimpleNamespace(
        geometry_weight=0.12,
        plane_weight=0.10,
        normal_weight=0.04,
        ordinal_weight=0.015,
    )
    loss, values = _geometry_losses(
        package, evidence, torch.ones(2, 2), args
    )
    assert values["chart"] == 0.0
    assert values["inverse"] > 0.0
    assert values["chart_pixels"] == 4
    assert values["inverse_pixels"] == 4
    assert values["raw_chart_fallback_pixels"] == 0
    loss.backward()
    assert depth.grad.abs().sum() > 0


def test_fused_cache_drops_invalid_boundary_instead_of_raw_fallback():
    depth = torch.full((1, 2, 2), 4.0, requires_grad=True)
    package = SimpleNamespace(
        depth=depth,
        normal_world=torch.zeros(3, 2, 2),
    )
    evidence = {
        # If this atlas edge leaked into the loss it would dominate.
        "chart_depth": torch.full((1, 2, 2), 1e-5),
        "chart_weight": torch.ones(1, 2, 2),
        "rho_mean": torch.full((1, 2, 2), 0.5),
        "rho_variance": torch.full((1, 2, 2), 0.01),
        "source_bitmask": torch.full(
            (1, 2, 2), 2, dtype=torch.uint8
        ),
        "support_view_count": torch.ones(1, 2, 2),
    }
    evidence["rho_mean"][0, 0, 0] = 0
    args = SimpleNamespace(
        geometry_weight=0.12,
        plane_weight=0.10,
        normal_weight=0.04,
        ordinal_weight=0.015,
    )
    loss, values = _geometry_losses(
        package, evidence, torch.ones(2, 2), args
    )
    assert torch.isfinite(loss)
    assert values["inverse_pixels"] == 3
    assert values["chart"] == 0.0
    assert values["raw_chart_fallback_pixels"] == 0


def test_chart_native_factor_downsamples_prediction_not_measurement():
    evidence = OutdoorGeometryEvidence.__new__(OutdoorGeometryEvidence)
    evidence.frame_by_stem = {"frame00001": 0}
    evidence.chart = {
        "active": np.asarray([True]),
        "depth": np.full((1, 2, 2), 4.0, dtype=np.float32),
        "confidence": np.ones((1, 2, 2), dtype=np.float32),
        "reference_mask": np.ones((1, 2, 2), dtype=bool),
    }
    evidence.consumed = {"chart_native_factor": 0}
    depth = torch.full((4, 4), 4.0, requires_grad=True)
    package = SimpleNamespace(
        surface_depth=depth[None],
        surface_alpha=torch.ones(1, 4, 4),
    )
    matched, audit = evidence.chart_native_factor(
        "frame00001.png", package, torch.ones(4, 4)
    )
    assert audit["pixels"] == 4
    assert matched < 1e-6
    mismatched_package = SimpleNamespace(
        surface_depth=(depth * 2.0)[None],
        surface_alpha=torch.ones(1, 4, 4),
    )
    mismatched, _ = evidence.chart_native_factor(
        "frame00001.png", mismatched_package, torch.ones(4, 4)
    )
    assert mismatched > 0
    evidence.chart["precision"] = np.full(
        (1, 2, 2), 0.1, dtype=np.float32
    )
    low_trust, low_trust_audit = evidence.chart_native_factor(
        "frame00001.png", mismatched_package, torch.ones(4, 4)
    )
    assert low_trust < 0.2 * mismatched
    assert low_trust_audit["absolute_precision"] < 0.11
    mismatched.backward()
    assert depth.grad.abs().sum() > 0


def test_dense_mast3r_pointmap_factor_uses_fixed_camera_world_depth():
    evidence = OutdoorGeometryEvidence.__new__(OutdoorGeometryEvidence)
    evidence.pointmap_records = {"frame00001": {"path": "unused"}}
    evidence._pointmap_cache = {
        "frame00001": (
            np.asarray(
                [
                    [-0.5, -0.5, 4.0],
                    [0.5, -0.5, 4.0],
                    [-0.5, 0.5, 4.0],
                    [0.5, 0.5, 4.0],
                ],
                dtype=np.float32,
            ),
            np.full(4, 2.0, dtype=np.float32),
        )
    }
    evidence.consumed = {"mast3r_pointmap_native_factor": 0}
    view = SimpleNamespace(
        image_name="frame00001.png",
        image_width=4,
        image_height=4,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
    )
    depth = torch.full((4, 4), 4.0, requires_grad=True)
    package = SimpleNamespace(
        surface_depth=depth[None],
        surface_alpha=torch.ones(1, 4, 4),
    )
    matched, audit = evidence.mast3r_pointmap_native_factor(
        view, package, torch.ones(4, 4)
    )
    assert audit["pixels"] == 4
    assert matched < 1e-6
    mismatched, _ = evidence.mast3r_pointmap_native_factor(
        view,
        SimpleNamespace(
            surface_depth=(2.0 * depth)[None],
            surface_alpha=torch.ones(1, 4, 4),
        ),
        torch.ones(4, 4),
    )
    assert mismatched > 0
    mismatched.backward()
    assert depth.grad.abs().sum() > 0
    assert evidence.consumed["mast3r_pointmap_native_factor"] == 2


def test_dense_pointmap_factor_respects_cross_sequence_precision():
    evidence = OutdoorGeometryEvidence.__new__(OutdoorGeometryEvidence)
    evidence.pointmap_records = {
        "frame00001": {
            "path": "unused",
            "cross_sequence_posterior": {"path": "unused"},
        }
    }
    evidence._pointmap_cache = {
        "frame00001": (
            np.asarray(
                [
                    [-0.5, -0.5, 4.0],
                    [0.5, -0.5, 4.0],
                    [-0.5, 0.5, 4.0],
                    [0.5, 0.5, 4.0],
                ],
                dtype=np.float32,
            ),
            np.full(4, 2.0, dtype=np.float32),
        )
    }
    evidence._pointmap_posterior_cache = {
        "frame00001": np.full((2, 2), 0.05, dtype=np.float32)
    }
    evidence._pointmap_posterior_stats = {
        "factor_calls": 0,
        "pixels": 0,
        "cross_sequence_supported_pixels": 0,
        "precision_sum": 0.0,
    }
    evidence.consumed = {"mast3r_pointmap_native_factor": 0}
    view = SimpleNamespace(
        image_name="frame00001.png",
        image_width=4,
        image_height=4,
        R=np.eye(3, dtype=np.float32),
        T=np.zeros(3, dtype=np.float32),
    )
    package = SimpleNamespace(
        surface_depth=torch.full((1, 4, 4), 8.0),
        surface_alpha=torch.ones(1, 4, 4),
    )
    low_trust, audit = evidence.mast3r_pointmap_native_factor(
        view, package, torch.ones(4, 4)
    )

    evidence._pointmap_posterior_cache["frame00001"].fill(1.0)
    full_trust, _ = evidence.mast3r_pointmap_native_factor(
        view, package, torch.ones(4, 4)
    )

    assert audit["absolute_precision"] < 0.06
    assert low_trust < 0.1 * full_trust
    assert evidence._pointmap_posterior_stats["factor_calls"] == 2


def test_foliage_unknown_ray_is_not_misclassified_as_free_space():
    def evidence(kind):
        return FoliageRayEvidence(
            {
                "camera_ids": torch.tensor([7], dtype=torch.int32),
                "pixels": torch.tensor([[0.5, 0.5]]),
                "source_image_sizes": torch.tensor(
                    [[1, 1]], dtype=torch.int32
                ),
                "free_end_depth": torch.tensor([3.0]),
                "hit_start_depth": torch.tensor([3.0]),
                "hit_end_depth": torch.tensor([4.0]),
                "observation_type": torch.tensor(
                    [kind], dtype=torch.int8
                ),
                "confidence": torch.ones(1),
            }
        )

    view = SimpleNamespace(
        colmap_id=7,
        image_width=1,
        image_height=1,
    )
    package = SimpleNamespace(
        # This mass is in front of the confirmed-free boundary and therefore
        # would be penalized if an unknown observation leaked into free-space.
        volume_depth=torch.tensor([[[1.0]]]),
        volume_alpha=torch.tensor([[[0.5]]]),
    )
    unknown_loss, unknown_audit = evidence(0).factor(view, package)
    free_loss, free_audit = evidence(-1).factor(view, package)

    evidence_audit = evidence(0).audit()
    assert {
        key: evidence_audit[key]
        for key in (
            "ray_count",
            "hit_count",
            "confirmed_free_count",
            "unknown_count",
            "camera_count",
            "explicit_source_resolution",
            "unknown_is_excluded_from_free_loss",
        )
    } == {
        "ray_count": 1,
        "hit_count": 0,
        "confirmed_free_count": 0,
        "unknown_count": 1,
        "camera_count": 1,
        "explicit_source_resolution": True,
        "unknown_is_excluded_from_free_loss": True,
    }
    assert unknown_audit["rays"] == 1
    assert unknown_loss == 0
    assert free_audit["free"] > 0
    assert free_loss > 0


def test_foliage_ray_pixels_are_scaled_from_source_to_render_resolution():
    evidence = FoliageRayEvidence(
        {
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            # Centre of source pixel (2, 2) in a 4x4 source raster maps to
            # render pixel (1, 1) in the 2x2 training raster.
            "pixels": torch.tensor([[2.0, 2.0]]),
            "source_image_sizes": torch.tensor(
                [[4, 4]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([3.0]),
            "hit_start_depth": torch.tensor([3.0]),
            "hit_end_depth": torch.tensor([4.0]),
            "observation_type": torch.tensor([-1], dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    view = SimpleNamespace(colmap_id=7, image_width=2, image_height=2)
    package = SimpleNamespace(
        volume_depth=torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]),
        volume_alpha=torch.tensor([[[0.0, 0.0], [0.0, 0.5]]]),
    )

    loss, audit = evidence.factor(view, package)

    assert audit["rays"] == 1
    assert audit["free"] > 0
    assert loss > 0


def test_foliage_ray_table_without_source_resolution_fails_closed():
    with np.testing.assert_raises_regex(
        RuntimeError, "source-raster resolution"
    ):
        FoliageRayEvidence(
            {
                "camera_ids": torch.tensor([7], dtype=torch.int32),
                "pixels": torch.tensor([[10.0, 20.0]]),
                "free_end_depth": torch.tensor([3.0]),
                "hit_start_depth": torch.tensor([3.0]),
                "hit_end_depth": torch.tensor([4.0]),
                "observation_type": torch.tensor([1], dtype=torch.int8),
                "confidence": torch.ones(1),
            }
        )


def _interval_evidence(kind):
    return FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 1], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.tensor([[50.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.tensor(
                [kind], dtype=torch.int8
            ),
            "confidence": torch.ones(1),
        }
    )


def _interval_foliage(z):
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor([[0.0, 0.0, float(z)]]),
        "scales": torch.tensor([[0.2, 0.2, 0.2]]),
        "colors": torch.tensor([[0.2, 0.4, 0.6]]),
        "opacities": torch.tensor([[0.5]]),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "ray_evidence": {
            "offsets": torch.tensor([0, 1], dtype=torch.int64)
        },
    }
    model = VolumetricFoliageModel(1, device="cpu")
    model.initialize_from_volume_state(payload)
    return model


def _interval_camera():
    return SimpleNamespace(
        colmap_id=7,
        R=np.eye(3, dtype=np.float32),
        camera_center=torch.zeros(3),
        focal_x=100.0,
        focal_y=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
    )


def test_candidate_interval_factor_rewards_mass_inside_measured_hit():
    evidence = _interval_evidence(1)
    correct = _interval_foliage(3.05)
    wrong = _interval_foliage(1.0)

    correct_loss, audit = evidence.interval_factor(
        _interval_camera(), correct
    )
    wrong_loss, _ = evidence.interval_factor(
        _interval_camera(), wrong
    )

    assert audit["candidate_evaluations"] == 1
    assert audit["hit_rays"] == 1
    assert correct_loss < wrong_loss
    correct_loss.backward()
    assert correct.opacity_logits.grad is not None
    assert correct.opacity_logits.grad.abs().sum() > 0


def test_candidate_interval_factor_converts_camera_z_on_off_axis_ray():
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.tensor([0, 1], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            # fx=50 and cx=50 below: x=80 is direction (0.6, 0, 1).
            "pixels": torch.tensor([[80.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    foliage = _interval_foliage(3.0)
    with torch.no_grad():
        foliage.xyz[0] = torch.tensor([1.8, 0.0, 3.0])
    camera = _interval_camera()
    camera.focal_x = 50.0
    camera.focal_y = 50.0

    loss, audit = evidence.interval_factor(camera, foliage)

    # The Gaussian has Euclidean range ~=3.50 but camera z=3.0.  It is a
    # measured hit only after the evidence interval is converted to the unit
    # ray coordinate used by the analytic integral.
    assert audit["depth_coordinate"] == (
        "camera_z_converted_to_unit_ray_distance"
    )
    assert audit["candidate_evaluations"] == 1
    assert audit["hit"] < 2.0
    loss.backward()
    assert foliage.xyz.grad is not None


def test_candidate_interval_tiny_hit_mass_retains_opacity_gradient():
    evidence = _interval_evidence(1)
    foliage = _interval_foliage(3.0)
    with torch.no_grad():
        foliage.opacity_logits.fill_(-20.0)

    loss, audit = evidence.interval_factor(
        _interval_camera(), foliage
    )
    loss.backward()

    assert audit["hit"] > 0
    assert foliage.opacity_logits.grad is not None
    assert foliage.opacity_logits.grad.abs().sum() > 0


def test_ownerless_observation_space_ray_is_retained_but_dynamic_only():
    evidence = FoliageRayEvidence(
        {
            # Primitive zero owns the first row.  The second row is a dense
            # observation-space posterior with no seed identity.
            "offsets": torch.tensor([0, 1], dtype=torch.int64),
            "camera_ids": torch.tensor([7, 7], dtype=torch.int32),
            "pixels": torch.full((2, 2), 50.0),
            "source_image_sizes": torch.full(
                (2, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((2,), 2.5),
            "hit_start_depth": torch.full((2,), 2.8),
            "hit_end_depth": torch.full((2,), 3.2),
            "observation_type": torch.ones(2, dtype=torch.int8),
            "confidence": torch.ones(2),
        }
    )
    audit = evidence.audit()
    assert audit["seed_bound_ray_count"] == 1
    assert audit["observation_space_ray_count"] == 1
    assert audit["factor_camera_count"] == 1

    foliage = _interval_foliage(3.0)
    loss, interval = evidence.interval_factor(
        _interval_camera(), foliage, maximum_rays=2
    )
    assert interval["rays"] == 2
    assert interval["hit_rays"] == 1
    assert interval["ownerless_hit_rays"] == 1
    assert interval["verified_ownerless_hit_rays"] == 0
    assert interval["excluded_ownerless_hit_rays"] == 1
    loss.backward()
    assert foliage.opacity_logits.grad.abs().sum() > 0


def test_ownerless_hit_without_canonical_interval_support_is_dynamic_only():
    evidence = FoliageRayEvidence(
        {
            # No seed owns this dense traversal-local observation.
            "offsets": torch.tensor([0, 0], dtype=torch.int64),
            "camera_ids": torch.tensor([7], dtype=torch.int32),
            "pixels": torch.tensor([[80.0, 50.0]]),
            "source_image_sizes": torch.tensor(
                [[100, 100]], dtype=torch.int32
            ),
            "free_end_depth": torch.tensor([2.5]),
            "hit_start_depth": torch.tensor([2.8]),
            "hit_end_depth": torch.tensor([3.2]),
            "observation_type": torch.ones(1, dtype=torch.int8),
            "confidence": torch.ones(1),
        }
    )
    # A canonical Gaussian on a different ray must not be pulled toward this
    # sequence-local leaf merely because it is the global top-k fallback.
    foliage = _interval_foliage(3.0)
    camera = _interval_camera()
    camera.focal_x = 50.0
    camera.focal_y = 50.0

    loss, audit = evidence.interval_factor(camera, foliage)

    assert audit["ownerless_hit_rays"] == 1
    assert audit["verified_ownerless_hit_rays"] == 0
    assert audit["excluded_ownerless_hit_rays"] == 1
    assert float(loss) == 0.0
    assert evidence.audit()["unique_verified_ownerless_rows"] == 0


def test_candidate_interval_descendant_index_rebuilds_after_split():
    evidence = _interval_evidence(1)
    foliage = _interval_foliage(3.0)

    _, before = evidence.interval_factor(_interval_camera(), foliage)
    foliage.split(torch.tensor([0]))
    _, after = evidence.interval_factor(_interval_camera(), foliage)

    assert before["candidate_evaluations"] == 1
    assert after["candidate_evaluations"] == 2
    assert foliage.evidence_primitive_id.tolist() == [0, 0]


def test_interval_factor_uses_all_gaussians_on_same_ray_not_only_lineage():
    evidence = _interval_evidence(1)
    wrong = _interval_foliage(1.0)
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        # Primitive zero produced the retained evidence row but lies in
        # front. Primitive one has different lineage and correctly explains
        # the measured hit interval.
        "centers": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 3.05]]
        ),
        "scales": torch.full((2, 3), 0.2),
        "colors": torch.full((2, 3), 0.4),
        "opacities": torch.full((2, 1), 0.5),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2
        ),
        "ray_evidence": {
            "offsets": torch.tensor([0, 1, 1], dtype=torch.int64)
        },
    }
    complete = VolumetricFoliageModel(1, device="cpu")
    complete.initialize_from_volume_state(payload)

    wrong_loss, _ = evidence.interval_factor(
        _interval_camera(), wrong
    )
    complete_loss, audit = evidence.interval_factor(
        _interval_camera(), complete
    )

    assert audit["candidate_evaluations"] == 2
    assert complete_loss < wrong_loss
    complete_loss.backward()
    assert complete.opacity_logits.grad[1].abs().sum() > 0


def test_candidate_interval_sampling_eventually_covers_every_row():
    count = 6
    evidence = FoliageRayEvidence(
        {
            "offsets": torch.arange(count + 1, dtype=torch.int64),
            "camera_ids": torch.full(
                (count,), 7, dtype=torch.int32
            ),
            "pixels": torch.full((count, 2), 50.0),
            "source_image_sizes": torch.full(
                (count, 2), 100, dtype=torch.int32
            ),
            "free_end_depth": torch.full((count,), 2.5),
            "hit_start_depth": torch.full((count,), 2.8),
            "hit_end_depth": torch.full((count,), 3.2),
            "observation_type": torch.ones(
                count, dtype=torch.int8
            ),
            "confidence": torch.ones(count),
        }
    )
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.tensor(
            [[0.0, 0.0, 3.0]] * count
        ),
        "scales": torch.full((count, 3), 0.2),
        "colors": torch.full((count, 3), 0.4),
        "opacities": torch.full((count, 1), 0.5),
        "quaternions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count
        ),
        "ray_evidence": {
            "offsets": torch.arange(count + 1, dtype=torch.int64)
        },
    }
    foliage = VolumetricFoliageModel(1, device="cpu")
    foliage.initialize_from_volume_state(payload)

    for update in range(3):
        evidence.interval_factor(
            _interval_camera(),
            foliage,
            maximum_rays=2,
            sample_update=update,
        )

    assert evidence.audit()["interval_unique_rows"] == count
    assert evidence.audit()["interval_row_coverage"] == 1.0


def test_candidate_interval_factor_penalizes_only_mass_before_free_boundary():
    evidence = _interval_evidence(-1)
    before = _interval_foliage(1.0)
    behind = _interval_foliage(3.0)

    before_loss, audit = evidence.interval_factor(
        _interval_camera(), before
    )
    behind_loss, _ = evidence.interval_factor(
        _interval_camera(), behind
    )

    assert audit["confirmed_free_rays"] == 1
    assert before_loss > 100 * behind_loss


def test_foliage_ray_runtime_state_survives_exact_resume():
    evidence = _interval_evidence(-1)
    foliage = _interval_foliage(3.0)
    evidence.interval_factor(_interval_camera(), foliage)
    state = evidence.capture_runtime_state()

    restored = _interval_evidence(-1)
    restored.restore_runtime_state(state)

    assert restored.audit() == evidence.audit()
    assert torch.equal(
        restored.ownerless_canonical_verified,
        evidence.ownerless_canonical_verified,
    )


def test_hybrid_volume_owner_activates_immediately_after_bootstrap():
    assert TRAINING_PROFILES["hybrid_fast"]["foliage_start"] == 0.08
    assert TRAINING_PROFILES["hybrid_quality"]["foliage_start"] == 0.12
    assert (
        dict(TRAINING_PROFILES["hybrid_quality"]["phases"])[
            "static_foliage"
        ]
        == 0.50
    )
    assert (
        TRAINING_PROFILES["hybrid_quality"]["dynamic_start"]
        == 0.50
    )
    assert (
        TRAINING_PROFILES["hybrid_handoff_quality"]["dynamic_start"]
        == 0.02
    )
    assert not _dynamic_enabled(
        599, 30_000, "hybrid_handoff_quality"
    )
    assert _dynamic_enabled(
        600, 30_000, "hybrid_handoff_quality"
    )


def test_conditioned_schedule_balances_exact_evidence_in_every_prefix():
    rgb = np.arange(8, dtype=np.int64)
    schedule, audit = _evidence_biased_schedule(
        rgb, [10, 11], fraction=0.75, seed=7
    )

    evidence = np.isin(schedule, [10, 11])
    assert evidence.tolist() == [
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert int(evidence.sum()) == 6
    assert abs(int((schedule == 10).sum()) - int((schedule == 11).sum())) <= 1
    assert audit["evidence_steps"] == 6
    assert audit["realized_fraction"] == 0.75


def test_volume_split_selection_reserves_each_tree_before_refill():
    indices = torch.arange(8)
    score = torch.tensor([9.0, 8.0, 7.0, 6.0, 2.0, 1.0, 0.5, 0.1])
    # A global top-k would allocate everything to tree 0.
    instances = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    selected = _balanced_instance_topk(
        indices, score, instances, quota=4
    )

    selected_instances = instances[selected]
    assert set(selected_instances.tolist()) == {0, 1, 2}
    assert selected.tolist()[0] == 0


def test_volume_split_selection_balances_spatial_cells_inside_one_tree():
    indices = torch.arange(8)
    score = torch.tensor([9.0, 8.0, 7.0, 6.0, 2.0, 1.0, 0.5, 0.1])
    instances = torch.zeros(8, dtype=torch.int64)
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.1, 0.0, 0.0],
        ]
    )
    selected = _balanced_instance_topk(
        indices,
        score,
        instances,
        quota=4,
        xyz=xyz,
        spatial_cell_size=1.0,
    )

    selected_cells = torch.floor(xyz[selected, 0]).to(torch.int64)
    assert set(selected_cells.tolist()) == {0, 2, 4}


def test_volume_split_selection_reserves_exact_owner_contexts():
    indices = torch.arange(9)
    score = torch.tensor(
        [100.0, 90.0, 80.0, 70.0, 3.0, 2.0, 1.0, 0.5, 0.1]
    )
    instances = torch.zeros(9, dtype=torch.int64)
    contexts = torch.tensor([10, 10, 10, 10, 20, 20, 30, 30, 30])
    selected = _balanced_instance_topk(
        indices,
        score,
        instances,
        quota=4,
        context_id=contexts,
    )

    assert len(selected) == 4
    assert set(contexts[selected].tolist()) == {10, 20, 30}


def test_extra_surface_topology_uses_nonzero_rigid_weight():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_unified_outdoor_teacher.py"
    ).read_text()
    assert 'topology_task["w_plane"]' in source
    assert (
        'topology_task["p_rigid"]\n'
        '                * topology_task["w_topology"]'
    ) not in source


def test_canopy_topology_signal_tracks_target_high_frequency():
    task = {"w_topology": torch.ones(8, 8)}
    flat = torch.full((3, 8, 8), 0.5)
    checker = (
        torch.arange(8)[:, None] + torch.arange(8)[None, :]
    ).remainder(2).float()
    checker = checker[None].repeat(3, 1, 1)

    assert (
        _canopy_topology_signal(checker, task).mean()
        > _canopy_topology_signal(flat, task).mean()
    )


def test_surface_canopy_ownership_only_penalizes_front_layer():
    task = {
        "p_canopy_core": torch.ones(2, 2),
        "p_boundary_uncertain": torch.zeros(2, 2),
    }

    def package(surface_depth, volume_depth, volume_alpha=0.8):
        return SimpleNamespace(
            surface_alpha=torch.ones(1, 2, 2),
            volume_alpha=torch.full((1, 2, 2), volume_alpha),
            surface_depth=torch.full((1, 2, 2), surface_depth),
            volume_depth=torch.full((1, 2, 2), volume_depth),
        )

    in_front = _soft_surface_canopy_conflict(
        package(1.0, 2.0), task
    )
    behind = _soft_surface_canopy_conflict(
        package(3.0, 2.0), task
    )
    gap_ray = _soft_surface_canopy_conflict(
        package(3.0, 0.0, volume_alpha=0.0), task
    )

    assert in_front > 20 * behind
    assert behind < gap_ray
    assert gap_ray < 0.03
