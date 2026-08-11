import inspect
from types import SimpleNamespace

import pytest
import torch

import outdoor.hybrid_teacher_api as api
from outdoor.hybrid_teacher_api import (
    HybridTeacher,
    STATIC_RAY_NORMALIZED_OPTICAL_POLICY,
    SUPPORTED_TEACHER_PROTOCOLS,
    _apply_deployment_surface_geometry,
    _static_ray_normalized_optical_mixture,
    _static_surface_evidence_mixture,
    resolve_deployment_optical_contract,
    teacher_branch_validity,
)
from scene.gaussian_model import GaussianModel
from outdoor.hybrid_gaussian_renderer import HybridRenderOutput


def _optical_test_output(
    rgb: float, volume_alpha: float
) -> HybridRenderOutput:
    image = torch.full((3, 2, 3), float(rgb))
    alpha = torch.full((1, 2, 3), float(volume_alpha))
    rows = torch.tensor([1.0, 0.0])
    return HybridRenderOutput(
        render=image,
        alpha=alpha,
        depth=torch.ones(1, 2, 3),
        normal_world=image,
        median_depth=torch.ones(1, 2, 3),
        distortion=torch.zeros(1, 2, 3),
        radii=rows,
        means2d=None,
        structural_count=1,
        responsibility=rows,
        gate_responsibility=rows,
        surface_alpha=torch.zeros(1, 2, 3),
        volume_alpha=alpha,
        surface_depth=torch.ones(1, 2, 3),
        volume_depth=torch.ones(1, 2, 3),
        surface_means2d=None,
        volume_means2d=None,
        volume_replacement=rows,
    )


def test_checkpoint_deployment_geometry_overrides_only_render_geometry():
    model = GaussianModel(3)
    model._xyz = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    model._scaling = torch.nn.Parameter(
        torch.zeros(2, 2), requires_grad=False
    )
    model._rotation = torch.nn.Parameter(
        torch.zeros(2, 4), requires_grad=False
    )
    payload = {
        "version": "native-2dgs-deployment-geometry-v1",
        "xyz": torch.ones(2, 3),
        "scaling": torch.full((2, 2), -2.0),
        "rotation": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        "point_count": 2,
        "chart_baked_rows": 1,
        "resume_surface_unchanged": True,
    }

    audit = _apply_deployment_surface_geometry(model, payload)

    assert audit["applied"]
    assert audit["chart_baked_rows"] == 1
    torch.testing.assert_close(model._xyz, payload["xyz"])
    torch.testing.assert_close(model._scaling, payload["scaling"])
    torch.testing.assert_close(model._rotation, payload["rotation"])


def test_checkpoint_deployment_geometry_rejects_row_mismatch():
    model = GaussianModel(3)
    model._xyz = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    model._scaling = torch.nn.Parameter(
        torch.zeros(2, 2), requires_grad=False
    )
    model._rotation = torch.nn.Parameter(
        torch.zeros(2, 4), requires_grad=False
    )
    with pytest.raises(RuntimeError, match="point count differs"):
        _apply_deployment_surface_geometry(
            model,
            {
                "version": "native-2dgs-deployment-geometry-v1",
                "point_count": 1,
            },
        )


def test_static_ray_normalized_optical_mixture_uses_optical_responsibility():
    envelope = _optical_test_output(0.0, 0.2)
    detail = _optical_test_output(1.0, 0.2)
    mixed, detail_weight = _static_ray_normalized_optical_mixture(
        envelope, detail
    )
    torch.testing.assert_close(detail_weight, torch.full_like(detail_weight, 0.5))
    torch.testing.assert_close(mixed.render, torch.full_like(mixed.render, 0.5))


def test_static_ray_normalized_optical_mixture_preserves_single_owner():
    envelope = _optical_test_output(0.25, 0.35)
    detail = _optical_test_output(0.9, 0.0)
    mixed, detail_weight = _static_ray_normalized_optical_mixture(
        envelope, detail
    )
    torch.testing.assert_close(detail_weight, torch.zeros_like(detail_weight))
    torch.testing.assert_close(mixed.render, envelope.render)


def test_static_ray_normalized_optical_mixture_is_split_mass_invariant():
    # Two alpha=0.2 contributors have combined optical depth
    # -log((1-.2)^2), hence combined alpha 1-(1-.2)^2.
    split_invariant_alpha = 1.0 - (1.0 - 0.2) ** 2
    envelope = _optical_test_output(0.1, split_invariant_alpha)
    detail = _optical_test_output(0.8, 0.2)
    _, detail_weight = _static_ray_normalized_optical_mixture(
        envelope, detail
    )
    expected = 1.0 / 3.0
    torch.testing.assert_close(
        detail_weight, torch.full_like(detail_weight, expected)
    )


def test_static_optical_prior_regularizes_only_two_owner_rays():
    envelope = _optical_test_output(0.1, 0.05)
    detail = _optical_test_output(0.8, 0.8)
    _, raw_weight = _static_ray_normalized_optical_mixture(
        envelope, detail
    )
    _, regularized_weight = _static_ray_normalized_optical_mixture(
        envelope, detail, symmetric_optical_prior=0.25
    )
    assert bool((regularized_weight < raw_weight).all())
    assert bool((regularized_weight > 0.5).all())

    absent_detail = _optical_test_output(0.8, 0.0)
    _, single_owner_weight = _static_ray_normalized_optical_mixture(
        envelope,
        absent_detail,
        symmetric_optical_prior=0.25,
    )
    torch.testing.assert_close(
        single_owner_weight, torch.zeros_like(single_owner_weight)
    )


def test_static_deployment_contract_defaults_to_retained_two_pass_policy():
    state = {
        "training_contract": {
            "reconstruction_target": "static",
            "deployment_static_contract": {
                "optical_replacement_policy": (
                    STATIC_RAY_NORMALIZED_OPTICAL_POLICY
                ),
                "optical_responsibility_prior": 0.25,
            },
        }
    }
    resolved = resolve_deployment_optical_contract(state)
    assert resolved["policy"] == STATIC_RAY_NORMALIZED_OPTICAL_POLICY
    assert resolved["optical_responsibility_prior"] == pytest.approx(0.25)
    assert resolved["source"] == "checkpoint_deployment_contract"
    assert not resolved["explicit_override"]

    overridden = resolve_deployment_optical_contract(
        state, requested_policy="view_depth_local"
    )
    assert overridden["policy"] == "view_depth_local"
    assert overridden["optical_responsibility_prior"] == 0.0
    assert overridden["source"] == "explicit_override"


def test_static_api_uses_checkpoint_deployment_compositor_by_default(
    monkeypatch,
):
    policies = []

    def fake_render(*_args, **kwargs):
        policies.append(kwargs["optical_replacement_policy"])
        return _optical_test_output(0.5, 0.2)

    monkeypatch.setattr(api, "render_hybrid", fake_render)
    monkeypatch.setattr(
        api,
        "composite_white_background",
        lambda image, _alpha, _sky: image,
    )
    teacher = HybridTeacher(
        surface=SimpleNamespace(get_xyz=torch.zeros(1, 3)),
        foliage=SimpleNamespace(
            layer_role=torch.zeros(3, dtype=torch.int8),
            static_skeleton_mask=torch.tensor([True, False, False]),
            persistent_envelope_mask=torch.tensor([False, True, False]),
            static_leaf_mask=torch.tensor([False, False, True]),
        ),
        sky=lambda _camera: torch.ones(3, 2, 3),
        appearance=object(),
        state={
            "iteration": 2,
            "training_contract": {
                "reconstruction_target": "static",
                "branch_activation": {"foliage_iteration": 1},
                "deployment_static_contract": {
                    "optical_replacement_policy": (
                        STATIC_RAY_NORMALIZED_OPTICAL_POLICY
                    ),
                    "optical_responsibility_prior": 0.25,
                },
            },
        },
    )
    rendered = teacher.render(SimpleNamespace(), conditioned=False)
    assert policies == ["disabled", "disabled"]
    assert (
        rendered["optical_compositing_policy"]
        == STATIC_RAY_NORMALIZED_OPTICAL_POLICY
    )
    assert rendered["optical_responsibility_prior"] == pytest.approx(0.25)
    assert not rendered["optical_compositing_contract"]["explicit_override"]


def test_static_surface_evidence_is_soft_and_depth_ordered():
    canopy = _optical_test_output(0.0, 0.5)
    canopy.volume_depth.fill_(2.0)
    surface = _optical_test_output(1.0, 0.0)
    surface.alpha.fill_(0.8)
    surface.surface_alpha.fill_(0.8)
    surface.depth.fill_(1.0)
    mixed, weight = _static_surface_evidence_mixture(canopy, surface)
    assert bool((weight > 0.5).all())
    torch.testing.assert_close(mixed.render, weight.expand_as(mixed.render))

    surface.depth.fill_(4.0)
    _, behind_weight = _static_surface_evidence_mixture(canopy, surface)
    assert bool((behind_weight < 0.01).all())


def test_static_surface_evidence_preserves_unopposed_surface():
    canopy = _optical_test_output(0.0, 0.0)
    surface = _optical_test_output(0.75, 0.0)
    surface.alpha.fill_(0.8)
    surface.surface_alpha.fill_(0.8)
    mixed, weight = _static_surface_evidence_mixture(canopy, surface)
    torch.testing.assert_close(weight, torch.ones_like(weight))
    torch.testing.assert_close(mixed.render, surface.render)


def test_causal_repair_teacher_protocol_is_publicly_loadable():
    assert (
        "cambridge_native_hybrid_teacher_v4_causal_repair"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v5_contract_closed"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v6_independent_observation_contract"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v7_candidate_ray_posterior_contract"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v8_global_ray_transmittance_contract"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v9_optical_mass_handoff_contract"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v10_dense_ray_bandwidth_contract"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v11_demand_limited_volume_topology"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v12_owner_exclusive_dynamic_gradients"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v13_soft_temporal_dynamic_gradients"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v14_parameter_family_ownership"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v15_retirement_only_fallback_opacity"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v16_base_residual_dynamic_ownership"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v17_sequence_graph_continuous_retirement"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v18_optical_mass_dynamic_replacement"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v19_branch_isolated_rgb_ownership"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v20_mature_surface_continuation"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v22_conditioned_owner_isolation"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v23_evidence_adaptive_role_capacity"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v24_population_normalized_role_capacity"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v25_schedule_preserving_surface_handoff"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v26_absolute_schedule_independent_posterior"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v27_pixel_and_primitive_owned_conditioning"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v28_local_temporal_mass_conservation"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v29_context_adaptive_volume_topology"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v32_preserved_handoff_topology_contract"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v33_exact_ray_camera_plane_topology"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v34_exact_owner_optical_mass_calibration"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v35_atomic_volume_replace_split"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v36_counterfactual_transparency_complete_ray_epoch"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v37_exact_per_camera_ray_epoch_topology_settle"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v38_resume_aware_ray_epoch_topology_settle"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v39_conditioned_topology_settle"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v40_projected_optical_footprint"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v46_static_detail_isolated_"
        "ownership_topology"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v47_static_staged_detail_"
        "ownership_topology"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v48_static_scene_snapshot_"
        "detail_ownership"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v49_static_scene_snapshot_"
        "canonical_detail_ownership"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v50_static_scene_snapshot_"
        "canonical_detail_schedule"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v51_static_scene_snapshot_"
        "visible_canonical_detail_schedule"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v52_static_ray_local_optical_handoff"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v64_static_role_evidence_mature_"
        "replace_split_surface"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v65_static_residual_topology_"
        "and_weak_rigid_completion"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v70_persistent_detail_"
        "lineage_debt_safe_local_handoff"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v71_static_optical_mass_"
        "ownership_closed"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v72_evidence_continuous_"
        "static_detail_mass"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v73_topology_safe_"
        "static_detail_audit"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v74_single_view_static_"
        "occupancy_lifecycle"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v75_single_view_birth_"
        "deferred_retirement"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v76_absolute_bandwidth_"
        "topology_capacity"
        in SUPPORTED_TEACHER_PROTOCOLS
    )
    assert (
        "cambridge_native_hybrid_teacher_v85_sequence_intrinsic_color_coverage"
        in SUPPORTED_TEACHER_PROTOCOLS
    )


def test_teacher_api_distinguishes_surface_only_from_canonical_mixed_render():
    parameters = inspect.signature(api.HybridTeacher.render).parameters
    assert "surface_only" in parameters
    assert parameters["surface_only"].default is False


def test_teacher_with_embedded_renderer_hashes_fails_closed_on_mismatch():
    historical = api._validate_render_implementation({})
    assert not historical["exact"]
    with pytest.raises(RuntimeError, match="hybrid_renderer"):
        api._validate_render_implementation(
            {
                "implementation_hashes": {
                    "appearance_uncertainty": api.sha256_file(
                        api.REPO_ROOT
                        / "outdoor/appearance_uncertainty.py"
                    ),
                    "dataset_reader": api.sha256_file(
                        api.SURFEL_ROOT / "scene/dataset_readers.py"
                    ),
                    "gaussian_model": api.sha256_file(
                        api.SURFEL_ROOT / "scene/gaussian_model.py"
                    ),
                    "hybrid_renderer": "stale",
                    "mixed_forward_cuda": api.sha256_file(
                        api.SURFEL_ROOT
                        / (
                            "submodules/diff-surfel-rasterization/"
                            "cuda_rasterizer/forward.cu"
                        )
                    ),
                }
            }
        )


def test_audited_render_equivalent_renderer_migration_is_narrow(monkeypatch):
    hashes = {
        "appearance_uncertainty": api.sha256_file(
            api.REPO_ROOT / "outdoor/appearance_uncertainty.py"
        ),
        "dataset_reader": api.sha256_file(
            api.SURFEL_ROOT / "scene/dataset_readers.py"
        ),
        "gaussian_model": api.sha256_file(
            api.SURFEL_ROOT / "scene/gaussian_model.py"
        ),
        "hybrid_renderer": "stored-renderer",
        "mixed_forward_cuda": api.sha256_file(
            api.SURFEL_ROOT
            / (
                "submodules/diff-surfel-rasterization/"
                "cuda_rasterizer/forward.cu"
            )
        ),
    }
    runtime = api.sha256_file(
        api.REPO_ROOT / "outdoor/hybrid_gaussian_renderer.py"
    )
    monkeypatch.setattr(
        api,
        "RENDER_EQUIVALENT_IMPLEMENTATION_PAIRS",
        {
            "hybrid_renderer": {
                ("stored-renderer", runtime): "forward-equivalent test pair"
            }
        },
    )
    audit = api._validate_render_implementation(
        {"implementation_hashes": hashes}
    )
    assert audit["status"] == "render_equivalent_migration"
    assert not audit["exact"]
    assert (
        audit["render_equivalent_migrations"]["hybrid_renderer"]["reason"]
        == "forward-equivalent test pair"
    )


def test_projected_optical_footprint_repair_requires_explicit_exact_predecessor(
    monkeypatch,
):
    predecessor = api.PROJECTED_OPTICAL_FOOTPRINT_REPAIR_PREDECESSOR
    hashes = {
        "appearance_uncertainty": api.sha256_file(
            api.REPO_ROOT / "outdoor/appearance_uncertainty.py"
        ),
        "dataset_reader": api.sha256_file(
            api.SURFEL_ROOT / "scene/dataset_readers.py"
        ),
        "gaussian_model": api.sha256_file(
            api.SURFEL_ROOT / "scene/gaussian_model.py"
        ),
        "hybrid_renderer": predecessor["hybrid_renderer"],
        "mixed_forward_cuda": api.sha256_file(
            api.SURFEL_ROOT
            / "submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu"
        ),
    }
    state = {
        "protocol": predecessor["protocol"],
        "implementation_hashes": hashes,
    }
    with pytest.raises(RuntimeError, match="hybrid_renderer"):
        api._validate_render_implementation(state)
    audit = api._validate_render_implementation(
        state, allow_projected_optical_footprint_repair=True
    )
    assert audit["status"] == "projected_optical_footprint_causal_repair"
    assert not audit["exact"]
    assert audit["causal_render_repair"] is not None

    wrong = {**state, "protocol": "unrelated"}
    with pytest.raises(RuntimeError, match="hybrid_renderer"):
        api._validate_render_implementation(
            wrong, allow_projected_optical_footprint_repair=True
        )


def test_audited_dav2_source_role_migration_is_render_equivalent():
    current = {
        "appearance_uncertainty": api.sha256_file(
            api.REPO_ROOT / "outdoor/appearance_uncertainty.py"
        ),
        "dataset_reader": api.sha256_file(
            api.SURFEL_ROOT / "scene/dataset_readers.py"
        ),
        "gaussian_model": (
            "e0439788449ba69e8590a37383373f38d7ba63752a952c4ac2032a2440868015"
        ),
        "hybrid_renderer": api.sha256_file(
            api.REPO_ROOT / "outdoor/hybrid_gaussian_renderer.py"
        ),
        "mixed_forward_cuda": api.sha256_file(
            api.SURFEL_ROOT
            / (
                "submodules/diff-surfel-rasterization/"
                "cuda_rasterizer/forward.cu"
            )
        ),
    }
    audit = api._validate_render_implementation(
        {"implementation_hashes": current}
    )
    assert audit["status"] == "render_equivalent_migration"
    assert not audit["exact"]
    assert set(audit["render_equivalent_migrations"]) == {
        "gaussian_model"
    }
    assert "source role 4" in (
        audit["render_equivalent_migrations"]["gaussian_model"]["reason"]
    )


def test_rigid_stage_state_cannot_reenable_untrained_foliage(monkeypatch):
    observed = {}

    def fake_render(_camera, _surface, _foliage, **kwargs):
        observed.update(kwargs)
        image = torch.zeros(3, 2, 3)
        scalar = torch.zeros(1, 2, 3)
        return SimpleNamespace(
            render=image,
            alpha=scalar,
            depth=scalar,
            surface_depth=scalar,
            volume_depth=scalar,
            normal_world=image,
            surface_alpha=scalar,
            volume_alpha=scalar,
        )

    monkeypatch.setattr(api, "render_hybrid", fake_render)
    monkeypatch.setattr(
        api,
        "dynamic_visibility_gate",
        lambda *_args, **_kwargs: torch.ones(1),
    )
    monkeypatch.setattr(
        api,
        "composite_white_background",
        lambda image, _alpha, _sky: image,
    )
    teacher = HybridTeacher(
        surface=SimpleNamespace(get_xyz=torch.zeros(1, 3)),
        foliage=object(),
        sky=lambda _camera: torch.ones(3, 2, 3),
        appearance=object(),
        state={"training_profile": "hybrid_rigid_stage1"},
    )
    teacher.render(SimpleNamespace(), conditioned=False)
    assert observed["volume_opacity_scale"] == 0.0
    with pytest.raises(RuntimeError, match="conditioned branch"):
        teacher.render(SimpleNamespace(), task={}, conditioned=True)


def test_checkpoint_phase_controls_branch_validity():
    bootstrap = teacher_branch_validity(
        {
            "training_profile": "hybrid_quality",
            "phase": "canonical_bootstrap",
        }
    )
    assert bootstrap["canonical_surface"]
    assert not bootstrap["canonical_canopy"]
    assert not bootstrap["conditioned"]

    topology = teacher_branch_validity(
        {"training_profile": "hybrid_quality", "phase": "topology"}
    )
    assert topology["canonical_canopy"]
    assert not topology["conditioned"]

    dynamic = teacher_branch_validity(
        {
            "training_profile": "hybrid_quality",
            "phase": "dynamic_appearance",
        }
    )
    assert dynamic["canonical_canopy"]
    assert dynamic["conditioned"]


def test_handoff_branch_validity_uses_activation_not_phase_name():
    before = teacher_branch_validity(
        {
            "training_profile": "hybrid_handoff_quality",
            "phase": "topology",
            "iteration": 2_400,
            "training_contract": {"iterations": 30_000},
        }
    )
    after = teacher_branch_validity(
        {
            "training_profile": "hybrid_handoff_quality",
            "phase": "topology",
            "iteration": 3_000,
            "training_contract": {"iterations": 30_000},
        }
    )

    assert before["canonical_canopy"]
    assert not before["conditioned"]
    assert after["canonical_canopy"]
    assert after["conditioned"]
    assert after["reason"].startswith("checkpoint_branch_schedule:")


def test_branch_validity_prefers_absolute_schedule_over_requested_stop():
    state = {
        "training_profile": "hybrid_handoff_quality",
        "phase": "topology",
        "iteration": 241,
        "training_contract": {
            "schedule_horizon": 12_000,
            "branch_activation": {
                "foliage": 0.0,
                "dynamic": 0.02,
                "foliage_iteration": 1,
                "dynamic_iteration": 241,
            },
        },
    }
    validity = teacher_branch_validity(state)
    assert validity["canonical_canopy"]
    assert validity["conditioned"]
    assert "dynamic>=241" in validity["reason"]


def test_bootstrap_checkpoint_hides_untrained_canonical_volume(monkeypatch):
    observed = {}

    def fake_render(_camera, _surface, _foliage, **kwargs):
        observed.update(kwargs)
        image = torch.zeros(3, 2, 3)
        scalar = torch.zeros(1, 2, 3)
        return SimpleNamespace(
            render=image,
            alpha=scalar,
            depth=scalar,
            surface_depth=scalar,
            volume_depth=scalar,
            normal_world=image,
            surface_alpha=scalar,
            volume_alpha=scalar,
        )

    monkeypatch.setattr(api, "render_hybrid", fake_render)
    monkeypatch.setattr(
        api,
        "composite_white_background",
        lambda image, _alpha, _sky: image,
    )
    teacher = HybridTeacher(
        surface=SimpleNamespace(get_xyz=torch.zeros(1, 3)),
        foliage=object(),
        sky=lambda _camera: torch.ones(3, 2, 3),
        appearance=object(),
        state={
            "training_profile": "hybrid_quality",
            "phase": "canonical_bootstrap",
        },
    )
    teacher.render(SimpleNamespace(), conditioned=False)
    assert observed["volume_opacity_scale"] == 0.0


def test_conditioned_render_defaults_to_rendered_owner_semantics(monkeypatch):
    image = torch.zeros(3, 2, 3)
    alpha = torch.full((1, 2, 3), 0.75)
    surface_alpha = torch.full((1, 2, 3), 0.50)
    volume_alpha = torch.full((1, 2, 3), 0.25)

    def fake_render(*_args, **_kwargs):
        return SimpleNamespace(
            render=image,
            alpha=alpha,
            depth=alpha,
            surface_depth=alpha,
            volume_depth=alpha,
            normal_world=image,
            surface_alpha=surface_alpha,
            volume_alpha=volume_alpha,
        )

    class Appearance:
        def temporal_code(self, _name):
            return torch.zeros(1)

        def __call__(self, rgb, _camera, task):
            assert task["semantic_condition_source"] == "rendered_owner_alpha"
            torch.testing.assert_close(
                task["p_rigid"], torch.full((2, 3), 0.50)
            )
            torch.testing.assert_close(
                task["p_canopy"], torch.full((2, 3), 0.25)
            )
            torch.testing.assert_close(
                task["p_sky"], torch.full((2, 3), 0.25)
            )
            return rgb

    monkeypatch.setattr(api, "render_hybrid", fake_render)
    monkeypatch.setattr(
        api,
        "dynamic_visibility_gate",
        lambda *_args, **_kwargs: torch.ones(1),
    )
    monkeypatch.setattr(
        api,
        "composite_white_background",
        lambda value, _alpha, _sky: value,
    )
    teacher = HybridTeacher(
        surface=SimpleNamespace(get_xyz=torch.zeros(1, 3)),
        foliage=SimpleNamespace(xyz=torch.zeros(1, 3)),
        sky=lambda _camera: image,
        appearance=Appearance(),
        state={
            "training_profile": "hybrid_quality",
            "phase": "dynamic_appearance",
        },
    )
    camera = SimpleNamespace(image_name="view", colmap_id=0)
    teacher.render(camera, conditioned=True)
