import inspect
from types import SimpleNamespace

import pytest
import torch

import outdoor.hybrid_teacher_api as api
from outdoor.hybrid_teacher_api import (
    HybridTeacher,
    SUPPORTED_TEACHER_PROTOCOLS,
    teacher_branch_validity,
)


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
