from types import SimpleNamespace

import pytest
import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    LAYER_STATIC_SKELETON,
    VolumetricFoliageModel,
    bounded_exact_ray_render_scales,
    conserve_local_temporal_fallback_mass,
    local_optical_mass_replacement,
    projected_gaussian_cross_section,
)
from scripts.train_layered_foliage_v6 import (
    _adaptive_topology,
    _joint_replacement_match,
    _optimizer,
    _optimizer_with_migrated_state,
    _remap_candidate_values,
    _restore_geometry_evidence,
)


def _payload():
    count = 4
    return {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "centers": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
        "scales": torch.full((count, 3), 0.1),
        "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "opacities": torch.full((count, 1), 0.02),
        "colors": torch.full((count, 3), 0.5),
        "primitive_role": torch.tensor([0, 1, 1, 0], dtype=torch.int8),
        "track_linearity": torch.tensor([0.0, 3.0, 1.2, 0.0]),
        "track_id": torch.tensor([-1, 20, 21, -1]),
        "tree_instance_id": torch.tensor([2, 2, 2, 3], dtype=torch.int32),
        "support_camera_ids": torch.tensor(
            [[1, 2], [1, 3], [2, 3], [4, 5]], dtype=torch.int32
        ),
        "support_view_count": torch.tensor([3, 5, 4, 3], dtype=torch.int16),
        "support_sequence_count": torch.tensor([2, 3, 2, 2], dtype=torch.int16),
        "occupancy_probability": torch.tensor([0.7, 0.9, 0.8, 0.7]),
        "position_covariance": torch.eye(3).repeat(count, 1, 1) * 0.01,
        "reprojection_error": torch.tensor([float("nan"), 0.4, 0.8, float("nan")]),
    }


def test_metadata_survives_dynamic_clone_split_prune_and_restore():
    model = VolumetricFoliageModel(0, device="cpu")
    model.initialize_from_volume_state(_payload())
    assert model.layer_role[1].item() == LAYER_STATIC_SKELETON

    model.append_dynamic_leaves(torch.tensor([2]))
    assert model.layer_role[-1].item() == LAYER_DYNAMIC_LEAF
    assert model.track_id[-1].item() == 21

    report = model.split(torch.tensor([0, 1]))
    assert report["split_parents"] == 1
    assert report["children"] == 2
    assert torch.equal(
        report["_new_to_old"],
        torch.tensor([1, 2, 3, 4, 0, 0]),
    )
    # Split children own their local posterior instead of being pulled back
    # to their deleted parent's centre.
    assert torch.allclose(
        model.initialization_center[-2:], model.xyz[-2:]
    )
    assert (model.layer_role == LAYER_STATIC_SKELETON).sum().item() == 1

    model.prune(model.layer_role == LAYER_DYNAMIC_LEAF)
    assert not bool((model.layer_role == LAYER_DYNAMIC_LEAF).any())

    captured = model.capture()
    restored = VolumetricFoliageModel(0, device="cpu")
    restored.restore(captured)
    assert torch.equal(restored.track_id, model.track_id)
    assert torch.equal(restored.tree_instance_id, model.tree_instance_id)
    assert torch.equal(restored.support_camera_ids, model.support_camera_ids)


def test_adaptive_quaternary_split_spends_three_net_growth_slots():
    model = VolumetricFoliageModel(0, device="cpu")
    model.initialize_from_volume_state(_payload())
    parent_scale = model.scales[0].clone()
    parent_opacity = model.opacities[0].clone()

    report = model.split_adaptive(
        torch.tensor([0]),
        torch.tensor([4]),
    )

    assert report["split_parents"] == 1
    assert report["children"] == 4
    assert report["net_growth"] == 3
    assert report["quaternary_parents"] == 1
    assert len(model) == 7
    torch.testing.assert_close(
        model.scales[-4:],
        parent_scale[None].repeat(4, 1) / 2.0,
    )
    torch.testing.assert_close(
        model.opacities[-4:],
        parent_opacity.repeat(4),
    )
    assert len(torch.unique(model.xyz[-4:], dim=0)) == 4


def test_exact_ray_split_refines_camera_plane_without_tightening_depth():
    payload = _payload()
    payload["centers"] = payload["centers"][:1].clone()
    payload["scales"] = torch.tensor([[0.10, 0.20, 0.30]])
    payload["quaternions"] = payload["quaternions"][:1].clone()
    payload["opacities"] = payload["opacities"][:1].clone()
    payload["colors"] = payload["colors"][:1].clone()
    payload["primitive_role"] = payload["primitive_role"][:1].clone()
    payload["track_linearity"] = payload["track_linearity"][:1].clone()
    payload["track_id"] = payload["track_id"][:1].clone()
    payload["tree_instance_id"] = payload["tree_instance_id"][:1].clone()
    payload["support_camera_ids"] = payload["support_camera_ids"][:1].clone()
    payload["support_view_count"] = payload["support_view_count"][:1].clone()
    payload["support_sequence_count"] = payload[
        "support_sequence_count"
    ][:1].clone()
    payload["occupancy_probability"] = payload[
        "occupancy_probability"
    ][:1].clone()
    payload["position_covariance"] = torch.diag(
        torch.tensor([0.01, 0.02, 0.50])
    )[None]
    payload["reprojection_error"] = payload[
        "reprojection_error"
    ][:1].clone()
    model = VolumetricFoliageModel(0, device="cpu")
    model.initialize_from_volume_state(payload)
    before_center = model.xyz[0].clone()
    before_covariance = model.position_covariance[0].clone()

    report = model.split_adaptive(
        torch.tensor([0]),
        torch.tensor([4]),
        split_plane_normals=torch.tensor([[0.0, 0.0, 1.0]]),
    )

    assert report["camera_plane_parents"] == 1
    torch.testing.assert_close(
        model.scales,
        torch.tensor([[0.05, 0.10, 0.30]]).repeat(4, 1),
    )
    torch.testing.assert_close(
        model.position_covariance,
        before_covariance[None].repeat(4, 1, 1),
    )
    torch.testing.assert_close(
        model.xyz[:, 2], before_center[2].repeat(4)
    )
    assert len(torch.unique(model.xyz[:, :2], dim=0)) == 4


def test_candidate_state_remaps_by_stable_structural_index():
    remapped = _remap_candidate_values(
        torch.tensor([7, 2, 9, 4]),
        torch.tensor([4, 7]),
        torch.tensor([0.4, 0.7]),
        torch.ones(4),
    )
    assert torch.allclose(remapped, torch.tensor([0.7, 1.0, 1.0, 0.4]))


def test_local_replacement_requires_depth_and_counterfactual_evidence():
    depth = torch.tensor([1.0, 0.0, 0.8])
    counterfactual = torch.tensor([0.0, 1.0, 0.5])
    match = _joint_replacement_match(depth, counterfactual)
    assert torch.equal(match, torch.tensor([0.0, 0.0, 0.4]))


def test_local_optical_mass_replacement_is_split_invariant():
    roles = torch.tensor(
        [LAYER_CANONICAL_CROWN, LAYER_DYNAMIC_LEAF],
        dtype=torch.int8,
    )
    groups = torch.tensor([0, 0], dtype=torch.int32)
    opacities = torch.tensor([[0.04], [0.02]])
    scales = torch.full((2, 3), 0.1)
    parent = local_optical_mass_replacement(
        roles, groups, opacities, scales
    )

    split_roles = torch.tensor(
        [
            LAYER_CANONICAL_CROWN,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
        ],
        dtype=torch.int8,
    )
    split_groups = torch.tensor([0, 0, 0], dtype=torch.int32)
    split_opacities = torch.tensor([[0.04], [0.02], [0.02]])
    split_scales = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
        ]
    )
    split_scales[1:] /= 2.0**0.5
    children = local_optical_mass_replacement(
        split_roles,
        split_groups,
        split_opacities,
        split_scales,
    )
    assert torch.allclose(parent, children, atol=1.0e-6)


def test_projected_optical_mass_is_camera_plane_split_invariant():
    roles = torch.tensor(
        [LAYER_CANONICAL_CROWN, LAYER_DYNAMIC_LEAF], dtype=torch.int8
    )
    groups = torch.zeros(2, dtype=torch.int32)
    opacity = torch.tensor([[0.08], [0.04]])
    scales = torch.tensor([[0.2, 0.2, 0.2], [0.1, 0.2, 1.0]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    view_vectors = torch.tensor([[0.0, 0.0, 2.0]]).repeat(2, 1)
    parent = local_optical_mass_replacement(
        roles,
        groups,
        opacity,
        scales,
        rotations=rotations,
        view_vectors=view_vectors,
    )

    child_roles = torch.tensor(
        [LAYER_CANONICAL_CROWN] + [LAYER_DYNAMIC_LEAF] * 4,
        dtype=torch.int8,
    )
    child_groups = torch.zeros(5, dtype=torch.int32)
    child_opacity = torch.cat([opacity[:1], opacity[1:].repeat(4, 1)])
    child_scales = torch.cat(
        [scales[:1], torch.tensor([[0.05, 0.10, 1.0]]).repeat(4, 1)]
    )
    child_rotations = rotations[:1].repeat(5, 1)
    child_views = view_vectors[:1].repeat(5, 1)
    children = local_optical_mass_replacement(
        child_roles,
        child_groups,
        child_opacity,
        child_scales,
        rotations=child_rotations,
        view_vectors=child_views,
    )
    torch.testing.assert_close(parent, children, atol=1.0e-7, rtol=1.0e-6)


def test_projected_cross_section_excludes_depth_axis_in_owner_view():
    scales = torch.tensor([[0.1, 0.2, 1.0]])
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    view = torch.tensor([[0.0, 0.0, 2.0]])
    area = projected_gaussian_cross_section(scales, rotation, view)
    torch.testing.assert_close(area, torch.tensor([0.1 * 0.2 / 4.0]))


def test_exact_ray_render_bound_changes_only_visible_exact_ray_footprint():
    foliage = SimpleNamespace(
        dynamic_leaf_mask=torch.tensor([True, True, True, False]),
        initialization_source=torch.tensor([4, 4, 3, 4]),
        observation_camera_ids=torch.tensor(
            [[7, -1], [7, -1], [7, -1], [7, -1]]
        ),
        position_covariance=torch.eye(3).repeat(4, 1, 1),
    )
    scales = torch.tensor(
        [[0.01, 0.02, 1.0]] * 4, requires_grad=True
    )
    gate = torch.tensor([1.0, 0.25, 0.25, 0.25])
    before_covariance = foliage.position_covariance.clone()
    bounded = bounded_exact_ray_render_scales(
        foliage, scales, gate, maximum_depth_to_tangent_ratio=8.0
    )

    torch.testing.assert_close(
        bounded[0], torch.tensor([0.01, 0.02, 0.16])
    )
    torch.testing.assert_close(
        bounded[1], torch.tensor([0.01, 0.02, 0.16])
    )
    torch.testing.assert_close(bounded[2:], scales[2:])
    torch.testing.assert_close(
        foliage.position_covariance, before_covariance
    )
    bounded.sum().backward()
    assert scales.grad[1, 2].item() == 0.0


def test_local_optical_mass_replacement_accumulates_local_descendants():
    roles = torch.tensor(
        [
            LAYER_CANONICAL_CROWN,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
        ],
        dtype=torch.int8,
    )
    groups = torch.tensor([0, 0, 0], dtype=torch.int32)
    scales = torch.full((3, 3), 0.1)
    one = local_optical_mass_replacement(
        roles[:2],
        groups[:2],
        torch.tensor([[0.08], [0.01]]),
        scales[:2],
    )
    two = local_optical_mass_replacement(
        roles,
        groups,
        torch.tensor([[0.08], [0.01], [0.01]]),
        scales,
    )
    assert 0.0 < one.item() < two.item() < 1.0


def test_temporal_fallback_fills_but_cannot_exceed_local_canonical_mass():
    roles = torch.tensor(
        [
            LAYER_CANONICAL_CROWN,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
        ],
        dtype=torch.int8,
    )
    groups = torch.zeros(4, dtype=torch.int32)
    scales = torch.full((4, 3), 0.1)
    opacity = torch.tensor([[0.08], [0.02], [0.08], [0.08]])
    gate = torch.tensor([1.0, 1.0, 0.5, 0.5])

    adjusted = conserve_local_temporal_fallback_mass(
        roles, groups, opacity * gate[:, None], scales, gate
    )
    area = scales[:, 0] * scales[:, 1]
    mass = -torch.log1p(-adjusted[:, 0]) * area
    canonical_mass = mass[0]
    exact_mass = mass[1]
    fallback_mass = mass[2:].sum()

    torch.testing.assert_close(
        exact_mass + fallback_mass, canonical_mass, atol=1.0e-7, rtol=1.0e-5
    )
    torch.testing.assert_close(adjusted[1], opacity[1])
    assert bool((adjusted[2:] < opacity[2:] * 0.5).all())


def test_oversubscribed_exact_descendants_remain_direct_observations():
    roles = torch.tensor(
        [
            LAYER_CANONICAL_CROWN,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
        ],
        dtype=torch.int8,
    )
    groups = torch.zeros(3, dtype=torch.int32)
    scales = torch.full((3, 3), 0.1)
    opacity = torch.tensor([[0.04], [0.40], [0.40]])
    gate = torch.ones(3)
    adjusted = conserve_local_temporal_fallback_mass(
        roles, groups, opacity, scales, gate
    )
    torch.testing.assert_close(adjusted, opacity)


def test_temporal_fallback_mass_cap_is_split_invariant():
    parent_roles = torch.tensor(
        [LAYER_CANONICAL_CROWN, LAYER_DYNAMIC_LEAF], dtype=torch.int8
    )
    parent_groups = torch.zeros(2, dtype=torch.int32)
    parent_scales = torch.full((2, 3), 0.1)
    parent_opacity = torch.tensor([[0.04], [0.12]])
    parent_gate = torch.tensor([1.0, 0.5])
    parent = conserve_local_temporal_fallback_mass(
        parent_roles,
        parent_groups,
        parent_opacity * parent_gate[:, None],
        parent_scales,
        parent_gate,
    )

    child_roles = torch.tensor(
        [
            LAYER_CANONICAL_CROWN,
            LAYER_DYNAMIC_LEAF,
            LAYER_DYNAMIC_LEAF,
        ],
        dtype=torch.int8,
    )
    child_groups = torch.zeros(3, dtype=torch.int32)
    child_scales = torch.full((3, 3), 0.1)
    child_scales[1:] /= 2.0**0.5
    child_opacity = torch.tensor([[0.04], [0.12], [0.12]])
    child_gate = torch.tensor([1.0, 0.5, 0.5])
    children = conserve_local_temporal_fallback_mass(
        child_roles,
        child_groups,
        child_opacity * child_gate[:, None],
        child_scales,
        child_gate,
    )

    def optical_mass(alpha, scale):
        area = scale.prod(dim=1) / scale.min(dim=1).values
        return (-torch.log1p(-alpha[:, 0]) * area).sum()

    torch.testing.assert_close(
        optical_mass(parent[1:], parent_scales[1:]),
        optical_mass(children[1:], child_scales[1:]),
        atol=1.0e-7,
        rtol=1.0e-5,
    )


def test_temporal_fallback_without_live_canonical_reference_is_preserved():
    roles = torch.tensor([LAYER_DYNAMIC_LEAF], dtype=torch.int8)
    groups = torch.tensor([7], dtype=torch.int32)
    opacity = torch.tensor([[0.03]])
    adjusted = conserve_local_temporal_fallback_mass(
        roles,
        groups,
        opacity,
        torch.full((1, 3), 0.1),
        torch.tensor([0.25]),
    )
    torch.testing.assert_close(adjusted, opacity)


def test_adaptive_split_reserves_dynamic_and_canonical_roles():
    class FakeFoliage:
        def __init__(self):
            self.static_skeleton_mask = torch.zeros(20, dtype=torch.bool)
            self.support_sequence_count = torch.full((20,), 2)
            self.dynamic_leaf_mask = torch.zeros(20, dtype=torch.bool)
            self.dynamic_leaf_mask[:10] = True
            self.opacities = torch.ones(20)

        def __len__(self):
            return 20

        def split(self, indices):
            return {
                "split_parents": len(indices),
                "children": 2 * len(indices),
            }

        def prune(self, remove):
            return int(remove.sum())

    foliage = FakeFoliage()
    score = torch.linspace(1, 2, 20)
    statistics = {
        "gradient": score.clone(),
        "gradient_count": torch.ones(20),
        "residual": score.clone(),
        "contribution": torch.ones(20),
        "rigid": torch.zeros(20),
        "radius": torch.full((20,), 10.0),
    }
    args = SimpleNamespace(
        split_radius=8.0,
        maximum_gaussians=100,
        maximum_splits=8,
        dynamic_split_fraction=0.5,
        prune_opacity=0.0025,
    )
    report = _adaptive_topology(args, foliage, statistics)
    assert report["dynamic_split_parents"] == 4
    assert report["canonical_split_parents"] == 4


def test_adam_history_is_migrated_across_volume_split():
    model = VolumetricFoliageModel(0, device="cpu")
    model.initialize_from_volume_state(_payload())
    atlas = torch.nn.Parameter(torch.ones(2, 4, 4))
    appearance = torch.nn.Linear(1, 1)
    args = SimpleNamespace(
        position_lr=1e-3,
        feature_lr=1e-3,
        opacity_lr=1e-3,
        scale_lr=1e-3,
        rotation_lr=1e-3,
        dynamic_lr=1e-3,
        gate_lr=1e-3,
        appearance_lr=1e-3,
    )
    optimizer = _optimizer(args, model, atlas, appearance)
    loss = sum(
        parameter.square().sum()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()
    old_xyz_moment = optimizer.state[model.xyz]["exp_avg"].clone()
    old_atlas_moment = optimizer.state[atlas]["exp_avg"].clone()

    report = model.split(torch.tensor([0]))
    migrated = _optimizer_with_migrated_state(
        args,
        model,
        atlas,
        appearance,
        optimizer,
        report["_new_to_old"],
    )
    assert torch.equal(
        migrated.state[model.xyz]["exp_avg"],
        old_xyz_moment[report["_new_to_old"]],
    )
    assert torch.equal(
        migrated.state[atlas]["exp_avg"], old_atlas_moment
    )


def test_legacy_state_recovers_independent_ray_evidence_by_seed_center():
    payload = _payload()
    payload["ray_depth_nll"] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    payload["free_space_violation_count"] = torch.tensor(
        [1, 2, 3, 4], dtype=torch.int16
    )
    payload["unknown_view_count"] = torch.tensor(
        [4, 3, 2, 1], dtype=torch.int16
    )
    source = VolumetricFoliageModel(0, device="cpu")
    source.initialize_from_volume_state(payload)
    source.append_dynamic_leaves(torch.tensor([2]))
    captured = source.capture()
    for name in (
        "ray_depth_nll",
        "free_space_violation_count",
        "unknown_view_count",
    ):
        captured.pop(name)
    restored = VolumetricFoliageModel(0, device="cpu")
    restored.restore(captured)
    audit = _restore_geometry_evidence(restored, payload, captured)
    assert audit["matched"] == 5
    assert restored.ray_depth_nll[-1].item() == pytest.approx(0.3)
    assert restored.free_space_violation_count[-1].item() == 3
