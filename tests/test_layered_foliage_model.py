from types import SimpleNamespace

import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_DYNAMIC_LEAF,
    LAYER_STATIC_SKELETON,
    VolumetricFoliageModel,
)
from scripts.train_layered_foliage_v6 import (
    _adaptive_topology,
    _joint_replacement_match,
    _remap_candidate_values,
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
    assert report == {"split_parents": 1, "children": 2}
    assert (model.layer_role == LAYER_STATIC_SKELETON).sum().item() == 1

    model.prune(model.layer_role == LAYER_DYNAMIC_LEAF)
    assert not bool((model.layer_role == LAYER_DYNAMIC_LEAF).any())

    captured = model.capture()
    restored = VolumetricFoliageModel(0, device="cpu")
    restored.restore(captured)
    assert torch.equal(restored.track_id, model.track_id)
    assert torch.equal(restored.tree_instance_id, model.tree_instance_id)
    assert torch.equal(restored.support_camera_ids, model.support_camera_ids)


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
