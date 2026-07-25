import torch

from scripts.expand_foliage_replacement_audit import (
    expanded_candidate_mask,
)


def _statistics():
    return {
        "total_contribution": torch.tensor([1.0, 1.0, 1.0]),
        "support_views": torch.tensor([3, 3, 3]),
        "support_sequences": torch.tensor([2, 2, 2]),
        "canopy_responsibility": torch.tensor([0.7, 0.7, 0.01]),
        "rigid_responsibility": torch.tensor([0.8, 0.0, 0.0]),
        "maximum_view_rigid_responsibility": torch.tensor(
            [0.9, 0.0, 0.0]
        ),
        "canopy_residual": torch.tensor([0.2, 0.2, 0.2]),
        "minimum_camera_depth": torch.tensor([4.0, 4.0, 4.0]),
        "giant_radius_view_count": torch.tensor([1, 1, 1]),
        "risky_radius_view_count": torch.tensor([3, 3, 3]),
    }


def test_uv_local_expansion_admits_mixed_rigid_giant():
    mask = expanded_candidate_mask(
        _statistics(),
        minimum_support_views=2,
        minimum_support_sequences=1,
        minimum_canopy_responsibility=0.05,
        maximum_rigid_responsibility=0.2,
        maximum_view_rigid_responsibility=0.2,
        minimum_canopy_residual=0.0,
        minimum_camera_depth=1.5,
        allow_giant_candidates=True,
        uv_local_ownership=True,
        minimum_risky_radius_views=3,
    )
    assert mask.tolist() == [True, True, False]


def test_legacy_expansion_still_rejects_mixed_rigid_parent():
    mask = expanded_candidate_mask(
        _statistics(),
        minimum_support_views=2,
        minimum_support_sequences=1,
        minimum_canopy_responsibility=0.05,
        maximum_rigid_responsibility=0.2,
        maximum_view_rigid_responsibility=0.2,
        minimum_canopy_residual=0.0,
        minimum_camera_depth=1.5,
        allow_giant_candidates=True,
        uv_local_ownership=False,
        minimum_risky_radius_views=3,
    )
    assert mask.tolist() == [False, True, False]
