import torch

from outdoor.hybrid_gaussian_renderer import (
    LAYER_CANONICAL_CROWN,
    LAYER_DYNAMIC_LEAF,
    LAYER_STATIC_SKELETON,
)
from scripts.upgrade_static_foliage_seed import upgrade_payload


def test_upgrade_promotes_only_strong_existing_track_and_rebuilds_groups():
    payload = {
        "centers": torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.21, 0.0, 0.0]]
        ),
        "layer_role": torch.tensor(
            [LAYER_CANONICAL_CROWN, LAYER_CANONICAL_CROWN, LAYER_DYNAMIC_LEAF],
            dtype=torch.int8,
        ),
        "track_id": torch.tensor([11, 12, -1]),
        "tree_instance_id": torch.zeros(3, dtype=torch.int32),
        "support_view_count": torch.tensor([4, 2, 1], dtype=torch.int16),
        "support_sequence_count": torch.tensor([2, 2, 1], dtype=torch.int16),
        "reprojection_error": torch.tensor([0.5, 0.5, 0.0]),
        "occupancy_probability": torch.tensor([0.9, 0.9, 0.9]),
        "track_linearity": torch.tensor([2.2, 2.2, 1.0]),
        "static_skeleton_confidence": torch.zeros(3),
        "opacities": torch.full((3, 1), 0.025),
        "replacement_group": torch.tensor([0, 1, 1]),
    }
    result, audit = upgrade_payload(payload)
    assert result["layer_role"].tolist() == [
        LAYER_STATIC_SKELETON,
        LAYER_CANONICAL_CROWN,
        LAYER_DYNAMIC_LEAF,
    ]
    assert result["replacement_group"].tolist() == [-1, 0, 0]
    assert float(result["static_skeleton_confidence"][0]) > 0
    assert torch.isclose(
        result["opacities"][0, 0], torch.tensor(0.04)
    )
    assert audit["promoted_track_rows"] == 1
    assert audit["skeleton_after"] == 1
