from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "2d-gaussian-splatting"))

from planes.semantic_support import minimum_plane_area, plane_support_mask  # noqa: E402


def test_plane_minimum_area_scales_with_semantic_structural_support():
    image_shape = (100, 100)
    support = np.zeros(image_shape, dtype=bool)
    support[20:30, 20:70] = True

    assert support.sum() == 500
    assert minimum_plane_area(support, min_size_ratio=0.01) == 5
    assert minimum_plane_area(
        plane_support_mask(None, image_shape),
        min_size_ratio=0.01,
    ) == 100


def test_plane_semantic_support_is_a_hard_shape_checked_gate():
    semantic = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    support = plane_support_mask(semantic, (2, 2))
    assert support.dtype == bool
    assert support.tolist() == [[True, False], [False, True]]
    with pytest.raises(ValueError, match="does not match"):
        plane_support_mask(semantic, (3, 2))
