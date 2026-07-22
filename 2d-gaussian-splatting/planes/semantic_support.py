"""Small, dependency-free helpers for semantic-gated plane extraction."""

from __future__ import annotations

import math

import numpy as np


def plane_support_mask(
    semantic_keep: np.ndarray | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Return the pixels eligible to seed a structural plane.

    ``None`` intentionally reproduces the original all-image plane extractor.
    A supplied mask is a hard geometric gate: foliage, sky, transient and
    otherwise masked pixels cannot inflate a SAM/normal segment's area or turn
    into a plane support point.
    """
    if semantic_keep is None:
        return np.ones(image_shape, dtype=bool)
    support = np.asarray(semantic_keep, dtype=bool)
    if support.shape != tuple(image_shape):
        raise ValueError(
            "semantic plane-support mask shape does not match image: "
            f"mask={support.shape}, image={tuple(image_shape)}"
        )
    return support


def minimum_plane_area(
    support_mask: np.ndarray,
    *,
    min_size_ratio: float,
) -> int:
    """Scale the local plane-area threshold by usable structural pixels.

    A 1%-of-full-image rule silently removes a small distant facade whenever
    foliage or sky occupies most of an outdoor frame.  The correct reference
    measure is the structural semantic support that is allowed to enter plane
    fitting.  At least one pixel is retained as a defensive lower bound; later
    normal/SAM and multi-view gates still decide whether it is a real plane.
    """
    if not 0.0 < float(min_size_ratio) <= 1.0:
        raise ValueError("min_size_ratio must be in (0, 1]")
    support = np.asarray(support_mask, dtype=bool)
    if support.ndim != 2:
        raise ValueError(f"plane support mask must be 2D, got {support.shape}")
    return max(1, int(math.ceil(float(support.sum()) * float(min_size_ratio))))
