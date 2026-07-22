"""Small contracts for calibrated-image rectification.

MASt3R's image preprocessing may require a rectified image whose principal
point and focal differ from the source camera.  The image and its intrinsics
are one indivisible contract: changing only one of them silently creates a
geometrically inconsistent SfM scene.
"""

from __future__ import annotations

import numpy as np


def rectification_is_identity(
    source_intrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    atol: float = 1e-7,
) -> bool:
    """Whether a calibrated rectification leaves image pixels unchanged.

    Shapes use OpenCV's ``(height, width)`` convention.  A no-op transform is
    intentionally detected before calling ``cv2.remap`` so identity-calibrated
    datasets retain byte-identical source imagery in strict ablations.
    """
    return (
        tuple(map(int, source_shape)) == tuple(map(int, target_shape))
        and np.allclose(
            np.asarray(source_intrinsics, dtype=np.float64),
            np.asarray(target_intrinsics, dtype=np.float64),
            rtol=0.0,
            atol=float(atol),
        )
    )
