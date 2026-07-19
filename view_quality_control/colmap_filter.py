from __future__ import annotations

from typing import Iterable

import numpy as np


def filter_points3d_tracks(
    points3d: dict,
    kept_image_ids: Iterable[int],
    *,
    min_track_length: int = 2,
) -> tuple[dict, dict[str, int]]:
    """Filter COLMAP point tracks and drop points unsupported by kept cameras."""
    kept_ids = set(int(image_id) for image_id in kept_image_ids)
    filtered = {}
    original_track_entries = 0
    kept_track_entries = 0
    for point_id, point in points3d.items():
        original_track_entries += len(point.image_ids)
        keep = np.asarray([int(image_id) in kept_ids for image_id in point.image_ids])
        image_ids = point.image_ids[keep]
        point2d_indices = point.point2D_idxs[keep]
        if len(image_ids) < min_track_length:
            continue
        filtered[point_id] = point._replace(
            image_ids=image_ids,
            point2D_idxs=point2d_indices,
        )
        kept_track_entries += len(image_ids)

    return filtered, {
        "input_points": len(points3d),
        "output_points": len(filtered),
        "input_track_entries": original_track_entries,
        "output_track_entries": kept_track_entries,
        "min_track_length": min_track_length,
    }

