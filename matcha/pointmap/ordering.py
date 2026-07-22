"""Identity-preserving helpers for MASt3R pointmap artifacts.

MASt3R writes one JSON pointmap per image.  Filesystem order is not a camera
order contract: candidate selection may reorder, subset, or reserve Chart
views.  Consumers must therefore resolve pointmaps by image identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def pointmap_paths_for_camera_filepaths(
    camera_filepaths: Sequence[str | Path],
    pointmaps_dir: str | Path,
) -> list[Path]:
    """Return pointmaps in exactly the supplied camera order.

    MASt3R pointmaps are named ``<image-stem>.json``.  Reject missing or
    duplicate identities instead of silently pairing a depth map with the
    wrong camera after a Chart subset has been reordered.
    """
    pointmaps_path = Path(pointmaps_dir)
    if not pointmaps_path.is_dir():
        raise FileNotFoundError(f"MASt3R pointmaps directory is missing: {pointmaps_path}")

    by_stem: dict[str, Path] = {}
    duplicates: set[str] = set()
    for pointmap_path in pointmaps_path.glob("*.json"):
        stem = pointmap_path.stem
        if stem in by_stem:
            duplicates.add(stem)
        else:
            by_stem[stem] = pointmap_path
    if duplicates:
        raise RuntimeError(
            "MASt3R pointmaps contain duplicate image stems: "
            f"{sorted(duplicates)[:3]}"
        )

    camera_stems = [Path(filepath).stem for filepath in camera_filepaths]
    duplicate_cameras = sorted(
        {stem for stem in camera_stems if camera_stems.count(stem) > 1}
    )
    if duplicate_cameras:
        raise RuntimeError(
            "MASt3R camera list contains duplicate image stems: "
            f"{duplicate_cameras[:3]}"
        )

    missing = sorted(stem for stem in camera_stems if stem not in by_stem)
    if missing:
        raise FileNotFoundError(
            "MASt3R pointmaps are missing camera image(s): "
            f"{missing[:3]}"
        )
    return [by_stem[stem] for stem in camera_stems]
