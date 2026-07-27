"""Geometry-aware camera support rules for outdoor foliage evidence."""

from __future__ import annotations

import re

import numpy as np


_SEQUENCE_PATTERN = re.compile(r"^(?P<sequence>[^_/]+)(?:__|/)")


def sequence_id(image_name: str) -> str:
    match = _SEQUENCE_PATTERN.match(str(image_name))
    if match:
        return match.group("sequence")
    return str(image_name).split("__", 1)[0].split("/", 1)[0]


def camera_center(qvec, tvec, qvec_to_rotation):
    world_to_camera = qvec_to_rotation(qvec)
    return -(world_to_camera.T @ np.asarray(tvec, dtype=np.float64))


def triangulation_angle_degrees(point, center_a, center_b):
    ray_a = np.asarray(point) - np.asarray(center_a)
    ray_b = np.asarray(point) - np.asarray(center_b)
    norm = np.linalg.norm(ray_a) * np.linalg.norm(ray_b)
    if norm <= 1e-12:
        return 0.0
    cosine = np.clip(np.dot(ray_a, ray_b) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def support_geometry(point, camera_ids, centers_by_id, sequence_by_id):
    camera_ids = [int(value) for value in camera_ids]
    sequences = {sequence_by_id[value] for value in camera_ids}
    max_baseline = 0.0
    max_angle = 0.0
    for offset, first in enumerate(camera_ids):
        for second in camera_ids[offset + 1 :]:
            baseline = float(
                np.linalg.norm(centers_by_id[first] - centers_by_id[second])
            )
            max_baseline = max(max_baseline, baseline)
            max_angle = max(
                max_angle,
                triangulation_angle_degrees(
                    point, centers_by_id[first], centers_by_id[second]
                ),
            )
    return {
        "camera_count": len(set(camera_ids)),
        "sequence_count": len(sequences),
        "max_baseline": max_baseline,
        "max_triangulation_angle_degrees": max_angle,
    }


def greedy_diverse_views(records, *, limit, minimum_center_distance=0.5):
    """Select high-canopy cameras while avoiding consecutive near duplicates."""
    ranked = sorted(
        records,
        key=lambda row: (-float(row["canopy_fraction"]), str(row["image_name"])),
    )
    selected = []
    sequence_counts = {}
    while ranked and len(selected) < int(limit):
        best_index = None
        best_score = None
        for index, row in enumerate(ranked):
            center = np.asarray(row["camera_center"])
            if selected:
                minimum_distance = min(
                    np.linalg.norm(center - np.asarray(item["camera_center"]))
                    for item in selected
                )
            else:
                minimum_distance = float("inf")
            sequence = row["sequence_id"]
            novelty = 1.0 / (1.0 + sequence_counts.get(sequence, 0))
            separation = min(1.0, minimum_distance / minimum_center_distance)
            score = float(row["canopy_fraction"]) * (0.5 + 0.5 * separation) * novelty
            if best_score is None or score > best_score:
                best_index, best_score = index, score
        chosen = ranked.pop(best_index)
        selected.append(chosen)
        sequence = chosen["sequence_id"]
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
    return selected


def sequence_balanced_diverse_views(
    records, *, limit, minimum_center_distance=0.5
):
    """Select real cameras without letting one traversal own the hull.

    The first pass reserves one useful, spatially diverse camera per
    sequence.  The normal greedy rule then fills the remaining capacity.
    This matters for foliage: a globally ranked canopy list can otherwise be
    entirely consumed by one traversal and make a nominal visual hull a
    single-sequence extrusion.
    """
    records = list(records)
    limit = int(limit)
    if limit <= 0 or not records:
        return []
    by_sequence = {}
    for row in records:
        by_sequence.setdefault(str(row["sequence_id"]), []).append(row)
    selected = []
    # Prefer sequences with stronger canopy support, but reserve at most one
    # view until every represented traversal has had a chance.
    sequence_order = sorted(
        by_sequence,
        key=lambda sequence: (
            -max(float(row["canopy_fraction"]) for row in by_sequence[sequence]),
            sequence,
        ),
    )
    for sequence in sequence_order[:limit]:
        candidates = greedy_diverse_views(
            by_sequence[sequence],
            limit=1,
            minimum_center_distance=minimum_center_distance,
        )
        if candidates:
            selected.extend(candidates)
    if len(selected) >= limit:
        return selected[:limit]
    selected_ids = {int(row["image_id"]) for row in selected}
    remainder = [
        row for row in records if int(row["image_id"]) not in selected_ids
    ]
    # Seed the score with already selected views by greedily choosing from the
    # remainder and rejecting near-identical duplicates afterwards.
    fill = greedy_diverse_views(
        remainder,
        limit=limit - len(selected),
        minimum_center_distance=minimum_center_distance,
    )
    return [*selected, *fill]
