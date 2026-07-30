"""Geometry-aware camera support rules for outdoor foliage evidence."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


_SEQUENCE_PATTERN = re.compile(r"^(?P<sequence>[^_/]+)(?:__|/)")
_FRAME_PATTERN = re.compile(r"(?:frame)?(?P<frame>\d+)(?=\D*$)")


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


def _frame_index(record) -> int:
    """Return a stable temporal coordinate for one calibrated camera."""
    if "frame_index" in record:
        return int(record["frame_index"])
    match = _FRAME_PATTERN.search(str(record.get("image_name", "")))
    if match:
        return int(match.group("frame"))
    return int(record["image_id"])


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


def greedy_diverse_views(
    records,
    *,
    limit,
    minimum_center_distance=0.5,
    initial_selected=(),
):
    """Select canopy cameras with explicit spatial and temporal coverage.

    ``initial_selected`` participates in the diversity score but is not
    returned.  This is important when several tree-instance reservation
    passes share one global camera budget: without that context each pass
    independently selected the same short temporal neighbourhood.
    """
    ranked = sorted(
        records,
        key=lambda row: (-float(row["canopy_fraction"]), str(row["image_name"])),
    )
    if not ranked or int(limit) <= 0:
        return []
    selected = []
    context = list(initial_selected)
    sequence_counts = Counter(
        str(row["sequence_id"]) for row in context
    )
    temporal_extent = {}
    for row in [*ranked, *context]:
        sequence = str(row["sequence_id"])
        temporal_extent.setdefault(sequence, []).append(_frame_index(row))
    temporal_extent = {
        sequence: max(max(values) - min(values), 1)
        for sequence, values in temporal_extent.items()
    }
    centers = np.asarray(
        [row["camera_center"] for row in ranked], dtype=np.float64
    )
    sequences = np.asarray(
        [str(row["sequence_id"]) for row in ranked], dtype=object
    )
    frames = np.asarray([_frame_index(row) for row in ranked], dtype=np.int64)
    canopy = np.clip(
        np.asarray(
            [float(row["canopy_fraction"]) for row in ranked],
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    available = np.ones(len(ranked), dtype=bool)
    minimum_distance = np.full(len(ranked), np.inf, dtype=np.float64)
    minimum_temporal_distance = np.full(
        len(ranked), np.inf, dtype=np.float64
    )

    def update_context(row):
        nonlocal minimum_distance, minimum_temporal_distance
        center = np.asarray(row["camera_center"], dtype=np.float64)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(centers - center[None], axis=1),
        )
        sequence = str(row["sequence_id"])
        same_sequence = sequences == sequence
        minimum_temporal_distance[same_sequence] = np.minimum(
            minimum_temporal_distance[same_sequence],
            np.abs(frames[same_sequence] - _frame_index(row)),
        )

    for row in context:
        update_context(row)

    # The score is unchanged, but distance and temporal novelty are maintained
    # incrementally in vector form.  The former nested Python implementation
    # recomputed distance to every selected camera for every candidate at
    # every one of 256 selection steps.
    while bool(available.any()) and len(selected) < int(limit):
        separation = np.minimum(
            1.0,
            minimum_distance
            / max(float(minimum_center_distance), 1e-6),
        )
        temporal_denominator = np.asarray(
            [temporal_extent[str(value)] for value in sequences],
            dtype=np.float64,
        )
        temporal_separation = np.minimum(
            1.0,
            minimum_temporal_distance / temporal_denominator,
        )
        novelty = np.asarray(
            [
                1.0 / (1.0 + sequence_counts.get(str(value), 0))
                for value in sequences
            ],
            dtype=np.float64,
        )
        score = (
            (0.25 + 0.75 * canopy)
            * (0.10 + 0.90 * separation)
            * (0.10 + 0.90 * temporal_separation)
            * novelty
        )
        score[~available] = -np.inf
        best_index = int(np.argmax(score))
        chosen = ranked[best_index]
        available[best_index] = False
        selected.append(chosen)
        update_context(chosen)
        sequence = str(chosen["sequence_id"])
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
    return selected


def sequence_balanced_diverse_views(
    records,
    *,
    limit,
    minimum_center_distance=0.5,
    initial_selected=(),
):
    """Select real cameras without letting one traversal own the hull.

    The first pass reserves one useful, spatially diverse camera per
    sequence.  The normal greedy rule then fills the remaining capacity.
    This matters for foliage: a globally ranked canopy list can otherwise be
    entirely consumed by one traversal and make a nominal visual hull a
    single-sequence extrusion.
    """
    records = list(records)
    initial_selected = list(initial_selected)
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
            initial_selected=initial_selected,
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
        initial_selected=[*initial_selected, *selected],
    )
    return [*selected, *fill]


def instance_balanced_diverse_views(
    records,
    *,
    support_by_instance,
    limit,
    minimum_center_distance=0.5,
    maximum_reserved_per_instance=12,
):
    """Reserve posterior cameras for every observed tree instance.

    A globally diverse canopy schedule can spend nearly its entire budget on
    the largest foreground crown.  Smaller trees then retain valid candidates
    but no ray/depth factors, so their volume is optimized only by RGB.  This
    selector first allocates an equal bounded quota to each instance using
    that instance's real observation counts, then fills the remaining budget
    with the ordinary sequence-balanced schedule.
    """
    records = list(records)
    limit = int(limit)
    if limit <= 0 or not records:
        return []
    by_image_id = {int(row["image_id"]): row for row in records}
    support = {
        int(instance): Counter(
            {
                int(image_id): int(count)
                for image_id, count in dict(camera_counts).items()
                if int(count) > 0 and int(image_id) in by_image_id
            }
        )
        for instance, camera_counts in dict(support_by_instance).items()
    }
    support = {
        instance: counts for instance, counts in support.items() if counts
    }
    if not support:
        return sequence_balanced_diverse_views(
            records,
            limit=limit,
            minimum_center_distance=minimum_center_distance,
        )

    instance_count = len(support)
    quota = max(
        1,
        min(
            int(maximum_reserved_per_instance),
            limit // max(instance_count, 1),
        ),
    )
    selected = []
    selected_ids = set()
    # Instances with fewer supporting cameras go first; otherwise a large
    # crown can consume shared cameras before a small instance is represented.
    instance_order = sorted(
        support,
        key=lambda instance: (len(support[instance]), instance),
    )
    for instance in instance_order:
        counts = support[instance]
        maximum = max(counts.values())
        candidates = []
        for image_id, count in counts.items():
            if image_id in selected_ids:
                continue
            row = dict(by_image_id[image_id])
            observation_strength = float(count) / max(float(maximum), 1.0)
            row["canopy_fraction"] = max(
                float(row.get("canopy_fraction", 0.0)),
                observation_strength,
            )
            candidates.append(row)
        reserved = sequence_balanced_diverse_views(
            candidates,
            limit=min(quota, limit - len(selected)),
            minimum_center_distance=minimum_center_distance,
            initial_selected=selected,
        )
        for row in reserved:
            image_id = int(row["image_id"])
            if image_id not in selected_ids:
                selected.append(by_image_id[image_id])
                selected_ids.add(image_id)
        if len(selected) >= limit:
            return selected[:limit]

    observed_ids = {
        image_id
        for counts in support.values()
        for image_id in counts
    }
    remainder = [
        row
        for row in records
        if int(row["image_id"]) in observed_ids
        and int(row["image_id"]) not in selected_ids
    ]
    fill = sequence_balanced_diverse_views(
        remainder,
        limit=limit - len(selected),
        minimum_center_distance=minimum_center_distance,
        initial_selected=selected,
    )
    return [*selected, *fill]
