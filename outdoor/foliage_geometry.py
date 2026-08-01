"""Independent SfM-track and semantic visual-hull foliage geometry."""

from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from numba import njit
except ImportError:  # pragma: no cover - production environment has numba
    njit = None

from matcha.cambridge_masks import CambridgeMaskLookup
from outdoor.foliage_view_graph import (
    camera_center,
    greedy_diverse_views,
    sequence_id,
)


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
}


def evidence_conditioned_leaf_optical_mass(
    occupancy_probability,
    support_view_count,
    unknown_view_count,
    ray_depth_nll,
    free_space_violation_count,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert local observation evidence into bounded leaf opacity priors.

    Visual-hull occupancy is an existence probability, not the alpha target
    of every overlapping Gaussian. Conversely, initializing every exact
    observation-space birth near alpha=0.10 leaves a basis representing
    roughly two hundred source pixels almost transparent: on a real StMarys
    checkpoint, three owner-camera visits moved its median only to 0.12 while
    an opacity-only counterfactual peaked around 0.35--0.40. Calibrate the
    initialization to that independently measured optical coverage, while
    retaining the much lower geometry-aware training floor so RGB is free to
    retire a false or over-opaque hypothesis.

    Both outputs are spatial/per-candidate values. No historical model or RGB
    fit is consumed, and free-space/depth contradictions reduce them
    continuously.
    """
    occupancy = torch.as_tensor(occupancy_probability)
    if not occupancy.is_floating_point():
        occupancy = occupancy.float()
    device, dtype = occupancy.device, occupancy.dtype

    def value(x):
        return torch.as_tensor(x, device=device, dtype=dtype)

    support = value(support_view_count).clamp_min(0)
    unknown = value(unknown_view_count).clamp_min(0)
    depth_nll = value(ray_depth_nll).clamp_min(0)
    free = value(free_space_violation_count).clamp_min(0)
    known_fraction = support / (support + unknown).clamp_min(1)
    # One exact calibrated observation already owns its source projection;
    # repeated observations increase confidence without multiplying alpha.
    support_strength = 0.75 + 0.25 * (
        1.0 - torch.exp(-support / 2.0)
    )
    # Optical existence and metric placement are different posteriors.  A
    # tree-labelled owner pixel proves that some foreground optical mass is
    # present along its ray even when a monocular depth fit is broad.  Let
    # support/unknown/free-space evidence initialize that optical mass, while
    # occupancy and depth NLL continue to govern the geometry-training floor.
    optical_existence = (
        known_fraction
        * support_strength
        * torch.exp(-0.5 * free)
    ).clamp(0.0, 1.0)
    geometry_reliability = (
        occupancy.clamp(0.0, 1.0)
        * known_fraction
        * support_strength
        * torch.rsqrt(1.0 + depth_nll)
        * torch.exp(-0.5 * free)
    ).clamp(0.0, 1.0)
    initial_opacity = (
        0.08 + 0.32 * optical_existence
    ).clamp(0.08, 0.40)
    training_floor = 0.020 + 0.040 * geometry_reliability
    return initial_opacity, training_floor


def evidence_conditioned_dynamic_opacity_ceiling(
    occupancy_probability,
    support_view_count,
    unknown_view_count,
    ray_depth_nll,
    free_space_violation_count,
    replacement_group,
    *,
    maximum_opacity=0.40,
) -> torch.Tensor:
    """Bound dynamic alpha by spatial geometry and ownership authority.

    An exact owner-view RGB residual can validate appearance along a ray, but
    it cannot turn a high-variance single-view depth hypothesis into an
    opaque 3D occluder.  ``ray_depth_nll`` stores that per-candidate
    uncertainty.  A physically local canonical replacement group supplies an
    optical-mass denominator; an ownerless row additionally has to pay its
    occupancy probability because otherwise it is an unconserved additive
    layer.

    This is a continuous per-primitive ceiling, not an accept/reject gate.
    Reliable or locally confirmed rows retain the configured global maximum,
    while weak ownerless hypotheses remain near their evidence-conditioned
    survival floor.  Split children inherit every input, so the bound is
    invariant to topology refinement.
    """
    occupancy = torch.as_tensor(occupancy_probability)
    if not occupancy.is_floating_point():
        occupancy = occupancy.float()
    device, dtype = occupancy.device, occupancy.dtype

    def value(x):
        return torch.as_tensor(x, device=device, dtype=dtype)

    _, floor = evidence_conditioned_leaf_optical_mass(
        occupancy,
        support_view_count,
        unknown_view_count,
        ray_depth_nll,
        free_space_violation_count,
    )
    depth_reliability = torch.rsqrt(
        1.0 + value(ray_depth_nll).clamp_min(0.0)
    ).clamp(0.0, 1.0)
    groups = torch.as_tensor(
        replacement_group, device=device
    ).reshape(-1)
    if groups.shape != occupancy.shape:
        raise ValueError(
            "replacement_group and occupancy_probability must match"
        )
    local_authority = torch.where(
        groups >= 0,
        torch.ones_like(occupancy),
        occupancy.clamp(0.0, 1.0),
    )
    authority = (depth_reliability * local_authority).clamp(0.0, 1.0)
    maximum = value(maximum_opacity).clamp(1.0e-6, 1.0 - 1.0e-6)
    ceiling = floor + authority * (maximum - floor)
    return torch.maximum(floor, ceiling).clamp(
        1.0e-6, 1.0 - 1.0e-6
    )


def _bounded_extent_union_roots_python(
    xyz: np.ndarray,
    pairs: np.ndarray,
    maximum_component_extent: float,
) -> np.ndarray:
    """Union sorted local edges while enforcing a physical bbox extent."""
    parent = np.arange(len(xyz), dtype=np.int32)
    component_min = xyz.copy()
    component_max = xyz.copy()
    for pair_index in range(len(pairs)):
        first = int(pairs[pair_index, 0])
        second = int(pairs[pair_index, 1])
        while parent[first] != first:
            parent[first] = parent[parent[first]]
            first = int(parent[first])
        while parent[second] != second:
            parent[second] = parent[parent[second]]
            second = int(parent[second])
        if first == second:
            continue
        merged_min = np.minimum(
            component_min[first], component_min[second]
        )
        merged_max = np.maximum(
            component_max[first], component_max[second]
        )
        if (
            maximum_component_extent >= 0
            and np.sqrt(
                np.sum((merged_max - merged_min) ** 2)
            )
            > maximum_component_extent
        ):
            continue
        parent[second] = first
        component_min[first] = merged_min
        component_max[first] = merged_max
    roots = np.empty(len(xyz), dtype=np.int32)
    for row in range(len(xyz)):
        current = row
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        roots[row] = current
    return roots


_bounded_extent_union_roots = (
    njit(cache=True)(_bounded_extent_union_roots_python)
    if njit is not None
    else _bounded_extent_union_roots_python
)


def _read(handle, count, pattern):
    return struct.unpack("<" + pattern, handle.read(count))


def quaternion_to_rotation(qvec):
    q = np.asarray(qvec, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_cameras_binary(path):
    cameras = {}
    with Path(path).open("rb") as handle:
        count = _read(handle, 8, "Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = _read(handle, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise RuntimeError(f"Unsupported camera model id {model_id}")
            model, parameter_count = CAMERA_MODELS[model_id]
            params = np.asarray(_read(handle, 8 * parameter_count, "d" * parameter_count))
            cameras[camera_id] = {
                "id": camera_id,
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_binary(path):
    images = {}
    with Path(path).open("rb") as handle:
        count = _read(handle, 8, "Q")[0]
        for _ in range(count):
            values = _read(handle, 64, "idddddddi")
            image_id = values[0]
            name_bytes = bytearray()
            while True:
                value = handle.read(1)
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            point_count = _read(handle, 8, "Q")[0]
            observations = _read(handle, 24 * point_count, "ddq" * point_count)
            images[image_id] = {
                "id": image_id,
                "qvec": np.asarray(values[1:5]),
                "tvec": np.asarray(values[5:8]),
                "camera_id": values[8],
                "name": name_bytes.decode("utf-8"),
                "xys": np.column_stack(
                    [
                        np.asarray(observations[0::3], dtype=np.float64),
                        np.asarray(observations[1::3], dtype=np.float64),
                    ]
                ),
                "point3D_ids": np.asarray(observations[2::3], dtype=np.int64),
            }
    return images


def read_points3d_binary_with_tracks(path):
    points = []
    with Path(path).open("rb") as handle:
        count = _read(handle, 8, "Q")[0]
        for _ in range(count):
            values = _read(handle, 43, "QdddBBBd")
            track_length = _read(handle, 8, "Q")[0]
            track = _read(handle, 8 * track_length, "ii" * track_length)
            points.append(
                {
                    "id": int(values[0]),
                    "xyz": np.asarray(values[1:4], dtype=np.float64),
                    "rgb": np.asarray(values[4:7], dtype=np.uint8),
                    "error": float(values[7]),
                    "image_ids": np.asarray(track[0::2], dtype=np.int32),
                    "point2d_indices": np.asarray(track[1::2], dtype=np.int32),
                }
            )
    return points


def _tree_sample(mask_lookup, image, camera, xy):
    key = mask_lookup.source_name_for(image["name"])
    keep = mask_lookup.masks[key][3]
    height, width = keep.shape
    column = int(np.clip(xy[0] / camera["width"] * width, 0, width - 1))
    row = int(np.clip(xy[1] / camera["height"] * height, 0, height - 1))
    return not bool(keep[row, column])


def _calibrated_point_projection(point, image, camera):
    camera_xyz = (
        quaternion_to_rotation(image["qvec"]) @ np.asarray(point, dtype=np.float64)
        + image["tvec"]
    )
    if camera_xyz[2] <= 0.05:
        return None
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = float(camera["params"][0])
        cx, cy = map(float, camera["params"][1:3])
    else:
        fx, fy, cx, cy = map(float, camera["params"][:4])
    return np.asarray(
        [
            camera_xyz[0] / camera_xyz[2] * fx + cx,
            camera_xyz[1] / camera_xyz[2] * fy + cy,
        ]
    )


def semantic_tree_tracks(
    points,
    images,
    cameras,
    mask_lookup,
    *,
    minimum_tree_observations=3,
    minimum_sequences=2,
    minimum_tree_fraction=0.7,
    maximum_reprojection_error=2.0,
):
    if not any(len(image["xys"]) for image in images.values()):
        return _semantic_tree_tracks_from_calibrated_reprojection(
            points,
            images,
            cameras,
            mask_lookup,
            minimum_tree_observations=minimum_tree_observations,
            minimum_sequences=minimum_sequences,
            minimum_tree_fraction=minimum_tree_fraction,
            maximum_reprojection_error=maximum_reprojection_error,
        )
    accepted = []
    for point in points:
        if point["error"] > maximum_reprojection_error:
            continue
        tree_observations = []
        non_tree = 0
        sequences = set()
        for image_id, point2d_index in zip(
            point["image_ids"], point["point2d_indices"]
        ):
            image = images.get(int(image_id))
            if image is None:
                continue
            if 0 <= point2d_index < len(image["xys"]):
                xy = image["xys"][point2d_index]
            else:
                # Cambridge's provided model preserves point-to-image track
                # IDs but intentionally stores zero 2D observations in
                # images.bin.  Reproject the same tracked 3D point through its
                # calibrated camera; never infer a correspondence from RGB.
                xy = _calibrated_point_projection(
                    point["xyz"], image, cameras[image["camera_id"]]
                )
                if xy is None:
                    continue
            tree = _tree_sample(
                mask_lookup,
                image,
                cameras[image["camera_id"]],
                xy,
            )
            if tree:
                tree_observations.append(int(image_id))
                sequences.add(sequence_id(image["name"]))
            else:
                non_tree += 1
        total = len(tree_observations) + non_tree
        if (
            len(tree_observations) >= minimum_tree_observations
            and len(sequences) >= minimum_sequences
            and total > 0
            and len(tree_observations) / total >= minimum_tree_fraction
        ):
            accepted.append(
                {
                    **point,
                    "tree_image_ids": np.asarray(tree_observations, dtype=np.int32),
                    "tree_sequence_count": len(sequences),
                    "tree_fraction": len(tree_observations) / total,
                }
            )
    return accepted


def _semantic_tree_tracks_from_calibrated_reprojection(
    points,
    images,
    cameras,
    mask_lookup,
    *,
    minimum_tree_observations,
    minimum_sequences,
    minimum_tree_fraction,
    maximum_reprojection_error,
):
    """Vectorized fallback for Cambridge models with zero point2D rows."""
    candidate = np.asarray(
        [point["error"] <= maximum_reprojection_error for point in points],
        dtype=bool,
    )
    tracks_by_image = defaultdict(list)
    for point_index in np.nonzero(candidate)[0]:
        for image_id in points[point_index]["image_ids"]:
            if int(image_id) in images:
                tracks_by_image[int(image_id)].append(int(point_index))

    sequence_names = sorted(
        {sequence_id(images[image_id]["name"]) for image_id in tracks_by_image}
    )
    if len(sequence_names) > 63:
        raise RuntimeError("Sequence bitset supports at most 63 sequences")
    sequence_bit = {name: 1 << index for index, name in enumerate(sequence_names)}
    tree_count = np.zeros(len(points), dtype=np.int16)
    total_count = np.zeros(len(points), dtype=np.int16)
    tree_sequence_bits = np.zeros(len(points), dtype=np.uint64)
    tree_image_ids = defaultdict(list)

    for image_id, point_indices_list in tracks_by_image.items():
        image = images[image_id]
        camera = cameras[image["camera_id"]]
        point_indices = np.asarray(point_indices_list, dtype=np.int64)
        xyz = np.stack([points[index]["xyz"] for index in point_indices])
        rotation = quaternion_to_rotation(image["qvec"])
        camera_xyz = xyz @ rotation.T + image["tvec"][None]
        depth = camera_xyz[:, 2]
        if camera["model"] == "SIMPLE_PINHOLE":
            fx = fy = float(camera["params"][0])
            cx, cy = map(float, camera["params"][1:3])
        else:
            fx, fy, cx, cy = map(float, camera["params"][:4])
        u = camera_xyz[:, 0] / np.maximum(depth, 1e-8) * fx + cx
        v = camera_xyz[:, 1] / np.maximum(depth, 1e-8) * fy + cy
        valid = (
            (depth > 0.05)
            & (u >= 0)
            & (u < camera["width"])
            & (v >= 0)
            & (v < camera["height"])
        )
        mask_key = mask_lookup.source_name_for(image["name"])
        keep = mask_lookup.masks[mask_key][3].numpy().astype(bool, copy=False)
        height, width = keep.shape
        columns = np.clip(
            np.floor(u / camera["width"] * width).astype(np.int64),
            0,
            width - 1,
        )
        rows = np.clip(
            np.floor(v / camera["height"] * height).astype(np.int64),
            0,
            height - 1,
        )
        tree = valid & (~keep[rows, columns])
        np.add.at(total_count, point_indices[valid], 1)
        np.add.at(tree_count, point_indices[tree], 1)
        tree_indices = point_indices[tree]
        tree_sequence_bits[tree_indices] |= np.uint64(
            sequence_bit[sequence_id(image["name"])]
        )
        for point_index in tree_indices:
            tree_image_ids[int(point_index)].append(int(image_id))

    sequence_count = np.fromiter(
        (bin(int(value)).count("1") for value in tree_sequence_bits),
        dtype=np.int16,
        count=len(points),
    )
    fraction = tree_count / np.maximum(total_count, 1)
    keep = (
        candidate
        & (tree_count >= minimum_tree_observations)
        & (sequence_count >= minimum_sequences)
        & (fraction >= minimum_tree_fraction)
    )
    accepted = []
    for point_index in np.nonzero(keep)[0]:
        accepted.append(
            {
                **points[point_index],
                "tree_image_ids": np.asarray(
                    tree_image_ids[int(point_index)], dtype=np.int32
                ),
                "tree_sequence_count": int(sequence_count[point_index]),
                "tree_fraction": float(fraction[point_index]),
            }
        )
    return accepted


def _camera_record(image, camera, mask_lookup):
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = float(camera["params"][0])
        cx, cy = map(float, camera["params"][1:3])
    else:
        fx, fy, cx, cy = map(float, camera["params"][:4])
    key = mask_lookup.source_name_for(image["name"])
    keep = mask_lookup.masks[key][3]
    return {
        "image_id": image["id"],
        "image_name": image["name"],
        "sequence_id": sequence_id(image["name"]),
        "camera_center": camera_center(
            image["qvec"], image["tvec"], quaternion_to_rotation
        ),
        "canopy_fraction": float((~keep.bool()).float().mean().item()),
        "rotation": quaternion_to_rotation(image["qvec"]),
        "translation": image["tvec"],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": camera["width"],
        "height": camera["height"],
        "tree_keep_mask": keep.numpy().astype(bool, copy=False),
    }


def _candidate_voxels(anchor_xyz, voxel_size, dilation_steps, maximum_voxels, seed):
    anchor_cells = np.unique(np.round(anchor_xyz / voxel_size).astype(np.int32), axis=0)
    axis = np.arange(-dilation_steps, dilation_steps + 1, dtype=np.int32)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    candidates = np.unique(
        (anchor_cells[:, None, :] + offsets[None, :, :]).reshape(-1, 3), axis=0
    )
    if len(candidates) > maximum_voxels:
        rng = np.random.default_rng(seed)
        candidates = candidates[
            np.sort(rng.choice(len(candidates), maximum_voxels, replace=False))
        ]
    return candidates, candidates.astype(np.float64) * voxel_size


def _priority_candidate_voxels(
    primary_anchor_xyz,
    supplemental_anchor_xyz,
    *,
    voxel_size,
    primary_dilation_steps,
    supplemental_dilation_steps,
    maximum_voxels,
    seed,
):
    """Grow proposals without letting dense mono depth evict SfM cells."""
    primary_cells, _ = _candidate_voxels(
        np.asarray(primary_anchor_xyz),
        voxel_size,
        primary_dilation_steps,
        maximum_voxels,
        seed,
    )
    supplemental_anchor_xyz = np.asarray(supplemental_anchor_xyz)
    if (
        supplemental_anchor_xyz.size == 0
        or len(primary_cells) >= int(maximum_voxels)
    ):
        return primary_cells, primary_cells.astype(np.float64) * voxel_size

    # DAV2 is a hole-filling proposal source. Build its cells independently so
    # its higher sampling density cannot change which MASt3R/Chart cells
    # survive the global budget.
    supplemental_cells, _ = _candidate_voxels(
        supplemental_anchor_xyz.reshape(-1, 3),
        voxel_size,
        supplemental_dilation_steps,
        np.iinfo(np.int32).max,
        seed + 1,
    )
    primary_keys = {tuple(row) for row in primary_cells.tolist()}
    supplemental_cells = np.asarray(
        [
            row
            for row in supplemental_cells.tolist()
            if tuple(row) not in primary_keys
        ],
        dtype=np.int32,
    ).reshape(-1, 3)
    remaining = int(maximum_voxels) - len(primary_cells)
    if len(supplemental_cells) > remaining:
        rng = np.random.default_rng(seed + 1)
        supplemental_cells = supplemental_cells[
            np.sort(
                rng.choice(
                    len(supplemental_cells), remaining, replace=False
                )
            )
        ]
    candidates = np.concatenate(
        [primary_cells, supplemental_cells], axis=0
    )
    return candidates, candidates.astype(np.float64) * voxel_size


def _confirmed_free_space_mask(
    valid: np.ndarray,
    occluded: np.ndarray,
    in_front_of_depth_posterior: np.ndarray,
) -> np.ndarray:
    """Return only observable free-space contradictions.

    Being in front of a local depth posterior is not sufficient on its own:
    an off-image or rigidly occluded projection is unknown evidence.  Keep
    this predicate shared by the ray labels and the persistent per-candidate
    contradiction counter so topology pruning cannot silently reinterpret
    unknown rays as confirmed free space.
    """
    return (
        np.asarray(valid, dtype=bool)
        & ~np.asarray(occluded, dtype=bool)
        & np.asarray(in_front_of_depth_posterior, dtype=bool)
    )


def _accumulate_supported_depth_nll(
    total: np.ndarray,
    residual: np.ndarray,
    sigma: np.ndarray,
    supported: np.ndarray,
) -> np.ndarray:
    """Accumulate depth likelihood only for measured hit intervals.

    Unknown/occluded rays are deliberately absent from this update.  Their
    residual to a first-hit posterior is not evidence against a deeper crown
    candidate and must not weaken its geometry covariance or split priority.
    The returned indices make the evidence classification directly auditable
    in unit tests and callers.
    """
    indices = np.flatnonzero(np.asarray(supported, dtype=bool))
    if len(indices):
        total[indices] += (
            np.asarray(residual, dtype=np.float64)[indices]
            / np.maximum(
                np.asarray(sigma, dtype=np.float64)[indices], 1e-6
            )
        ) ** 2
    return indices


def _supported_view_geometry_statistics(
    centers: np.ndarray,
    support_matrix: np.ndarray,
    camera_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact baseline/angle maxima from sparse positive support.

    Retained canopy candidates normally have only a handful of positive
    views. Scanning all candidates for all selected-camera pairs made the
    256-view initializer perform more than 13 billion boolean tests. Iterating
    each candidate's actual support list computes the identical pairwise
    maxima in O(sum(k_i²)), where k_i is its positive support count.
    """
    centers = np.asarray(centers, dtype=np.float64)
    support_matrix = np.asarray(support_matrix, dtype=bool)
    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    if support_matrix.ndim != 2:
        raise ValueError("support_matrix must have shape [candidate, view]")
    if centers.shape != (len(support_matrix), 3):
        raise ValueError("centers must have shape [candidate, 3]")
    if camera_centers.shape != (support_matrix.shape[1], 3):
        raise ValueError("camera_centers must have shape [view, 3]")

    maximum_baseline = np.zeros(len(centers), dtype=np.float32)
    maximum_angle = np.zeros(len(centers), dtype=np.float32)
    supported_rows = np.flatnonzero(support_matrix.sum(axis=1) >= 2)
    for row in supported_rows:
        view_indices = np.flatnonzero(support_matrix[row])
        cameras = camera_centers[view_indices]
        camera_delta = cameras[:, None, :] - cameras[None, :, :]
        maximum_baseline[row] = np.sqrt(
            np.sum(camera_delta * camera_delta, axis=-1).max()
        )
        rays = centers[row][None] - cameras
        rays /= np.maximum(
            np.linalg.norm(rays, axis=1, keepdims=True), 1e-8
        )
        minimum_cosine = np.clip((rays @ rays.T).min(), -1.0, 1.0)
        maximum_angle[row] = np.degrees(np.arccos(minimum_cosine))
    return maximum_baseline, maximum_angle


def cluster_tree_instances(
    xyz,
    *,
    connection_radius=1.5,
    minimum_tracks=8,
    maximum_component_extent=12.0,
    maximum_neighbours=24,
    minimum_core_neighbours=4,
):
    """Cluster tree tracks without sparse single-link bridges.

    A radius-only graph is vulnerable to chaining: a few erroneous or
    touching-canopy samples can join distinct crowns into one instance even
    when the resulting extent remains below the generous physical-crown cap.
    Establish connectivity through density-supported core observations;
    border observations may attach to one nearby core but may not themselves
    bridge two cores. A fully sparse component falls back to the bounded
    radius graph so weak evidence is retained rather than deleted.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)

    if len(xyz) == 0:
        return np.empty(0, dtype=np.int32)
    if int(maximum_neighbours) <= 0:
        raise ValueError("maximum_neighbours must be positive")
    if int(minimum_core_neighbours) <= 0:
        raise ValueError("minimum_core_neighbours must be positive")
    # ``query_pairs`` materializes every radius-graph edge.  A 12 cm visual
    # hull queried with a 1.5 m ownership radius can have thousands of edges
    # per voxel, even though only a sparse local graph is needed to establish
    # connectivity.  Keep the nearest bounded neighbourhood instead.  This
    # also prevents a dense crown from overwhelming the ownership builder
    # before any optimization starts.
    neighbour_count = min(len(xyz), int(maximum_neighbours) + 1)
    _, neighbours = cKDTree(xyz).query(
        xyz,
        k=neighbour_count,
        distance_upper_bound=float(connection_radius),
        workers=-1,
    )
    neighbours = np.asarray(neighbours)
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]
    rows = np.broadcast_to(
        np.arange(len(xyz), dtype=np.int64)[:, None],
        neighbours.shape,
    )
    valid = (neighbours < len(xyz)) & (neighbours != rows)
    all_pairs = np.column_stack(
        [rows[valid], neighbours[valid].astype(np.int64)]
    )
    if len(all_pairs):
        all_pairs.sort(axis=1)
        all_pairs = np.unique(all_pairs, axis=0)
    degree = valid.sum(axis=1)
    core = degree >= int(minimum_core_neighbours)
    if bool(core.any()):
        # Core-to-core edges establish components. Each border row gets at
        # most one attachment, so a sparse run of noisy samples cannot perform
        # transitive single-link merging between two well-supported crowns.
        core_pairs = all_pairs[
            core[all_pairs[:, 0]] & core[all_pairs[:, 1]]
        ]
        border_pairs = []
        distances, nearest = cKDTree(xyz[core]).query(
            xyz[~core],
            k=1,
            distance_upper_bound=float(connection_radius),
            workers=-1,
        )
        core_indices = np.flatnonzero(core)
        border_indices = np.flatnonzero(~core)
        attached = np.isfinite(distances) & (nearest < len(core_indices))
        if bool(attached.any()):
            border_pairs = np.column_stack(
                [
                    border_indices[attached],
                    core_indices[nearest[attached]],
                ]
            )
        pairs = (
            np.concatenate([core_pairs, border_pairs], axis=0)
            if len(border_pairs)
            else core_pairs
        )
    else:
        pairs = all_pairs
    if len(pairs):
        distance = np.linalg.norm(
            xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1
        )
        pairs = pairs[np.argsort(distance, kind="stable")]
    roots = _bounded_extent_union_roots(
        xyz,
        pairs,
        (
            -1.0
            if maximum_component_extent is None
            else float(maximum_component_extent)
        ),
    )
    unique, counts = np.unique(roots, return_counts=True)
    # Keep established components first for stable ids, then retain each
    # small connected component as one uncertain instance. The old fallback
    # assigned every point in such a component a different id.
    ordered = np.concatenate(
        [
            unique[counts >= int(minimum_tracks)],
            unique[counts < int(minimum_tracks)],
        ]
    )
    mapping = {
        int(value): offset for offset, value in enumerate(ordered)
    }
    labels = np.asarray(
        [mapping[int(value)] for value in roots], dtype=np.int32
    )
    return labels


def _component_instance_map(
    view,
    anchor_xyz,
    anchor_instances,
    observed_track_mask,
):
    """Assign each tree pixel to its nearest observed local instance.

    A connected semantic mask is not a tree-instance segmentation. Touching
    crowns (and a foreground branch crossing a background crown) routinely
    form one component containing several spatial 3D instances. The old
    component-wide majority vote rejected the *entire* component whenever no
    instance exceeded 75%, which erased all dense rays in 00408 despite 926
    calibrated observations landing inside its tree mask.

    Keep connected components only as a barrier against crossing non-tree
    pixels. Within each component, propagate the nearest projected measured
    instance per pixel. The later depth-neighbour radius still decides
    whether that assignment has enough local evidence to produce a posterior.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    tree = ~view["tree_keep_mask"]
    components, component_count = ndimage.label(tree)
    raster_instance = np.full(components.shape, -1, dtype=np.int32)
    if not np.any(observed_track_mask):
        return components, raster_instance
    u, v, _, valid = _project(anchor_xyz[observed_track_mask], view)
    instances = anchor_instances[observed_track_mask]
    rows = np.clip(
        np.round(v / view["height"] * tree.shape[0]).astype(int),
        0,
        tree.shape[0] - 1,
    )
    cols = np.clip(
        np.round(u / view["width"] * tree.shape[1]).astype(int),
        0,
        tree.shape[1] - 1,
    )
    labels = components[rows, cols]
    for label in np.unique(labels[valid & (labels > 0)]):
        track_choice = valid & (labels == label)
        pixel_rows, pixel_columns = np.nonzero(components == label)
        if not len(pixel_rows):
            continue
        pixel_u = (
            (pixel_columns.astype(np.float64) + 0.5)
            * float(view["width"])
            / float(components.shape[1])
        )
        pixel_v = (
            (pixel_rows.astype(np.float64) + 0.5)
            * float(view["height"])
            / float(components.shape[0])
        )
        nearest = cKDTree(
            np.column_stack([u[track_choice], v[track_choice]])
        ).query(np.column_stack([pixel_u, pixel_v]), k=1)[1]
        raster_instance[pixel_rows, pixel_columns] = instances[
            track_choice
        ][nearest]
    return components, raster_instance


def _allocate_dense_ray_budgets(
    selected_views,
    instance_rasters,
    *,
    maximum_per_view,
    total_budget=None,
    minimum_per_view=0,
):
    """Allocate one fixed global ray budget without dropping camera coverage.

    A fixed per-camera cap couples two different quantities: the number of
    calibrated cameras covered by the posterior and the source-image
    bandwidth retained in each camera.  Reducing the view count therefore
    sharpened exact keyframes while making interpolation views worse.  Keep
    all selected cameras and distribute an optional global budget according
    to the square root of their measured tree-pixel support.  The square root
    prevents one foreground crown from consuming the entire budget, while a
    per-view floor preserves the view-selection coverage contract.
    """
    maximum_per_view = int(maximum_per_view)
    minimum_per_view = int(minimum_per_view)
    if maximum_per_view < 0 or minimum_per_view < 0:
        raise ValueError("Dense ray budgets must be non-negative")
    if minimum_per_view > maximum_per_view:
        raise ValueError(
            "minimum_per_view cannot exceed maximum_per_view"
        )
    image_ids = np.asarray(
        [int(view["image_id"]) for view in selected_views],
        dtype=np.int64,
    )
    eligible = np.asarray(
        [
            int(
                np.count_nonzero(
                    np.asarray(instance_rasters[int(image_id)]) >= 0
                )
            )
            for image_id in image_ids
        ],
        dtype=np.int64,
    )
    capacity = np.minimum(eligible, maximum_per_view)
    if total_budget is None:
        allocated = capacity.copy()
        requested_total = int(capacity.sum())
        policy = "independent_per_view_cap"
    else:
        requested_total = int(total_budget)
        if requested_total < 0:
            raise ValueError("total_budget must be non-negative")
        requested_total = min(requested_total, int(capacity.sum()))
        floor = np.minimum(capacity, minimum_per_view)
        if int(floor.sum()) > requested_total:
            raise ValueError(
                "Global dense ray budget cannot satisfy the per-view floor: "
                f"required={int(floor.sum())}, total={requested_total}"
            )
        allocated = floor.copy()
        remaining = requested_total - int(allocated.sum())
        weights = np.sqrt(eligible.astype(np.float64))
        # Bounded proportional water filling.  At least one integer is
        # assigned in every non-terminal pass, so this is deterministic and
        # terminates in at most O(number_of_views) saturation rounds.
        while remaining > 0:
            available = capacity - allocated
            active = available > 0
            if not np.any(active):
                break
            active_weights = weights * active
            weight_sum = float(active_weights.sum())
            if weight_sum <= 0:
                order = np.flatnonzero(active)
                add = min(remaining, len(order))
                allocated[order[:add]] += 1
                remaining -= add
                continue
            ideal = remaining * active_weights / weight_sum
            increment = np.minimum(
                available,
                np.floor(ideal).astype(np.int64),
            )
            assigned = int(increment.sum())
            if assigned:
                allocated += increment
                remaining -= assigned
                continue
            # Largest-remainder tie-breaking is stable in selected-view
            # order, which is itself content/provenance deterministic.
            order = np.lexsort(
                (
                    np.arange(len(image_ids), dtype=np.int64),
                    -eligible,
                    -(ideal - np.floor(ideal)),
                )
            )
            order = order[active[order]]
            add = min(remaining, len(order))
            allocated[order[:add]] += 1
            remaining -= add
        policy = (
            "global_budget_sqrt_tree_area_with_per_view_floor_and_cap"
        )
    budgets = {
        int(image_id): int(value)
        for image_id, value in zip(image_ids, allocated)
    }
    nonzero = allocated[allocated > 0]
    audit = {
        "policy": policy,
        "selected_view_count": int(len(image_ids)),
        "requested_total": int(requested_total),
        "allocated_total": int(allocated.sum()),
        "maximum_per_view": int(maximum_per_view),
        "minimum_per_view": int(minimum_per_view),
        "eligible_pixel_total": int(eligible.sum()),
        "zero_support_view_count": int(np.count_nonzero(eligible == 0)),
        "allocated_minimum": int(nonzero.min()) if len(nonzero) else 0,
        "allocated_median": (
            float(np.median(nonzero)) if len(nonzero) else 0.0
        ),
        "allocated_p90": (
            float(np.quantile(nonzero, 0.90)) if len(nonzero) else 0.0
        ),
        "allocated_maximum": int(nonzero.max()) if len(nonzero) else 0,
    }
    return budgets, audit


def _invert_track_image_incidence(
    track_images,
    selected_image_ids,
):
    """Index observation rows once instead of scanning tracks per camera.

    The all-fixed Cambridge posterior has roughly 1,500 cameras and hundreds
    of thousands of observation rows.  Rebuilding
    ``[image_id in values for values in track_images]`` for every camera (and
    doing it once for the component raster and again for the ray posterior)
    performs more than a billion Python membership checks.  The incidence
    graph is immutable, so invert it in one pass and construct each temporary
    boolean mask with vectorized indexing.
    """
    selected = {int(value) for value in selected_image_ids}
    rows_by_image = {image_id: [] for image_id in selected}
    for row, image_ids in enumerate(track_images):
        for image_id in image_ids:
            image_id = int(image_id)
            if image_id in rows_by_image:
                rows_by_image[image_id].append(int(row))
    return {
        image_id: np.asarray(rows, dtype=np.int64)
        for image_id, rows in rows_by_image.items()
    }


def _incidence_mask(rows, row_count):
    mask = np.zeros(int(row_count), dtype=bool)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if len(rows):
        mask[rows] = True
    return mask


def _dynamic_birth_limit_for_view(
    tracks,
    observed_track_mask,
    *,
    ordinary_limit,
    weak_continuous_limit,
    eligible_pixel_count=0,
    target_pixels_per_birth=None,
):
    """Allocate local optical bandwidth from measured pixels and uncertainty.

    The ordinary limit is a coverage floor.  Large tree silhouettes need
    proportionally more finite-footprint basis rows even when their DAV2
    metric fit is accepted; broad weak-continuous fits additionally use the
    configured ceiling because smaller local splats are safer than enlarging
    one uncertain 3D primitive.  Neither condition changes geometry
    confidence or ray-factor strength.
    """
    observed = np.asarray(observed_track_mask, dtype=bool).reshape(-1)
    if len(observed) != len(tracks):
        raise ValueError("Observed-track mask must align with tracks")
    ordinary_limit = max(int(ordinary_limit), 0)
    weak_limit = (
        ordinary_limit
        if weak_continuous_limit is None
        else max(int(weak_continuous_limit), 0)
    )
    eligible_pixel_count = max(int(eligible_pixel_count), 0)
    if target_pixels_per_birth is None:
        area_limit = ordinary_limit
    else:
        target_pixels_per_birth = float(target_pixels_per_birth)
        if (
            not np.isfinite(target_pixels_per_birth)
            or target_pixels_per_birth <= 0
        ):
            raise ValueError(
                "target_pixels_per_birth must be finite and positive"
            )
        area_limit = int(
            np.ceil(eligible_pixel_count / target_pixels_per_birth)
        )
        area_limit = max(ordinary_limit, area_limit)
        # The weak-continuous cap is also the explicit maximum adaptive
        # renderer-basis budget.  This keeps silhouette allocation bounded by
        # the already persisted per-view ray budget.
        area_limit = min(area_limit, max(ordinary_limit, weak_limit))
    weak = any(
        str(tracks[row].get("dav2_alignment_status", ""))
        == "weak_continuous"
        for row in np.flatnonzero(observed)
    )
    limit = max(area_limit, weak_limit) if weak else area_limit
    return limit, weak, area_limit


def _spatially_stratified_coverage_rows(
    rows: np.ndarray,
    sample_u: np.ndarray,
    sample_v: np.ndarray,
    count: int,
) -> np.ndarray:
    """Choose deterministic 2D coverage witnesses from raster-ordered rows.

    Dense ray rows are stored in raster order. Applying ``linspace`` to their
    one-dimensional indices therefore traces scanlines/diagonals rather than
    covering the two-dimensional crown, and the resulting exact-RGB splats
    remain visible as coherent bands after training. This routine places a
    lattice over the observed image-space extent and assigns one distinct
    nearest row to each lattice site. The result is invariant to input row
    order and preserves the requested finite renderer-basis size exactly.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    count = min(max(int(count), 0), len(rows))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == len(rows):
        return np.sort(rows)
    rows = np.sort(rows)
    u = np.asarray(sample_u, dtype=np.float64)[rows]
    v = np.asarray(sample_v, dtype=np.float64)[rows]
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    u_span = max(u_max - u_min, 1.0)
    v_span = max(v_max - v_min, 1.0)
    aspect = np.clip(u_span / v_span, 1.0 / count, float(count))
    columns = max(1, int(np.ceil(np.sqrt(count * aspect))))
    row_count = max(1, int(np.ceil(count / columns)))
    normalized = np.column_stack(
        [(u - u_min) / u_span, (v - v_min) / v_span]
    )
    # A dense posterior may contain hundreds of thousands of valid pixels in
    # one camera.  Per-target nearest-neighbour scans are quadratic, while a
    # KD-tree per camera still spends most initialization time rebuilding the
    # same regular image-space index.  Assign candidates to the target lattice
    # in one vectorized pass, select the closest row to every occupied cell,
    # then distribute any refill across the remaining cells.  This is linear
    # apart from one deterministic sort and preserves exact cardinality.
    cell_x = np.minimum(
        np.floor(normalized[:, 0] * columns).astype(np.int64),
        columns - 1,
    )
    cell_y = np.minimum(
        np.floor(normalized[:, 1] * row_count).astype(np.int64),
        row_count - 1,
    )
    cell = cell_y * columns + cell_x
    cell_center_x = (cell_x.astype(np.float64) + 0.5) / columns
    cell_center_y = (cell_y.astype(np.float64) + 0.5) / row_count
    center_distance = (
        (normalized[:, 0] - cell_center_x) ** 2
        + (normalized[:, 1] - cell_center_y) ** 2
    )
    order = np.lexsort((rows, center_distance, cell))
    ordered_cell = cell[order]
    first = np.concatenate(
        [
            np.asarray([True]),
            ordered_cell[1:] != ordered_cell[:-1],
        ]
    )
    selected_local = order[first]
    if len(selected_local) > count:
        selected_local = selected_local[
            np.linspace(
                0, len(selected_local) - 1, count, dtype=np.int64
            )
        ]
    elif len(selected_local) < count:
        remaining_local = order[~first]
        required = count - len(selected_local)
        refill = remaining_local[
            np.linspace(
                0,
                len(remaining_local) - 1,
                required,
                dtype=np.int64,
            )
        ]
        selected_local = np.concatenate([selected_local, refill])
    return np.sort(rows[selected_local])


def _bounded_candidate_ray_rows(
    ray_indices,
    pixel_u,
    pixel_v,
    observation_type,
    *,
    maximum_rows,
):
    """Keep a spatial/type-stratified finite factor basis for one camera.

    Visual-hull verification is evaluated against every selected fixed
    camera, but persisting every candidate-camera pair couples camera
    coverage to factor cardinality.  In an all-database-camera run that turns
    a bounded dense posterior into tens of millions of redundant bound rows.
    Keep positive, confirmed-free and explicit-unknown semantics represented,
    then distribute the remaining budget by square-root class support and a
    deterministic image-space grid.

    This is a persistent factor-basis selection, not the runtime evidence
    sampler.  Runtime still uses independent per-camera shuffled epochs over
    every retained row.
    """
    rows = np.asarray(ray_indices, dtype=np.int64).reshape(-1)
    maximum_rows = int(maximum_rows)
    if maximum_rows < 0:
        raise ValueError("maximum_rows must be non-negative")
    if maximum_rows == 0 or not len(rows):
        return np.empty(0, dtype=np.int64)
    if len(rows) <= maximum_rows:
        return rows.copy()
    pixel_u = np.asarray(pixel_u, dtype=np.float64).reshape(-1)
    pixel_v = np.asarray(pixel_v, dtype=np.float64).reshape(-1)
    observation_type = np.asarray(observation_type, dtype=np.int8).reshape(-1)
    if (
        len(pixel_u) != len(observation_type)
        or len(pixel_v) != len(observation_type)
    ):
        raise ValueError("Ray pixel/type arrays must have equal length")
    if int(rows.max()) >= len(observation_type) or int(rows.min()) < 0:
        raise ValueError("Ray row lies outside the pixel/type arrays")

    # Hits and confirmed free-space are physical factors. Unknown rows remain
    # explicit provenance but receive the last minimum slot if an unusually
    # tiny caller budget cannot represent all three classes.
    class_order = (1, -1, 0)
    class_rows = [
        rows[observation_type[rows] == value] for value in class_order
    ]
    present = np.asarray([len(value) > 0 for value in class_rows], dtype=bool)
    counts = np.asarray([len(value) for value in class_rows], dtype=np.int64)
    quotas = np.zeros(len(class_order), dtype=np.int64)
    first = np.flatnonzero(present)[:maximum_rows]
    quotas[first] = 1
    remaining = maximum_rows - int(quotas.sum())
    capacity = counts - quotas
    while remaining > 0 and bool((capacity > 0).any()):
        active = capacity > 0
        weights = np.sqrt(counts.astype(np.float64))
        weights[~active] = 0.0
        ideal = remaining * weights / float(weights.sum())
        addition = np.minimum(
            np.floor(ideal).astype(np.int64), capacity
        )
        if int(addition.sum()) == 0:
            fractional = ideal - np.floor(ideal)
            order = np.lexsort(
                (
                    np.arange(len(class_order), dtype=np.int64),
                    -counts,
                    -fractional,
                )
            )
            order = order[active[order]]
            addition[order[: min(remaining, len(order))]] = 1
        quotas += addition
        capacity -= addition
        remaining -= int(addition.sum())

    selected = []
    for candidates, quota in zip(class_rows, quotas):
        quota = int(quota)
        if quota <= 0:
            continue
        if len(candidates) <= quota:
            selected.append(candidates)
            continue
        u = pixel_u[candidates]
        v = pixel_v[candidates]
        grid_side = max(1, int(np.ceil(np.sqrt(quota))))
        u_span = max(float(u.max() - u.min()), 1.0)
        v_span = max(float(v.max() - v.min()), 1.0)
        cell_x = np.minimum(
            np.floor(
                (u - float(u.min())) / u_span * grid_side
            ).astype(np.int64),
            grid_side - 1,
        )
        cell_y = np.minimum(
            np.floor(
                (v - float(v.min())) / v_span * grid_side
            ).astype(np.int64),
            grid_side - 1,
        )
        cell = cell_y * grid_side + cell_x
        order = np.lexsort((candidates, u, v, cell))
        ordered_cells = cell[order]
        first_in_cell = np.concatenate(
            [
                np.asarray([True]),
                ordered_cells[1:] != ordered_cells[:-1],
            ]
        )
        winners = candidates[order[first_in_cell]]
        if len(winners) > quota:
            winners = winners[
                np.linspace(
                    0, len(winners) - 1, quota, dtype=np.int64
                )
            ]
        if len(winners) < quota:
            unselected = candidates[
                ~np.isin(candidates, winners, assume_unique=False)
            ]
            refill = unselected[
                np.linspace(
                    0,
                    len(unselected) - 1,
                    quota - len(winners),
                    dtype=np.int64,
                )
            ]
            winners = np.concatenate([winners, refill])
        selected.append(winners)
    if not selected:
        return np.empty(0, dtype=np.int64)
    result = np.unique(np.concatenate(selected))
    if len(result) != maximum_rows:
        raise RuntimeError(
            "Candidate ray factor basis did not fill its bounded budget: "
            f"expected={maximum_rows}, actual={len(result)}"
        )
    return np.sort(result)


def strict_foliage_hit_interval_bounds(
    depth,
    sigma,
    *,
    free_space_margin=0.25,
    minimum_depth=0.05,
):
    """Build disjoint free-space and hit intervals in camera-z depth.

    A hit posterior has two different statements: space *before* the lower
    confidence bound is empty, while mass inside the confidence interval is
    supported.  Using ``depth - free_space_margin`` as the free endpoint
    without intersecting it with ``depth - 2.5 * sigma`` makes those two
    statements overlap whenever the posterior uncertainty exceeds
    ``free_space_margin / 2.5``.  That contradiction used to affect every
    Cambridge hit row and pushed foliage mass into broad, rearward layers.
    """
    depth = np.asarray(depth, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    depth, sigma = np.broadcast_arrays(depth, sigma)
    if (
        np.any(~np.isfinite(depth))
        or np.any(~np.isfinite(sigma))
        or np.any(depth <= 0)
        or np.any(sigma < 0)
    ):
        raise ValueError("Foliage hit depth and sigma must be finite and valid")
    minimum_depth = max(float(minimum_depth), 0.0)
    hit_start = np.maximum(
        depth - 2.5 * sigma, minimum_depth
    )
    hit_end = np.maximum(
        depth + 2.5 * sigma, hit_start + 1.0e-4
    )
    free_end = np.maximum(
        np.minimum(
            depth - float(free_space_margin), hit_start
        ),
        0.0,
    )
    if not (
        np.all(free_end <= hit_start)
        and np.all(hit_start < hit_end)
    ):
        raise RuntimeError("Failed to construct strict foliage ray intervals")
    return (
        free_end.astype(np.float32),
        hit_start.astype(np.float32),
        hit_end.astype(np.float32),
    )


def enforce_strict_foliage_ray_intervals(
    free_end,
    hit_start,
    hit_end,
    observation_type,
    *,
    minimum_depth=0.05,
):
    """Repair a persisted ray table without changing non-hit semantics."""
    free_end = np.asarray(free_end, dtype=np.float32).copy()
    hit_start = np.asarray(hit_start, dtype=np.float32).copy()
    hit_end = np.asarray(hit_end, dtype=np.float32).copy()
    observation_type = np.asarray(observation_type).reshape(-1)
    if not (
        free_end.shape
        == hit_start.shape
        == hit_end.shape
        == observation_type.shape
    ):
        raise ValueError("Foliage ray interval arrays must have equal shape")
    hit = observation_type > 0
    original_free = free_end.copy()
    original_start = hit_start.copy()
    original_end = hit_end.copy()
    hit_start[hit] = np.maximum(
        hit_start[hit], max(float(minimum_depth), 0.0)
    )
    hit_end[hit] = np.maximum(
        hit_end[hit], hit_start[hit] + 1.0e-4
    )
    free_end[hit] = np.maximum(
        np.minimum(free_end[hit], hit_start[hit]), 0.0
    )
    finite = (
        np.isfinite(free_end[hit])
        & np.isfinite(hit_start[hit])
        & np.isfinite(hit_end[hit])
    )
    valid = (
        finite
        & (free_end[hit] <= hit_start[hit])
        & (hit_start[hit] < hit_end[hit])
    )
    if not bool(valid.all()):
        raise RuntimeError("Persisted foliage hit intervals remain invalid")
    changed = hit & (
        (free_end != original_free)
        | (hit_start != original_start)
        | (hit_end != original_end)
    )
    return free_end, hit_start, hit_end, {
        "hit_rows": int(hit.sum()),
        "changed_hit_rows": int(changed.sum()),
        "overlap_rows_before": int(
            (hit & (original_free > original_start)).sum()
        ),
        "overlap_rows_after": int(
            (hit & (free_end > hit_start)).sum()
        ),
        "negative_hit_start_rows_before": int(
            (hit & (original_start < 0)).sum()
        ),
        "strict_interval_contract": (
            "zero_le_free_end_le_hit_start_lt_hit_end"
        ),
    }


def _dense_component_ray_posterior(
    view,
    observation_xyz,
    observation_instances,
    observation_errors,
    observation_colors,
    observed_track_mask,
    *,
    observation_depth_sigmas=None,
    rigid_depth=None,
    depth_neighbor_pixels=48.0,
    depth_sigma_floor=0.20,
    occlusion_margin=0.15,
    free_space_margin=0.25,
    maximum_rays=4096,
    maximum_proposals=512,
    maximum_dynamic_births=1024,
    renderer_reference_width=640,
    instance_raster=None,
):
    """Interpolate calibrated hit intervals over observed tree components.

    Candidate-centre projections are useful for validating an existing visual
    hull cell, but they cannot describe a canopy pixel for which no cell was
    initialized.  This routine samples the *observation domain* instead:
    pixels inside a connected tree-mask component receive a local depth
    posterior only when that component has a same-instance measured track.
    The returned ray rows are independent of renderer primitive identities.
    A bounded subset of their hit points is also returned as supplemental
    (single-view) proposals; the normal multi-view hull tests still decide
    whether any proposal is allowed into the canonical model.  The complete
    valid ray table is retained as training evidence, while a spatially
    stratified basis of sequence-local dynamic births consumes it.  These are
    deliberately different cardinalities: one anisotropic Gaussian covers a
    finite image patch and can explain several nearby calibrated rays.
    Instantiating one renderer primitive per evidence row exhausts the global
    volume budget before topology measures either canonical or dynamic
    screen-bandwidth demand.
    """
    from scipy.spatial import cKDTree

    observed_track_mask = np.asarray(observed_track_mask, dtype=bool)
    if not np.any(observed_track_mask) or int(maximum_rays) <= 0:
        return None, [], []
    if instance_raster is None:
        _, instance_raster = _component_instance_map(
            view,
            observation_xyz,
            observation_instances,
            observed_track_mask,
        )
    else:
        instance_raster = np.asarray(instance_raster, dtype=np.int32)
        if instance_raster.shape != view["tree_keep_mask"].shape:
            raise ValueError(
                "Dense posterior instance raster does not match tree mask"
            )
    eligible = instance_raster >= 0
    eligible_rows, eligible_columns = np.nonzero(eligible)
    if not len(eligible_rows):
        return None, [], []
    # Do not spend the renderer-ray budget before posterior validity is
    # known.  The old ordering sampled ``maximum_rays`` silhouette pixels and
    # then discarded rows without a local depth anchor or rows occluded by
    # rigid geometry.  Difficult crowns therefore retained only about half
    # their allocated owner rows, and no later split could recreate the lost
    # camera/ray ownership.  Evaluate the observation domain first and take a
    # deterministic, spatially even subset of *valid* rows below.
    mask_height, mask_width = instance_raster.shape
    sample_u = (
        (eligible_columns.astype(np.float64) + 0.5)
        * float(view["width"])
        / float(mask_width)
    )
    sample_v = (
        (eligible_rows.astype(np.float64) + 0.5)
        * float(view["height"])
        / float(mask_height)
    )
    sample_instance = instance_raster[
        eligible_rows, eligible_columns
    ].astype(np.int32)

    observed_xyz = np.asarray(observation_xyz)[observed_track_mask]
    observed_instances = np.asarray(observation_instances)[
        observed_track_mask
    ]
    observed_errors = np.asarray(observation_errors, dtype=np.float64)[
        observed_track_mask
    ]
    if observation_depth_sigmas is None:
        observed_depth_sigmas = np.full(
            len(observed_xyz), np.nan, dtype=np.float64
        )
    else:
        observed_depth_sigmas = np.asarray(
            observation_depth_sigmas, dtype=np.float64
        )[observed_track_mask]
    observed_colors = np.asarray(observation_colors, dtype=np.float64)[
        observed_track_mask
    ]
    track_u, track_v, track_depth, track_valid = _project(
        observed_xyz, view
    )
    posterior_depth = np.full(len(sample_u), np.nan, dtype=np.float64)
    posterior_sigma = np.full(len(sample_u), np.nan, dtype=np.float64)
    posterior_color = np.zeros((len(sample_u), 3), dtype=np.float64)
    nearest_distance = np.full(len(sample_u), np.inf, dtype=np.float64)
    for instance in np.unique(sample_instance):
        sample_indices = np.flatnonzero(sample_instance == instance)
        track_choice = track_valid & (observed_instances == instance)
        if not np.any(track_choice):
            continue
        points_2d = np.column_stack(
            [track_u[track_choice], track_v[track_choice]]
        )
        neighbor_count = min(4, len(points_2d))
        distances, neighbors = cKDTree(points_2d).query(
            np.column_stack(
                [sample_u[sample_indices], sample_v[sample_indices]]
            ),
            k=neighbor_count,
        )
        if neighbor_count == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        weights = np.exp(
            -0.5
            * (
                distances / max(float(depth_neighbor_pixels), 1e-6)
            )
            ** 2
        )
        denominator = weights.sum(axis=1).clip(min=1e-8)
        depths = track_depth[track_choice][neighbors]
        mean = (weights * depths).sum(axis=1) / denominator
        spread = np.sqrt(
            (
                weights * (depths - mean[:, None]) ** 2
            ).sum(axis=1)
            / denominator
        )
        errors = observed_errors[track_choice][neighbors]
        error = (weights * errors).sum(axis=1) / denominator
        geometric_sigma = np.maximum(
            float(depth_sigma_floor),
            spread
            + error
            * mean
            / max(float(view["fx"]), float(view["fy"]), 1e-6),
        )
        supplied_sigma = observed_depth_sigmas[track_choice][neighbors]
        supplied_valid = np.isfinite(supplied_sigma) & (
            supplied_sigma > 0
        )
        supplied_weight = weights * supplied_valid
        supplied_denominator = supplied_weight.sum(axis=1)
        supplied_rms = np.sqrt(
            (
                supplied_weight
                * np.where(supplied_valid, supplied_sigma, 0.0) ** 2
            ).sum(axis=1)
            / np.maximum(supplied_denominator, 1e-8)
        )
        sigma = np.where(
            supplied_denominator > 0,
            np.sqrt(geometric_sigma**2 + supplied_rms**2),
            geometric_sigma,
        )
        colors = observed_colors[track_choice][neighbors]
        mean_color = (
            weights[:, :, None] * colors
        ).sum(axis=1) / denominator[:, None]
        found = distances[:, 0] <= float(depth_neighbor_pixels)
        rows = sample_indices[found]
        posterior_depth[rows] = mean[found]
        posterior_sigma[rows] = sigma[found]
        posterior_color[rows] = mean_color[found]
        nearest_distance[rows] = distances[found, 0]

    keep = (
        np.isfinite(posterior_depth)
        & np.isfinite(posterior_sigma)
        & (posterior_depth > 0.05)
    )
    if rigid_depth is not None and np.any(keep):
        rigid_depth = np.asarray(rigid_depth)
        rigid_rows = np.clip(
            np.floor(
                sample_v / float(view["height"]) * rigid_depth.shape[0]
            ).astype(np.int64),
            0,
            rigid_depth.shape[0] - 1,
        )
        rigid_columns = np.clip(
            np.floor(
                sample_u / float(view["width"]) * rigid_depth.shape[1]
            ).astype(np.int64),
            0,
            rigid_depth.shape[1] - 1,
        )
        rigid_values = rigid_depth[rigid_rows, rigid_columns]
        occluded = (
            np.isfinite(rigid_values)
            & (rigid_values > 0)
            & (
                rigid_values + float(occlusion_margin)
                < posterior_depth
            )
        )
        keep &= ~occluded
    valid_indices = np.flatnonzero(keep)
    if not len(valid_indices):
        return None, [], []
    indices = valid_indices
    if len(indices) > int(maximum_rays):
        # ``valid_indices`` follows raster scan order.  A one-dimensional
        # linspace over that array selects coherent scanline/diagonal bands;
        # exact-view Gaussians then make those bands visible in RGB even when
        # all later ownership and depth contracts are correct.  Select the
        # fixed evidence budget directly in image space instead.
        indices = _spatially_stratified_coverage_rows(
            indices,
            sample_u,
            sample_v,
            int(maximum_rays),
        )
    confidence = np.exp(
        -0.5
        * (
            nearest_distance[indices]
            / max(float(depth_neighbor_pixels), 1e-6)
        )
        ** 2
    )
    confidence *= np.clip(
        float(depth_sigma_floor)
        / np.maximum(posterior_sigma[indices], depth_sigma_floor),
        0.10,
        1.0,
    )
    confidence_by_row = np.zeros(
        len(sample_u), dtype=np.float64
    )
    confidence_by_row[indices] = confidence
    free_end, hit_start, hit_end = strict_foliage_hit_interval_bounds(
        posterior_depth[indices],
        posterior_sigma[indices],
        free_space_margin=free_space_margin,
    )
    record = {
        # -1 denotes an observation-space row with no seed owner.
        "candidate": np.full(len(indices), -1, dtype=np.int64),
        "camera_id": np.full(
            len(indices), int(view["image_id"]), dtype=np.int32
        ),
        "pixel": np.column_stack(
            [sample_u[indices], sample_v[indices]]
        ).astype(np.float32),
        "image_size": np.tile(
            np.asarray(
                [view["width"], view["height"]], dtype=np.int32
            ),
            (len(indices), 1),
        ),
        "free_end": free_end,
        "hit_start": hit_start,
        "hit_end": hit_end,
        "type": np.ones(len(indices), dtype=np.int8),
        "confidence": confidence.astype(np.float32),
    }

    # Load immutable owner-view RGB before selecting the renderer basis.  The
    # full ray posterior above remains untouched, but a compact appearance
    # basis must preserve image bandwidth rather than silhouette area alone.
    target_rgb = view.get("target_rgb")
    target_rgb_path = view.get("target_rgb_path")
    if target_rgb is None and target_rgb_path is not None:
        from PIL import Image

        with Image.open(target_rgb_path) as image:
            target_rgb = np.asarray(image.convert("RGB"))
    if target_rgb is not None:
        target_rgb = np.asarray(target_rgb)
        if (
            target_rgb.ndim != 3
            or target_rgb.shape[2] != 3
            or target_rgb.shape[0] <= 0
            or target_rgb.shape[1] <= 0
        ):
            raise ValueError(
                "Dense posterior target RGB must have shape [H,W,3]"
            )
        luminance = (
            0.2126 * target_rgb[..., 0].astype(np.float32)
            + 0.7152 * target_rgb[..., 1].astype(np.float32)
            + 0.0722 * target_rgb[..., 2].astype(np.float32)
        )
        gradient_x = np.zeros_like(luminance)
        gradient_y = np.zeros_like(luminance)
        gradient_x[:, 1:-1] = np.abs(
            luminance[:, 2:] - luminance[:, :-2]
        )
        gradient_y[1:-1] = np.abs(
            luminance[2:] - luminance[:-2]
        )
        target_rows_all = np.clip(
            np.floor(
                sample_v
                / float(view["height"])
                * target_rgb.shape[0]
            ).astype(np.int64),
            0,
            target_rgb.shape[0] - 1,
        )
        target_columns_all = np.clip(
            np.floor(
                sample_u
                / float(view["width"])
                * target_rgb.shape[1]
            ).astype(np.int64),
            0,
            target_rgb.shape[1] - 1,
        )
        high_frequency_score = (
            gradient_x[target_rows_all, target_columns_all]
            + gradient_y[target_rows_all, target_columns_all]
        )
    else:
        high_frequency_score = np.zeros(len(sample_u), dtype=np.float32)

    # Select a renderer basis independently of the immutable posterior rows.
    # Every visible tree instance receives coverage before the residual
    # budget is distributed in proportion to sqrt(area). Within an instance,
    # half the quota is a deterministic uniform scaffold and half is a
    # spatially distributed high-frequency residual pool. A later split can
    # refine geometry, but cannot recreate a discarded owner-view RGB sample.
    dynamic_rows = np.empty(0, dtype=np.int64)
    dynamic_detail_rows = np.empty(0, dtype=np.int64)
    dynamic_budget = min(
        max(int(maximum_dynamic_births), 0), len(indices)
    )
    if dynamic_budget:
        row_instances = sample_instance[indices]
        instances, instance_counts = np.unique(
            row_instances, return_counts=True
        )
        quotas = np.zeros(len(instances), dtype=np.int64)
        if dynamic_budget >= len(instances):
            quotas[:] = 1
        else:
            order = np.lexsort((instances, -instance_counts))
            quotas[order[:dynamic_budget]] = 1
        remaining = dynamic_budget - int(quotas.sum())
        capacity = instance_counts.astype(np.int64) - quotas
        while remaining > 0 and bool((capacity > 0).any()):
            active = capacity > 0
            weights = np.sqrt(instance_counts.astype(np.float64))
            weights[~active] = 0.0
            ideal = remaining * weights / weights.sum()
            addition = np.minimum(
                np.floor(ideal).astype(np.int64), capacity
            )
            if int(addition.sum()) == 0:
                remainder = ideal - np.floor(ideal)
                order = np.lexsort(
                    (instances, -instance_counts, -remainder)
                )
                order = order[active[order]]
                addition[order[: min(remaining, len(order))]] = 1
            quotas += addition
            capacity -= addition
            remaining -= int(addition.sum())
        selected_dynamic_rows = []
        selected_detail_rows = []
        for instance, quota in zip(instances, quotas):
            if int(quota) <= 0:
                continue
            instance_rows = indices[row_instances == instance]
            # Always reserve separate coverage and frequency-residual bases,
            # including when the birth budget retains every evidence row.
            # The old full-retention shortcut classified every row as a broad
            # coverage kernel, erasing the narrow high-frequency half exactly
            # in weak-continuous crowns that receive the largest budget.
            quota = min(int(quota), len(instance_rows))
            uniform_count = max(1, (quota + 1) // 2)
            uniform = _spatially_stratified_coverage_rows(
                instance_rows,
                sample_u,
                sample_v,
                uniform_count,
            )
            residual_count = quota - len(uniform)
            remaining_rows = instance_rows[
                ~np.isin(instance_rows, uniform, assume_unique=False)
            ]
            detail = np.empty(0, dtype=np.int64)
            if residual_count == len(remaining_rows):
                # Full retention: every non-coverage row is necessarily a
                # residual row.  Avoid constructing and iterating thousands
                # of detail cells merely to select the complete remainder.
                detail = remaining_rows
            elif residual_count > 0 and len(remaining_rows):
                grid_side = max(
                    1, int(np.ceil(np.sqrt(2 * residual_count)))
                )
                u = sample_u[remaining_rows]
                v = sample_v[remaining_rows]
                u_span = max(float(u.max() - u.min()), 1.0)
                v_span = max(float(v.max() - v.min()), 1.0)
                cell_x = np.minimum(
                    np.floor(
                        (u - float(u.min())) / u_span * grid_side
                    ).astype(np.int64),
                    grid_side - 1,
                )
                cell_y = np.minimum(
                    np.floor(
                        (v - float(v.min())) / v_span * grid_side
                    ).astype(np.int64),
                    grid_side - 1,
                )
                cell = cell_y * grid_side + cell_x
                cell_winners = []
                for cell_id in np.unique(cell):
                    rows = remaining_rows[cell == cell_id]
                    order = np.lexsort(
                        (rows, -high_frequency_score[rows])
                    )
                    cell_winners.append(rows[order[0]])
                cell_winners = np.asarray(
                    cell_winners, dtype=np.int64
                )
                winner_order = np.lexsort(
                    (
                        cell_winners,
                        -high_frequency_score[cell_winners],
                    )
                )
                detail = cell_winners[
                    winner_order[:residual_count]
                ]
                if len(detail) < residual_count:
                    refill = remaining_rows[
                        ~np.isin(
                            remaining_rows,
                            detail,
                            assume_unique=False,
                        )
                    ]
                    refill_order = np.lexsort(
                        (refill, -high_frequency_score[refill])
                    )
                    detail = np.concatenate(
                        [
                            detail,
                            refill[
                                refill_order[
                                    : residual_count - len(detail)
                                ]
                            ],
                        ]
                    )
            selected_dynamic_rows.append(
                np.concatenate([uniform, detail])
            )
            selected_detail_rows.append(detail)
        if selected_dynamic_rows:
            dynamic_rows = np.sort(
                np.concatenate(selected_dynamic_rows)
            )
            if selected_detail_rows:
                dynamic_detail_rows = np.sort(
                    np.concatenate(selected_detail_rows)
                )
    dynamic_instance = sample_instance[dynamic_rows]
    # Each birth represents its share of one connected tree instance, not its
    # share of the entire image.  This keeps overlap stable if only the
    # evidence-ray sampling density changes.
    source_pixel_spacing = np.ones(
        len(dynamic_rows), dtype=np.float64
    )
    raster_to_source_area = (
        float(view["width"])
        / float(mask_width)
        * float(view["height"])
        / float(mask_height)
    )
    for instance in np.unique(dynamic_instance):
        birth_choice = dynamic_instance == instance
        instance_eligible = int(
            np.count_nonzero(sample_instance == instance)
        )
        source_pixel_spacing[birth_choice] = np.sqrt(
            max(
                instance_eligible
                * raster_to_source_area
                / max(int(birth_choice.sum()), 1),
                1.0,
            )
        )
    dynamic_z = posterior_depth[dynamic_rows]
    dynamic_camera_xyz = np.column_stack(
        [
            (sample_u[dynamic_rows] - float(view["cx"]))
            * dynamic_z
            / float(view["fx"]),
            (sample_v[dynamic_rows] - float(view["cy"]))
            * dynamic_z
            / float(view["fy"]),
            dynamic_z,
        ]
    )
    dynamic_world_xyz = (
        dynamic_camera_xyz - np.asarray(view["translation"])[None]
    ) @ np.asarray(view["rotation"])
    dynamic_reprojection_error = np.clip(
        posterior_sigma[dynamic_rows]
        * max(float(view["fx"]), float(view["fy"]))
        / np.maximum(dynamic_z, 0.05),
        0.2,
        10.0,
    )
    # Preserve two distinct optical bandwidths. Uniform rows are a bounded
    # exact-view coverage scaffold (about 2.4 px at the reference raster);
    # gradient-selected rows are sharp local residual witnesses (about
    # 0.9 px). Applying the residual cap to both pools created holes in the
    # conditioned branch, while enlarging both pools recreated the old smooth
    # opacity blanket.
    reference_downsample = max(
        float(view["width"]) / max(float(renderer_reference_width), 1.0),
        1.0,
    )
    dynamic_is_detail = np.isin(
        dynamic_rows,
        dynamic_detail_rows,
        assume_unique=True,
    )
    represented_pixel_spacing = np.minimum(
        source_pixel_spacing,
        np.where(
            dynamic_is_detail,
            1.5 * reference_downsample,
            4.0 * reference_downsample,
        ),
    )
    dynamic_footprint = np.clip(
        0.60
        * dynamic_z
        * represented_pixel_spacing
        / np.sqrt(float(view["fx"]) * float(view["fy"])),
        0.008,
        0.12,
    )
    # Dynamic leaves are conditioned on this exact database image. Their
    # colour must therefore come from the immutable RGB target raster used by
    # training, not from a spatial interpolation of sparse MASt3R/Chart
    # anchors. The latter is still appropriate for the multi-view canonical
    # proposals below, but it removes leaf-scale frequencies and can be badly
    # wrong across an occlusion boundary.
    if target_rgb is not None:
        target_rows = np.clip(
            np.floor(
                sample_v[dynamic_rows]
                / float(view["height"])
                * target_rgb.shape[0]
            ).astype(np.int64),
            0,
            target_rgb.shape[0] - 1,
        )
        target_columns = np.clip(
            np.floor(
                sample_u[dynamic_rows]
                / float(view["width"])
                * target_rgb.shape[1]
            ).astype(np.int64),
            0,
            target_rgb.shape[1] - 1,
        )
        dynamic_rgb = target_rgb[target_rows, target_columns]
        dynamic_rgb_source = "exact_training_target_raster"
    else:
        # Kept only for direct synthetic/unit callers. Role-aware production
        # initialization always attaches the contracted target RGB path.
        dynamic_rgb = posterior_color[dynamic_rows]
        dynamic_rgb_source = "sparse_anchor_interpolation_fallback"
    dynamic_proposals = []
    for local, row in enumerate(dynamic_rows):
        dynamic_proposals.append(
            {
                # The role-aware caller assigns a globally unique negative
                # lineage id after all cameras have been concatenated.
                "id": -1,
                "xyz": dynamic_world_xyz[local].astype(np.float64),
                "rgb": np.clip(
                    np.rint(dynamic_rgb[local]), 0, 255
                ).astype(np.uint8),
                "error": float(dynamic_reprojection_error[local]),
                "position_sigma": float(
                    np.clip(
                        posterior_sigma[row],
                        float(depth_sigma_floor),
                        3.0,
                    )
                ),
                "image_ids": np.asarray(
                    [int(view["image_id"])], dtype=np.int32
                ),
                "point2d_indices": np.asarray([-1], dtype=np.int64),
                "tree_image_ids": np.asarray(
                    [int(view["image_id"])], dtype=np.int32
                ),
                "observation_camera_ids": np.asarray(
                    [int(view["image_id"])], dtype=np.int32
                ),
                "observation_uv": np.asarray(
                    [
                        [
                            sample_u[row] / float(view["width"]),
                            sample_v[row] / float(view["height"]),
                        ]
                    ],
                    dtype=np.float32,
                ),
                "observation_depth": np.asarray(
                    [dynamic_z[local]], dtype=np.float32
                ),
                "tree_sequence_count": 1,
                # Existence authority must stay proportional to the actual
                # local ray/depth posterior.  The former 0.65 base converted
                # even a 6% DAV2 posterior into 67% occupancy and then
                # initialized hundreds of broad exact-view occluders.
                "tree_fraction": float(confidence_by_row[row]),
                # Downstream topology, opacity and loss routing already use
                # ``1 / sqrt(1 + ray_depth_nll)`` as reliability.  Encode the
                # measured dense-ray confidence in that native contract
                # instead of leaving every single-view birth at NLL=0.
                "ray_depth_nll": float(
                    1.0
                    / max(float(confidence_by_row[row]), 1.0e-4) ** 2
                    - 1.0
                ),
                "ray_footprint_scale": float(
                    dynamic_footprint[local]
                ),
                "_tree_instance_id": int(sample_instance[row]),
                "_dense_ray_dynamic_birth": True,
                "_dense_ray_rgb_source": dynamic_rgb_source,
                "_dense_ray_basis_role": (
                    "frequency_residual"
                    if bool(dynamic_is_detail[local])
                    else "coverage"
                ),
            }
        )

    if int(maximum_proposals) <= 0:
        return record, [], dynamic_proposals
    proposal_rows = indices
    if len(proposal_rows) > int(maximum_proposals):
        proposal_rows = proposal_rows[
            np.linspace(
                0,
                len(proposal_rows) - 1,
                int(maximum_proposals),
                dtype=np.int64,
            )
        ]
    z = posterior_depth[proposal_rows]
    camera_xyz = np.column_stack(
        [
            (sample_u[proposal_rows] - float(view["cx"]))
            * z
            / float(view["fx"]),
            (sample_v[proposal_rows] - float(view["cy"]))
            * z
            / float(view["fy"]),
            z,
        ]
    )
    world_xyz = (
        camera_xyz - np.asarray(view["translation"])[None]
    ) @ np.asarray(view["rotation"])
    reprojection_error = np.clip(
        posterior_sigma[proposal_rows]
        * max(float(view["fx"]), float(view["fy"]))
        / np.maximum(z, 0.05),
        0.2,
        10.0,
    )
    proposals = []
    for local, row in enumerate(proposal_rows):
        proposals.append(
            {
                "id": -1,
                "xyz": world_xyz[local].astype(np.float64),
                "rgb": np.clip(
                    np.rint(posterior_color[row]), 0, 255
                ).astype(np.uint8),
                "error": float(reprojection_error[local]),
                "image_ids": np.asarray(
                    [int(view["image_id"])], dtype=np.int32
                ),
                "tree_image_ids": np.asarray(
                    [int(view["image_id"])], dtype=np.int32
                ),
                "tree_sequence_count": 1,
                "tree_fraction": 1.0,
                "_tree_instance_id": int(sample_instance[row]),
                "_dense_ray_proposal": True,
            }
        )
    return record, proposals, dynamic_proposals


def measured_static_skeleton_geometry(
    centers,
    colors,
    tree_instance_id,
    support_view_count,
    support_sequence_count,
    occupancy_probability,
    ray_depth_nll,
    free_space_violation_count,
    *,
    voxel_size=0.12,
    maximum_fraction_per_instance=0.05,
    minimum_confidence=0.18,
):
    """Estimate static trunk/branch Gaussians from measured local geometry.

    Every candidate must already be a cross-sequence canonical cell and have
    a locally elongated, connected neighbourhood. Colour, ray occupancy,
    free-space contradictions and depth likelihood contribute continuous
    confidence instead of a brittle chain of binary gates. This keeps weak
    branch evidence available without declaring the whole crown rigid.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    xyz = np.asarray(centers, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.float64)
    instance = np.asarray(tree_instance_id, dtype=np.int32).reshape(-1)
    views = np.asarray(support_view_count, dtype=np.float64).reshape(-1)
    sequences = np.asarray(
        support_sequence_count, dtype=np.float64
    ).reshape(-1)
    occupancy = np.asarray(
        occupancy_probability, dtype=np.float64
    ).reshape(-1)
    depth_nll = np.asarray(ray_depth_nll, dtype=np.float64).reshape(-1)
    free_space = np.asarray(
        free_space_violation_count, dtype=np.float64
    ).reshape(-1)
    count = len(xyz)
    if xyz.shape != (count, 3) or rgb.shape != (count, 3):
        raise ValueError("skeleton centers/colors must have shape [N, 3]")
    for name, value in (
        ("tree_instance_id", instance),
        ("support_view_count", views),
        ("support_sequence_count", sequences),
        ("occupancy_probability", occupancy),
        ("ray_depth_nll", depth_nll),
        ("free_space_violation_count", free_space),
    ):
        if len(value) != count:
            raise ValueError(f"{name} must have one value per center")
    if float(voxel_size) <= 0:
        raise ValueError("voxel_size must be positive")

    confidence = np.zeros(count, dtype=np.float32)
    linearity = np.zeros(count, dtype=np.float32)
    axis_alignment = np.zeros(count, dtype=np.float32)
    axial_spread = np.zeros(count, dtype=np.float32)
    coherent_count = np.zeros(count, dtype=np.int16)
    frames = np.tile(np.eye(3, dtype=np.float64)[None], (count, 1, 1))
    scales = np.full(
        (count, 3), float(voxel_size) * 0.20, dtype=np.float32
    )
    selected = np.zeros(count, dtype=bool)
    eligible = (
        (instance >= 0)
        & (sequences >= 2)
        & np.isfinite(xyz).all(axis=1)
        & np.isfinite(rgb).all(axis=1)
    )

    def sigmoid(value):
        value = np.clip(value, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-value))

    for tree_id in np.unique(instance[eligible]):
        rows = np.flatnonzero(eligible & (instance == tree_id))
        if len(rows) < 8:
            continue
        points = xyz[rows]
        neighbours = cKDTree(points).query_ball_point(
            points, r=3.0 * float(voxel_size)
        )
        local_score = np.zeros(len(rows), dtype=np.float64)
        for local_index, adjacent in enumerate(neighbours):
            if len(adjacent) < 8:
                continue
            adjacent = np.asarray(adjacent, dtype=np.int64)
            local = points[adjacent]
            centered = local - local.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(len(local) - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            ratio = eigenvalues[-1] / (
                eigenvalues[-2] + eigenvalues[-3]
            )
            spread = float(np.sqrt(eigenvalues[-1]))
            alignment = abs(float(eigenvectors[1, -1]))
            row = int(rows[local_index])
            linearity[row] = ratio
            axis_alignment[row] = alignment
            axial_spread[row] = spread

            # Volume covariance uses Sigma = R^T S^2 R, so measured local
            # axes are stored as rows and the long cylinder axis comes first.
            frame = eigenvectors[:, ::-1].T
            if np.linalg.det(frame) < 0:
                frame[2] *= -1
            frames[row] = frame
            long_scale = np.clip(
                0.60 * spread,
                0.25 * float(voxel_size),
                1.50 * float(voxel_size),
            )
            radial_scale = np.clip(
                0.50
                * np.sqrt(
                    0.5 * (eigenvalues[0] + eigenvalues[1])
                ),
                0.06 * float(voxel_size),
                0.40 * float(voxel_size),
            )
            scales[row] = (
                long_scale,
                radial_scale,
                radial_scale,
            )

            red, green, blue = rgb[row]
            green_excess = green - max(red, blue)
            wood_score = (
                sigmoid((0.10 - green_excess) / 0.035)
                * sigmoid((red - 0.55 * blue) / 0.06)
                * sigmoid((0.86 - rgb[row].mean()) / 0.10)
            )
            shape_score = max(
                sigmoid((ratio - 1.65) / 0.30)
                * sigmoid((alignment - 0.48) / 0.12),
                sigmoid((ratio - 2.20) / 0.35),
            ) * sigmoid((spread - 0.045) / 0.012)
            view_score = 1.0 - np.exp(
                -max(float(views[row]) - 1.0, 0.0) / 2.0
            )
            sequence_score = 1.0 - np.exp(
                -max(float(sequences[row]) - 1.0, 0.0)
            )
            occupancy_score = sigmoid(
                (float(occupancy[row]) - 0.52) / 0.08
            )
            finite_nll = (
                float(depth_nll[row])
                if np.isfinite(depth_nll[row])
                else 20.0
            )
            likelihood_score = np.exp(
                -max(finite_nll, 0.0) / 8.0
            )
            free_space_score = np.exp(
                -max(float(free_space[row]), 0.0)
            )
            local_score[local_index] = (
                shape_score
                * (0.25 + 0.75 * wood_score)
                * np.sqrt(max(view_score * sequence_score, 0.0))
                * occupancy_score
                * likelihood_score
                * free_space_score
            )

        # Isolated leaf edges keep a low confidence but cannot become static.
        proposal = local_score >= 0.08
        if bool(proposal.any()):
            proposal_tree = cKDTree(points[proposal])
            proposal_rows = np.flatnonzero(proposal)
            counts = np.asarray(
                [
                    len(values)
                    for values in proposal_tree.query_ball_point(
                        points[proposal],
                        r=4.0 * float(voxel_size),
                    )
                ],
                dtype=np.int16,
            )
            coherent_count[rows[proposal_rows]] = counts
            coherence = np.clip((counts - 1.0) / 4.0, 0.0, 1.0)
            local_score[proposal_rows] *= 0.25 + 0.75 * coherence
        confidence[rows] = local_score.astype(np.float32)
        candidates = rows[
            (local_score >= float(minimum_confidence))
            & (coherent_count[rows] >= 3)
        ]
        maximum = max(
            4,
            int(
                np.ceil(
                    float(maximum_fraction_per_instance) * len(rows)
                )
            ),
        )
        if len(candidates) > maximum:
            candidates = candidates[
                np.argsort(confidence[candidates])[-maximum:]
            ]
        selected[candidates] = True

    xyzw = Rotation.from_matrix(frames).as_quat().astype(np.float32)
    quaternion = np.column_stack(
        [xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]]
    ).astype(np.float32)
    return {
        "selected": selected,
        "confidence": confidence,
        "linearity": linearity,
        "axis_alignment": axis_alignment,
        "axial_spread": axial_spread,
        "coherent_count": coherent_count,
        "scales": scales,
        "quaternions": quaternion,
    }


def build_instance_aware_canopy_volume(
    tracks,
    images,
    cameras,
    mask_lookup,
    *,
    candidate_tracks=None,
    supplemental_candidate_tracks=None,
    rigid_depth_maps=None,
    selected_views=None,
    voxel_size=0.12,
    dilation_steps=3,
    maximum_voxels=400_000,
    selected_view_count=64,
    minimum_support_views=2,
    minimum_depth_support_views=2,
    minimum_support_sequences=2,
    minimum_occupancy=0.52,
    minimum_baseline=0.75,
    minimum_triangulation_angle_degrees=1.5,
    instance_connection_radius=1.5,
    instance_minimum_tracks=8,
    depth_neighbor_pixels=48.0,
    depth_sigma_floor=0.20,
    occlusion_margin=0.15,
    free_space_margin=0.25,
    maximum_dense_rays_per_view=4096,
    maximum_dense_rays_total=None,
    minimum_dense_rays_per_view=0,
    maximum_bound_rays_per_view=8192,
    maximum_dense_proposals_per_view=512,
    maximum_dynamic_births_per_view=1024,
    maximum_weak_continuous_dynamic_births_per_view=None,
    dynamic_birth_target_source_pixels_per_basis=None,
    allow_inferred_skeleton=False,
    seed=0,
):
    """Build porous, instance-aware crown seeds from ray/depth observations.

    Positive evidence requires the correct tree-mask component *and* a local
    SfM-track depth posterior. A protected-structure z-buffer converts hidden
    projections to unknown instead of negative evidence.
    """
    from scipy.spatial import cKDTree

    if int(maximum_dynamic_births_per_view) < 0:
        raise ValueError(
            "maximum_dynamic_births_per_view must be non-negative"
        )
    if (
        maximum_weak_continuous_dynamic_births_per_view is not None
        and int(maximum_weak_continuous_dynamic_births_per_view) < 0
    ):
        raise ValueError(
            "maximum_weak_continuous_dynamic_births_per_view must be "
            "non-negative"
        )
    if (
        dynamic_birth_target_source_pixels_per_basis is not None
        and (
            not np.isfinite(
                float(dynamic_birth_target_source_pixels_per_basis)
            )
            or float(dynamic_birth_target_source_pixels_per_basis) <= 0
        )
    ):
        raise ValueError(
            "dynamic_birth_target_source_pixels_per_basis must be finite "
            "and positive"
        )
    if int(maximum_bound_rays_per_view) < 0:
        raise ValueError(
            "maximum_bound_rays_per_view must be non-negative"
        )
    if not tracks:
        raise RuntimeError("No independent semantic SfM tree tracks were found")
    if candidate_tracks is None:
        candidate_tracks = tracks
    candidate_tracks = list(candidate_tracks)
    if not candidate_tracks:
        raise RuntimeError("No canopy voxel candidate tracks were provided")
    supplemental_candidate_tracks = list(
        supplemental_candidate_tracks or []
    )
    rigid_depth_maps = rigid_depth_maps or {}
    observation_xyz = np.stack([point["xyz"] for point in tracks])
    observation_errors = np.asarray(
        [point["error"] for point in tracks], dtype=np.float64
    )
    observation_depth_sigmas = np.asarray(
        [
            point.get("observation_depth_sigma", np.nan)
            for point in tracks
        ],
        dtype=np.float64,
    )
    observation_colors = np.stack(
        [point["rgb"] for point in tracks]
    )
    all_candidate_tracks = [
        *candidate_tracks,
        *supplemental_candidate_tracks,
    ]
    candidate_xyz = np.stack(
        [point["xyz"] for point in all_candidate_tracks]
    )
    candidate_rgb = np.stack(
        [point["rgb"] for point in all_candidate_tracks]
    )
    primary_candidate_xyz = np.stack(
        [point["xyz"] for point in candidate_tracks]
    )
    primary_instances = cluster_tree_instances(
        primary_candidate_xyz,
        connection_radius=instance_connection_radius,
        minimum_tracks=instance_minimum_tracks,
    )
    if supplemental_candidate_tracks:
        supplemental_instances = primary_instances[
            cKDTree(primary_candidate_xyz).query(
                supplemental_xyz := np.stack(
                    [
                        point["xyz"]
                        for point in supplemental_candidate_tracks
                    ]
                ),
                k=1,
            )[1]
        ]
        candidate_instances = np.concatenate(
            [primary_instances, supplemental_instances]
        ).astype(np.int32)
    else:
        supplemental_xyz = np.empty((0, 3), dtype=np.float64)
        candidate_instances = primary_instances
    # Dense per-view depth observations enrich the ray posterior but must not
    # independently dilate the voxel proposal set.  Map them to the stable
    # candidate-instance graph, then use them only as calibrated observations.
    observation_instances = candidate_instances[
        cKDTree(candidate_xyz).query(observation_xyz, k=1)[1]
    ]
    if selected_views is None:
        view_records = [
            _camera_record(image, cameras[image["camera_id"]], mask_lookup)
            for image in images.values()
        ]
        # A non-positive limit means every fixed database camera. The dense
        # ray allocator still enforces one global posterior budget and assigns
        # zero rows to cameras without measured tree support, so expanding
        # camera coverage does not imply expanding evidence cardinality.
        selection_limit = (
            len(view_records)
            if int(selected_view_count) <= 0
            else int(selected_view_count)
        )
        selected = greedy_diverse_views(
            view_records,
            limit=selection_limit,
            minimum_center_distance=minimum_baseline,
        )
    else:
        selected = list(selected_views)
    track_images = [
        {int(value) for value in point.get("tree_image_ids", ())}
        for point in tracks
    ]
    observation_rows_by_image = _invert_track_image_incidence(
        track_images,
        [int(view["image_id"]) for view in selected],
    )
    # Build observation-space posterior rows before the voxel proposal set.
    # This is the missing birth path for silhouette regions that had no
    # pre-existing candidate centre.  Dense hit points remain supplemental:
    # multi-view/sequence, baseline, angle and occupancy checks below still
    # decide whether they may enter the canonical crown.
    dense_ray_records: list[dict[str, np.ndarray]] = []
    dense_proposals: list[dict] = []
    dense_dynamic_births: list[dict] = []
    dense_ray_per_view_audit: list[dict[str, int]] = []
    dense_instance_rasters: dict[int, np.ndarray] = {}
    for view in selected:
        image_id = int(view["image_id"])
        observed = _incidence_mask(
            observation_rows_by_image.get(
                image_id, np.empty(0, dtype=np.int64)
            ),
            len(track_images),
        )
        _, instance_raster = _component_instance_map(
            view,
            observation_xyz,
            observation_instances,
            observed,
        )
        dense_instance_rasters[image_id] = instance_raster
    dense_ray_budgets, dense_ray_budget_audit = (
        _allocate_dense_ray_budgets(
            selected,
            dense_instance_rasters,
            maximum_per_view=maximum_dense_rays_per_view,
            total_budget=maximum_dense_rays_total,
            minimum_per_view=minimum_dense_rays_per_view,
        )
    )
    for view in selected:
        image_id = int(view["image_id"])
        observed = _incidence_mask(
            observation_rows_by_image.get(
                image_id, np.empty(0, dtype=np.int64)
            ),
            len(track_images),
        )
        instance_raster = dense_instance_rasters[image_id]
        eligible_pixel_count = int(
            np.count_nonzero(instance_raster >= 0)
        )
        (
            dynamic_birth_limit,
            weak_continuous_posterior,
            area_birth_limit,
        ) = (
            _dynamic_birth_limit_for_view(
                tracks,
                observed,
                ordinary_limit=maximum_dynamic_births_per_view,
                weak_continuous_limit=(
                    maximum_weak_continuous_dynamic_births_per_view
                ),
                eligible_pixel_count=eligible_pixel_count,
                target_pixels_per_birth=(
                    dynamic_birth_target_source_pixels_per_basis
                ),
            )
        )
        record, proposals, dynamic_births = _dense_component_ray_posterior(
            view,
            observation_xyz,
            observation_instances,
            observation_errors,
            observation_colors,
            observed,
            observation_depth_sigmas=observation_depth_sigmas,
            rigid_depth=rigid_depth_maps.get(image_id),
            depth_neighbor_pixels=depth_neighbor_pixels,
            depth_sigma_floor=depth_sigma_floor,
            occlusion_margin=occlusion_margin,
            free_space_margin=free_space_margin,
            maximum_rays=dense_ray_budgets[image_id],
            maximum_proposals=maximum_dense_proposals_per_view,
            maximum_dynamic_births=dynamic_birth_limit,
            instance_raster=instance_raster,
        )
        if record is not None:
            dense_ray_records.append(record)
        dense_proposals.extend(proposals)
        dense_dynamic_births.extend(dynamic_births)
        dense_ray_per_view_audit.append(
            {
                "image_id": image_id,
                "eligible": eligible_pixel_count,
                "allocated": int(dense_ray_budgets[image_id]),
                "accepted_rays": (
                    0 if record is None else int(len(record["camera_id"]))
                ),
                "dynamic_births": int(len(dynamic_births)),
                "dynamic_birth_limit": int(dynamic_birth_limit),
                "dynamic_birth_area_limit": int(area_birth_limit),
                "weak_continuous_depth_posterior": bool(
                    weak_continuous_posterior
                ),
            }
        )
    dense_ray_budget_audit["per_view"] = dense_ray_per_view_audit
    if dense_proposals:
        dense_xyz = np.stack(
            [point["xyz"] for point in dense_proposals]
        )
        dense_rgb = np.stack(
            [point["rgb"] for point in dense_proposals]
        )
        dense_instances = np.asarray(
            [
                point["_tree_instance_id"]
                for point in dense_proposals
            ],
            dtype=np.int32,
        )
        supplemental_xyz = np.concatenate(
            [supplemental_xyz, dense_xyz], axis=0
        )
        candidate_xyz = np.concatenate(
            [candidate_xyz, dense_xyz], axis=0
        )
        candidate_rgb = np.concatenate(
            [candidate_rgb, dense_rgb], axis=0
        )
        candidate_instances = np.concatenate(
            [candidate_instances, dense_instances], axis=0
        )
        all_candidate_tracks.extend(dense_proposals)
    _, centers = _priority_candidate_voxels(
        primary_candidate_xyz,
        supplemental_xyz,
        voxel_size=voxel_size,
        primary_dilation_steps=dilation_steps,
        supplemental_dilation_steps=1,
        maximum_voxels=maximum_voxels,
        seed=seed,
    )
    nearest_anchor = cKDTree(candidate_xyz).query(centers, k=1)[1]
    center_instances = candidate_instances[nearest_anchor]
    count = len(centers)
    positive = np.zeros(count, dtype=np.int16)
    negative = np.zeros(count, dtype=np.int16)
    unknown = np.zeros(count, dtype=np.int16)
    depth_support = np.zeros(count, dtype=np.int16)
    free_space_violations = np.zeros(count, dtype=np.int16)
    depth_nll_sum = np.zeros(count, dtype=np.float64)
    support_matrix = np.zeros((count, len(selected)), dtype=bool)
    information = np.zeros((count, 3, 3), dtype=np.float64)
    selected_image_ids = np.asarray(
        [view["image_id"] for view in selected], dtype=np.int32
    )
    ray_records: list[dict[str, np.ndarray]] = list(
        dense_ray_records
    )
    bound_ray_available_total = 0
    bound_ray_stored_total = 0
    bound_ray_truncated_view_count = 0
    dense_audit_by_image = {
        int(row["image_id"]): row
        for row in dense_ray_per_view_audit
    }
    for view_index, view in enumerate(selected):
        image_id = int(view["image_id"])
        observed = _incidence_mask(
            observation_rows_by_image.get(
                image_id, np.empty(0, dtype=np.int64)
            ),
            len(track_images),
        )
        raster_instance = dense_instance_rasters.pop(image_id)
        u, v, depth, valid = _project(centers, view)
        mask_height, mask_width = raster_instance.shape
        rows = np.clip(
            np.round(v / view["height"] * mask_height).astype(int),
            0,
            mask_height - 1,
        )
        cols = np.clip(
            np.round(u / view["width"] * mask_width).astype(int),
            0,
            mask_width - 1,
        )
        matched_instance = (
            raster_instance[rows, cols] == center_instances
        )

        rigid = rigid_depth_maps.get(image_id)
        if rigid is not None:
            rigid = np.asarray(rigid)
            rigid_rows = np.clip(
                np.round(v / view["height"] * rigid.shape[0]).astype(int),
                0,
                rigid.shape[0] - 1,
            )
            rigid_cols = np.clip(
                np.round(u / view["width"] * rigid.shape[1]).astype(int),
                0,
                rigid.shape[1] - 1,
            )
            rigid_depth = rigid[rigid_rows, rigid_cols]
            occluded = (
                valid
                & np.isfinite(rigid_depth)
                & (rigid_depth > 0)
                & (rigid_depth + float(occlusion_margin) < depth)
            )
        else:
            occluded = np.zeros(count, dtype=bool)

        posterior_found = np.zeros(count, dtype=bool)
        posterior_depth = np.zeros(count, dtype=np.float64)
        posterior_sigma = np.zeros(count, dtype=np.float64)
        if np.any(observed):
            track_u, track_v, track_depth, track_valid = _project(
                observation_xyz[observed], view
            )
            observed_instances = observation_instances[observed]
            observed_errors = np.asarray(
                [point["error"] for point in tracks], dtype=np.float64
            )[observed]
            for instance in np.unique(observed_instances[track_valid]):
                track_choice = track_valid & (observed_instances == instance)
                center_choice = (
                    valid
                    & matched_instance
                    & (center_instances == instance)
                    & ~occluded
                )
                center_indices = np.nonzero(center_choice)[0]
                if not len(center_indices):
                    continue
                points_2d = np.column_stack(
                    [track_u[track_choice], track_v[track_choice]]
                )
                neighbor_count = min(4, len(points_2d))
                distances, neighbors = cKDTree(points_2d).query(
                    np.column_stack([u[center_indices], v[center_indices]]),
                    k=neighbor_count,
                )
                if neighbor_count == 1:
                    distances = distances[:, None]
                    neighbors = neighbors[:, None]
                weights = np.exp(
                    -0.5
                    * (distances / float(depth_neighbor_pixels)) ** 2
                )
                track_depth_values = track_depth[track_choice][neighbors]
                denominator = weights.sum(axis=1)
                mean = (
                    weights * track_depth_values
                ).sum(axis=1) / np.maximum(denominator, 1e-8)
                spread = np.sqrt(
                    (
                        weights * (track_depth_values - mean[:, None]) ** 2
                    ).sum(axis=1)
                    / np.maximum(denominator, 1e-8)
                )
                error = (
                    weights * observed_errors[track_choice][neighbors]
                ).sum(axis=1) / np.maximum(denominator, 1e-8)
                sigma = np.maximum(
                    float(depth_sigma_floor),
                    spread + error * mean / max(view["fx"], view["fy"]),
                )
                found = distances[:, 0] <= float(depth_neighbor_pixels)
                chosen = center_indices[found]
                posterior_found[chosen] = True
                posterior_depth[chosen] = mean[found]
                posterior_sigma[chosen] = sigma[found]

        residual = depth - posterior_depth
        compatible = (
            posterior_found
            & (np.abs(residual) <= 2.5 * posterior_sigma)
        )
        free = (
            posterior_found
            & (depth < posterior_depth - float(free_space_margin))
        )
        supported = valid & matched_instance & compatible & ~occluded
        confirmed_negative = (
            _confirmed_free_space_mask(valid, occluded, free)
            # A mask miss without a local depth posterior is not free-space
            # evidence: the candidate may be behind an unmodelled occluder or
            # outside this traversal's observed crown. Only a measured
            # in-front-of-posterior violation is a true negative.
        )
        ambiguous = valid & ~(supported | confirmed_negative)
        ray_valid = valid & posterior_found & matched_instance
        ray_indices = np.nonzero(ray_valid)[0]
        candidate_observation_type = np.zeros(count, dtype=np.int8)
        candidate_observation_type[supported] = 1
        candidate_observation_type[confirmed_negative] = -1
        available_bound_rows = int(len(ray_indices))
        ray_indices = _bounded_candidate_ray_rows(
            ray_indices,
            u,
            v,
            candidate_observation_type,
            maximum_rows=min(
                int(maximum_bound_rays_per_view),
                available_bound_rows,
            ),
        )
        stored_bound_rows = int(len(ray_indices))
        bound_ray_available_total += available_bound_rows
        bound_ray_stored_total += stored_bound_rows
        bound_ray_truncated_view_count += int(
            stored_bound_rows < available_bound_rows
        )
        dense_audit_by_image[image_id].update(
            {
                "candidate_bound_available": available_bound_rows,
                "candidate_bound_stored": stored_bound_rows,
            }
        )
        if len(ray_indices):
            observation_type = candidate_observation_type[ray_indices]
            (
                candidate_free_end,
                candidate_hit_start,
                candidate_hit_end,
            ) = strict_foliage_hit_interval_bounds(
                posterior_depth[ray_indices],
                posterior_sigma[ray_indices],
                free_space_margin=free_space_margin,
            )
            ray_records.append(
                {
                    "candidate": ray_indices.astype(np.int64),
                    "camera_id": np.full(
                        len(ray_indices), image_id, dtype=np.int32
                    ),
                    "pixel": np.column_stack(
                        [u[ray_indices], v[ray_indices]]
                    ).astype(np.float32),
                    # Pixel coordinates live in the calibrated source-camera
                    # raster, not necessarily the later training render
                    # resolution.  Persist that frame explicitly; otherwise
                    # a 1920x1080 ray sampled from a 640x360 render lands
                    # outside grid_sample and silently contributes zero.
                    "image_size": np.tile(
                        np.asarray(
                            [view["width"], view["height"]],
                            dtype=np.int32,
                        ),
                        (len(ray_indices), 1),
                    ),
                    "free_end": candidate_free_end,
                    "hit_start": candidate_hit_start,
                    "hit_end": candidate_hit_end,
                    "type": observation_type,
                    "confidence": np.exp(
                        -0.5
                        * (
                            residual[ray_indices]
                            / np.maximum(
                                posterior_sigma[ray_indices], 1e-6
                            )
                        )
                        ** 2
                    ).astype(np.float32),
                }
            )
        positive += supported.astype(np.int16)
        depth_support += compatible.astype(np.int16)
        negative += confirmed_negative.astype(np.int16)
        unknown += (ambiguous | occluded).astype(np.int16)
        # Persist the exact confirmed-negative semantics used by the ray
        # likelihood.  Accumulating raw ``free`` here used to count invalid
        # and occluded projections as contradictions; the first adaptive
        # topology event then deleted valid canonical crown cells.
        free_space_violations += confirmed_negative.astype(np.int16)
        support_matrix[:, view_index] = supported
        # ``ambiguous`` includes candidates behind the measured first-hit
        # interval.  Such a candidate may be a real, occluded inner crown
        # layer, so the ray contract classifies it as unknown rather than a
        # hit or a confirmed free-space violation.  The former implementation
        # nevertheless accumulated its arbitrarily large standardized
        # residual into ``ray_depth_nll`` and divided only by the number of
        # compatible hits.  In a 256-view hull, two valid hits plus hundreds
        # of unknown traversals routinely produced NLLs above 1000.  That
        # silently inflated the position covariance to its ceiling and
        # suppressed otherwise valid adaptive splits.
        #
        # Keep the likelihood contract internally consistent: only measured
        # hit intervals constrain a candidate's depth.  Confirmed negatives
        # remain in the occupancy/free-space factor, and unknown rays remain
        # excluded from both.
        likelihood_indices = _accumulate_supported_depth_nll(
            depth_nll_sum,
            residual,
            posterior_sigma,
            supported,
        )
        if len(likelihood_indices):
            camera_center_value = np.asarray(view["camera_center"])
            rays = centers[likelihood_indices] - camera_center_value[None]
            rays /= np.maximum(
                np.linalg.norm(rays, axis=1, keepdims=True), 1e-8
            )
            sigma = posterior_sigma[likelihood_indices]
            lateral = np.maximum(
                depth[likelihood_indices] / max(view["fx"], view["fy"]),
                voxel_size * 0.25,
            )
            identity = np.eye(3)[None]
            outer = rays[:, :, None] * rays[:, None, :]
            covariance = (
                sigma[:, None, None] ** 2 * outer
                + lateral[:, None, None] ** 2 * (identity - outer)
            )
            information[likelihood_indices] += np.linalg.inv(covariance)

    occupancy = (positive + 1.0) / (positive + negative + 2.0)
    sequence_count = np.zeros(count, dtype=np.int16)
    for sequence in sorted({view["sequence_id"] for view in selected}):
        columns = [
            index
            for index, view in enumerate(selected)
            if view["sequence_id"] == sequence
        ]
        sequence_count += support_matrix[:, columns].any(axis=1)
    maximum_baseline, maximum_angle = (
        _supported_view_geometry_statistics(
            centers,
            support_matrix,
            np.stack(
                [
                    np.asarray(view["camera_center"])
                    for view in selected
                ]
            ),
        )
    )
    support_ids = np.where(
        support_matrix, selected_image_ids[None], np.int32(-1)
    )
    keep = (
        (positive >= int(minimum_support_views))
        & (depth_support >= int(minimum_depth_support_views))
        & (sequence_count >= int(minimum_support_sequences))
        & (occupancy >= float(minimum_occupancy))
        & (maximum_baseline >= float(minimum_baseline))
        & (
            maximum_angle
            >= float(minimum_triangulation_angle_degrees)
        )
    )
    if not np.any(keep):
        gate_counts = {
            "candidates": int(count),
            "positive_views": int(
                (positive >= int(minimum_support_views)).sum()
            ),
            "depth_views": int(
                (depth_support >= int(minimum_depth_support_views)).sum()
            ),
            "sequences": int(
                (sequence_count >= int(minimum_support_sequences)).sum()
            ),
            "occupancy": int(
                (occupancy >= float(minimum_occupancy)).sum()
            ),
            "baseline": int(
                (maximum_baseline >= float(minimum_baseline)).sum()
            ),
            "angle": int(
                (
                    maximum_angle
                    >= float(minimum_triangulation_angle_degrees)
                ).sum()
            ),
        }
        raise RuntimeError(
            "No instance-aware ray/depth-supported canopy voxel survived: "
            f"{gate_counts}"
        )
    accepted_information = information[keep]
    weak = np.linalg.det(accepted_information) <= 1e-12
    accepted_information[weak] += (
        np.eye(3)[None] / max(voxel_size * voxel_size, 1e-6)
    )
    accepted_depth_nll = depth_nll_sum[keep] / np.maximum(
        depth_support[keep], 1
    )
    # Information-matrix inversion assumes the supplied ray noise explains
    # the observed residual.  In foliage that assumption is often false:
    # traversal-dependent leaves can yield a very small formal covariance and
    # a very large reduced chi-square at the same voxel.  Treating the former
    # as a strong anchor locked wrong cells in place.  Calibrate the spatial
    # covariance by the measured reduced chi-square (bounded only for numeric
    # stability), so uncertainty remains spatial and per candidate.
    covariance_inflation = np.clip(
        np.maximum(accepted_depth_nll, 1.0), 1.0, 100.0
    )
    position_covariance = (
        np.linalg.inv(accepted_information)
        * covariance_inflation[:, None, None]
    )
    accepted_centers = centers[keep]
    accepted_instances = center_instances[keep]
    nearest = nearest_anchor[keep]
    accepted_sequence_count = sequence_count[keep]
    accepted_colors = (
        candidate_rgb[nearest].astype(np.float64) / 255.0
    )
    accepted_positive = positive[keep]
    accepted_negative = negative[keep]
    accepted_occupancy = occupancy[keep]
    skeleton_geometry = measured_static_skeleton_geometry(
        accepted_centers,
        accepted_colors,
        accepted_instances,
        accepted_positive,
        accepted_sequence_count,
        accepted_occupancy,
        accepted_depth_nll,
        accepted_negative,
        voxel_size=voxel_size,
    )
    skeleton = skeleton_geometry["selected"].copy()
    if not bool(allow_inferred_skeleton):
        skeleton.fill(False)
    skeleton_linearity = skeleton_geometry["linearity"]
    layer_role = np.zeros(int(keep.sum()), dtype=np.int8)
    layer_role[skeleton] = 1
    old_to_new = np.full(count, -1, dtype=np.int64)
    old_to_new[np.nonzero(keep)[0]] = np.arange(int(keep.sum()))
    if ray_records:
        ray_candidate = np.concatenate(
            [record["candidate"] for record in ray_records]
        )
        bound = ray_candidate >= 0
        if np.any(ray_candidate[bound] >= count):
            raise RuntimeError(
                "Foliage ray candidate index exceeds the visual-hull table"
            )
        ray_keep = ~bound
        ray_keep[bound] = old_to_new[ray_candidate[bound]] >= 0
        retained_candidate = ray_candidate[ray_keep]
        ray_primitive = np.full(
            len(retained_candidate), -1, dtype=np.int64
        )
        retained_bound = retained_candidate >= 0
        ray_primitive[retained_bound] = old_to_new[
            retained_candidate[retained_bound]
        ]
        ray_camera = np.concatenate(
            [record["camera_id"] for record in ray_records]
        )[ray_keep]
        ray_pixel = np.concatenate(
            [record["pixel"] for record in ray_records]
        )[ray_keep]
        ray_image_size = np.concatenate(
            [record["image_size"] for record in ray_records]
        )[ray_keep]
        # Projection validity is evaluated in float64, while the persistent
        # ray table is float32.  A coordinate infinitesimally below the
        # bottom/right endpoint can round to exactly ``height``/``width``
        # during that cast (observed once in 1.5 M Cambridge rays).  Keep the
        # half-open raster contract after serialization without accepting a
        # genuinely invalid producer coordinate.
        ray_pixel = half_open_raster_coordinates(
            ray_pixel, ray_image_size
        )
        ray_free_end = np.concatenate(
            [record["free_end"] for record in ray_records]
        )[ray_keep]
        ray_hit_start = np.concatenate(
            [record["hit_start"] for record in ray_records]
        )[ray_keep]
        ray_hit_end = np.concatenate(
            [record["hit_end"] for record in ray_records]
        )[ray_keep]
        ray_type = np.concatenate(
            [record["type"] for record in ray_records]
        )[ray_keep]
        ray_confidence = np.concatenate(
            [record["confidence"] for record in ray_records]
        )[ray_keep]
        (
            ray_free_end,
            ray_hit_start,
            ray_hit_end,
            interval_audit,
        ) = enforce_strict_foliage_ray_intervals(
            ray_free_end,
            ray_hit_start,
            ray_hit_end,
            ray_type,
        )
        if int(interval_audit["changed_hit_rows"]) != 0:
            raise RuntimeError(
                "A foliage ray producer emitted overlapping or invalid hit "
                f"intervals: {interval_audit}"
            )
        # Keep seed-bound rows in primitive-prefix order so the traditional
        # offset table remains a valid lineage diagnostic.  Dense
        # observation-space rays form an ownerless suffix and therefore
        # survive local replace-and-retire events without being reassigned to
        # an unrelated child.
        bound_rows = np.flatnonzero(ray_primitive >= 0)
        bound_order = bound_rows[
            np.argsort(ray_primitive[bound_rows], kind="stable")
        ]
        observation_order = np.flatnonzero(ray_primitive < 0)
        order = np.concatenate([bound_order, observation_order])
        ordered_primitive = ray_primitive[bound_order]
        counts = np.bincount(
            ordered_primitive, minlength=int(keep.sum())
        )
        ray_offsets = np.zeros(int(keep.sum()) + 1, dtype=np.int64)
        ray_offsets[1:] = np.cumsum(counts)
        ray_evidence = {
            "offsets": ray_offsets,
            "camera_ids": ray_camera[order],
            "pixels": ray_pixel[order],
            "source_image_sizes": ray_image_size[order],
            "free_end_depth": ray_free_end[order],
            "hit_start_depth": ray_hit_start[order],
            "hit_end_depth": ray_hit_end[order],
            "observation_type": ray_type[order],
            "confidence": ray_confidence[order],
        }
    else:
        ray_evidence = {
            "offsets": np.zeros(int(keep.sum()) + 1, dtype=np.int64),
            "camera_ids": np.empty(0, dtype=np.int32),
            "pixels": np.empty((0, 2), dtype=np.float32),
            "source_image_sizes": np.empty((0, 2), dtype=np.int32),
            "free_end_depth": np.empty(0, dtype=np.float32),
            "hit_start_depth": np.empty(0, dtype=np.float32),
            "hit_end_depth": np.empty(0, dtype=np.float32),
            "observation_type": np.empty(0, dtype=np.int8),
            "confidence": np.empty(0, dtype=np.float32),
        }
    return {
        "centers": accepted_centers.astype(np.float32),
        "colors": candidate_rgb[nearest].astype(np.float32) / 255.0,
        "scales": np.where(
            skeleton[:, None],
            skeleton_geometry["scales"],
            np.full(
                (int(keep.sum()), 3),
                voxel_size * 0.65,
                dtype=np.float32,
            ),
        ).astype(np.float32),
        "opacities": np.full(
            (int(keep.sum()), 1), 0.012, dtype=np.float32
        ),
        "quaternions": np.where(
            skeleton[:, None],
            skeleton_geometry["quaternions"],
            np.tile(
                np.asarray(
                    [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
                ),
                (int(keep.sum()), 1),
            ),
        ).astype(np.float32),
        "primitive_role": np.full(
            int(keep.sum()), 0, dtype=np.int8
        ),
        "layer_role": layer_role,
        "track_id": np.full(int(keep.sum()), -1, dtype=np.int64),
        "tree_instance_id": center_instances[keep].astype(np.int32),
        "initialization_source": np.full(
            int(keep.sum()), 0, dtype=np.int8
        ),
        "occupancy_probability": occupancy[keep].astype(np.float32),
        "position_covariance": position_covariance.astype(np.float32),
        "ray_depth_nll": (
            depth_nll_sum[keep]
            / np.maximum(depth_support[keep], 1)
        ).astype(np.float32),
        "free_space_violation_count": free_space_violations[keep],
        "unknown_view_count": unknown[keep],
        "support_camera_ids": support_ids[keep],
        "support_view_count": positive[keep],
        "support_sequence_count": sequence_count[keep],
        "maximum_baseline": maximum_baseline[keep],
        "maximum_triangulation_angle_degrees": maximum_angle[keep],
        "reprojection_error": np.asarray(
            [all_candidate_tracks[index]["error"] for index in nearest],
            dtype=np.float32,
        ),
        "track_linearity": skeleton_linearity,
        "static_skeleton_confidence": skeleton_geometry[
            "confidence"
        ].astype(np.float32),
        "selected_views": selected,
        "candidate_count": count,
        "dense_observation_ray_count": int(
            sum(len(record["camera_id"]) for record in dense_ray_records)
        ),
        "dense_hole_proposal_count": int(len(dense_proposals)),
        "dense_dynamic_birth_count": int(len(dense_dynamic_births)),
        "dense_dynamic_births": dense_dynamic_births,
        "dense_ray_budget": dense_ray_budget_audit,
        "candidate_bound_ray_budget": {
            "contract": (
                "per_camera_type_and_image_space_stratified_finite_basis"
            ),
            "maximum_per_view": int(maximum_bound_rays_per_view),
            "available_total": int(bound_ray_available_total),
            "stored_total": int(bound_ray_stored_total),
            "truncated_view_count": int(
                bound_ray_truncated_view_count
            ),
        },
        "tree_instance_count": int(center_instances.max()) + 1,
        "rigid_depth_view_count": len(rigid_depth_maps),
        "ray_evidence": ray_evidence,
    }


def _project(points, view):
    camera = points @ view["rotation"].T + view["translation"][None]
    depth = camera[:, 2]
    u = camera[:, 0] / np.maximum(depth, 1e-8) * view["fx"] + view["cx"]
    v = camera[:, 1] / np.maximum(depth, 1e-8) * view["fy"] + view["cy"]
    valid = (
        (depth > 0.05)
        & (u >= 0)
        & (u < view["width"])
        & (v >= 0)
        & (v < view["height"])
    )
    return u, v, depth, valid


def half_open_raster_coordinates(
    pixels: np.ndarray, image_sizes: np.ndarray
) -> np.ndarray:
    """Preserve a half-open raster domain after float32 serialization.

    Coordinates are validated before this helper in float64.  Casting an
    in-bounds value just below an integer endpoint can nevertheless round it
    to exactly that endpoint.  Repair only this sub-pixel representation
    effect and reject any materially invalid producer coordinate.
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    image_sizes = np.asarray(image_sizes, dtype=np.int32)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape [N, 2]")
    if image_sizes.shape != pixels.shape:
        raise ValueError("image_sizes must have shape [N, 2]")
    finite = np.isfinite(pixels).all(axis=1)
    gross_outside = (
        ~finite
        | (pixels < -1e-4).any(axis=1)
        | (
            pixels > image_sizes.astype(np.float32) + 1e-4
        ).any(axis=1)
    )
    if np.any(gross_outside):
        raise RuntimeError(
            "Foliage ray producer emitted coordinates outside its declared "
            "source raster"
        )
    upper = np.nextafter(
        image_sizes.astype(np.float32),
        np.full_like(pixels, -np.inf, dtype=np.float32),
    )
    return np.minimum(
        np.maximum(pixels, np.float32(0.0)),
        upper,
    )


def build_canopy_volume(
    tracks,
    images,
    cameras,
    mask_lookup,
    *,
    voxel_size=0.2,
    dilation_steps=2,
    maximum_voxels=250_000,
    selected_view_count=64,
    minimum_support_views=3,
    minimum_support_sequences=2,
    minimum_occupancy=0.6,
    minimum_baseline=0.75,
    minimum_triangulation_angle_degrees=1.5,
    minimum_camera_distance=0.75,
    maximum_projected_radius=128.0,
    seed=0,
):
    if not tracks:
        raise RuntimeError("No independent semantic SfM tree tracks were found")
    anchor_xyz = np.stack([point["xyz"] for point in tracks])
    anchor_rgb = np.stack([point["rgb"] for point in tracks])
    _, centers = _candidate_voxels(
        anchor_xyz, voxel_size, dilation_steps, maximum_voxels, seed
    )
    view_records = [
        _camera_record(image, cameras[image["camera_id"]], mask_lookup)
        for image in images.values()
    ]
    selected = greedy_diverse_views(
        view_records, limit=selected_view_count, minimum_center_distance=minimum_baseline
    )
    count = len(centers)
    support = np.zeros(count, dtype=np.int16)
    negative = np.zeros(count, dtype=np.int16)
    inverse_depth_sum = np.zeros(count, dtype=np.float64)
    inverse_depth_square_sum = np.zeros(count, dtype=np.float64)
    support_matrix = np.zeros((count, len(selected)), dtype=bool)
    nearest_camera_distance = np.full(count, np.inf, dtype=np.float32)
    largest_projected_radius = np.zeros(count, dtype=np.float32)
    for view_index, view in enumerate(selected):
        u, v, depth, valid = _project(centers, view)
        mask = view["tree_keep_mask"]
        safe_u = np.where(valid, u, 0.0)
        safe_v = np.where(valid, v, 0.0)
        rows = np.clip(np.round(safe_v / view["height"] * mask.shape[0]).astype(int), 0, mask.shape[0] - 1)
        cols = np.clip(np.round(safe_u / view["width"] * mask.shape[1]).astype(int), 0, mask.shape[1] - 1)
        tree = valid & (~mask[rows, cols])
        non_tree = valid & mask[rows, cols]
        support_matrix[:, view_index] = tree
        support += tree.astype(np.int16)
        negative += non_tree.astype(np.int16)
        inverse = np.zeros_like(depth)
        inverse[tree] = 1.0 / depth[tree]
        inverse_depth_sum += inverse
        inverse_depth_square_sum += inverse * inverse
        distance = np.linalg.norm(
            centers - np.asarray(view["camera_center"])[None], axis=1
        )
        nearest_camera_distance[valid] = np.minimum(
            nearest_camera_distance[valid], distance[valid]
        )
        radius = (
            max(float(view["fx"]), float(view["fy"]))
            * (float(voxel_size) * 0.75)
            / np.maximum(depth, 1e-8)
        )
        largest_projected_radius[valid] = np.maximum(
            largest_projected_radius[valid], radius[valid]
        )
    occupancy = (support + 1.0) / (support + negative + 2.0)
    valid_support = support >= minimum_support_views
    mean_inverse_depth = inverse_depth_sum / np.maximum(support, 1)
    variance_inverse_depth = (
        inverse_depth_square_sum / np.maximum(support, 1)
        - mean_inverse_depth * mean_inverse_depth
    ).clip(min=0)

    sequence_count = np.zeros(count, dtype=np.int16)
    maximum_baseline = np.zeros(count, dtype=np.float32)
    maximum_angle = np.zeros(count, dtype=np.float32)
    for sequence in sorted({view["sequence_id"] for view in selected}):
        sequence_columns = [
            index
            for index, view in enumerate(selected)
            if view["sequence_id"] == sequence
        ]
        sequence_count += support_matrix[:, sequence_columns].any(axis=1)
    for first in range(len(selected)):
        center_a = np.asarray(selected[first]["camera_center"])
        for second in range(first + 1, len(selected)):
            pair_valid = support_matrix[:, first] & support_matrix[:, second]
            indices = np.nonzero(pair_valid)[0]
            if not len(indices):
                continue
            center_b = np.asarray(selected[second]["camera_center"])
            baseline = float(np.linalg.norm(center_a - center_b))
            ray_a = centers[indices] - center_a[None]
            ray_b = centers[indices] - center_b[None]
            cosine = np.sum(ray_a * ray_b, axis=1) / np.maximum(
                np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1), 1e-8
            )
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            maximum_baseline[indices] = np.maximum(maximum_baseline[indices], baseline)
            maximum_angle[indices] = np.maximum(maximum_angle[indices], angle)
    selected_image_ids = np.asarray(
        [view["image_id"] for view in selected], dtype=np.int32
    )
    support_ids = np.where(
        support_matrix,
        selected_image_ids[None],
        np.int32(-1),
    )
    keep = (
        valid_support
        & (occupancy >= minimum_occupancy)
        & (sequence_count >= minimum_support_sequences)
        & (maximum_baseline >= minimum_baseline)
        & (maximum_angle >= minimum_triangulation_angle_degrees)
        & (nearest_camera_distance >= minimum_camera_distance)
        & (largest_projected_radius <= maximum_projected_radius)
    )
    if not np.any(keep):
        raise RuntimeError("No canopy voxel survived independent support gates")

    try:
        from scipy.spatial import cKDTree

        nearest = cKDTree(anchor_xyz).query(centers[keep], k=1)[1]
    except ImportError:
        nearest = np.asarray(
            [
                np.argmin(np.linalg.norm(anchor_xyz - center, axis=1))
                for center in centers[keep]
            ]
        )
    return {
        "centers": centers[keep].astype(np.float32),
        "colors": (anchor_rgb[nearest].astype(np.float32) / 255.0),
        "scales": np.full((int(keep.sum()), 3), voxel_size * 0.75, dtype=np.float32),
        "opacities": np.full((int(keep.sum()), 1), 0.02, dtype=np.float32),
        "quaternions": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (int(keep.sum()), 1),
        ),
        "occupancy_probability": occupancy[keep].astype(np.float32),
        "inverse_depth_mean": mean_inverse_depth[keep].astype(np.float32),
        "inverse_depth_variance": variance_inverse_depth[keep].astype(np.float32),
        "support_camera_ids": support_ids[keep],
        "support_view_count": support[keep],
        "support_sequence_count": sequence_count[keep],
        "maximum_baseline": maximum_baseline[keep],
        "maximum_triangulation_angle_degrees": maximum_angle[keep],
        "nearest_camera_distance": nearest_camera_distance[keep],
        "largest_projected_radius": largest_projected_radius[keep],
        "selected_views": selected,
        "candidate_count": count,
    }


def save_volume_state(path, volume, audit):
    payload = {
        "version": "independent_sfm_semantic_canopy_volume_v1",
        "audit": audit,
        **{
            key: torch.from_numpy(value)
            for key, value in volume.items()
            if isinstance(value, np.ndarray)
        },
    }
    torch.save(payload, path)
    return path
