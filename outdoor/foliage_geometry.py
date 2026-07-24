"""Independent SfM-track and semantic visual-hull foliage geometry."""

from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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


def cluster_tree_instances(
    xyz,
    *,
    connection_radius=1.5,
    minimum_tracks=8,
):
    """Cluster real tree tracks in world space without crossing tree instances."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    parent = np.arange(len(xyz), dtype=np.int32)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in cKDTree(xyz).query_pairs(float(connection_radius)):
        left, right = root(first), root(second)
        if left != right:
            parent[right] = left
    roots = np.asarray([root(index) for index in range(len(xyz))])
    unique, counts = np.unique(roots, return_counts=True)
    accepted = {
        int(value): offset
        for offset, value in enumerate(unique[counts >= int(minimum_tracks)])
    }
    labels = np.asarray(
        [accepted.get(int(value), -1) for value in roots], dtype=np.int32
    )
    # Isolated but valid tracks remain individual evidence anchors rather than
    # being silently assigned to a nearby, potentially different tree.
    next_label = len(accepted)
    for index in np.nonzero(labels < 0)[0]:
        labels[index] = next_label
        next_label += 1
    return labels


def _component_instance_map(
    view,
    anchor_xyz,
    anchor_instances,
    observed_track_mask,
):
    from scipy import ndimage

    tree = ~view["tree_keep_mask"]
    components, component_count = ndimage.label(tree)
    component_instance = np.full(component_count + 1, -1, dtype=np.int32)
    if not np.any(observed_track_mask):
        return components, component_instance
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
    votes = defaultdict(lambda: defaultdict(int))
    for keep, label, instance in zip(valid, labels, instances):
        if keep and label > 0:
            votes[int(label)][int(instance)] += 1
    for label, row in votes.items():
        winner, count = max(row.items(), key=lambda item: item[1])
        total = sum(row.values())
        if count >= 2 and count / max(total, 1) >= 0.6:
            component_instance[label] = winner
    return components, component_instance


def build_instance_aware_canopy_volume(
    tracks,
    images,
    cameras,
    mask_lookup,
    *,
    rigid_depth_maps=None,
    selected_views=None,
    voxel_size=0.12,
    dilation_steps=3,
    maximum_voxels=400_000,
    selected_view_count=64,
    minimum_support_views=3,
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
    seed=0,
):
    """Build porous, instance-aware crown seeds from ray/depth observations.

    Positive evidence requires the correct tree-mask component *and* a local
    SfM-track depth posterior. A protected-structure z-buffer converts hidden
    projections to unknown instead of negative evidence.
    """
    from scipy.spatial import cKDTree

    if not tracks:
        raise RuntimeError("No independent semantic SfM tree tracks were found")
    rigid_depth_maps = rigid_depth_maps or {}
    anchor_xyz = np.stack([point["xyz"] for point in tracks])
    anchor_rgb = np.stack([point["rgb"] for point in tracks])
    anchor_instances = cluster_tree_instances(
        anchor_xyz,
        connection_radius=instance_connection_radius,
        minimum_tracks=instance_minimum_tracks,
    )
    _, centers = _candidate_voxels(
        anchor_xyz, voxel_size, dilation_steps, maximum_voxels, seed
    )
    nearest_anchor = cKDTree(anchor_xyz).query(centers, k=1)[1]
    center_instances = anchor_instances[nearest_anchor]
    if selected_views is None:
        view_records = [
            _camera_record(image, cameras[image["camera_id"]], mask_lookup)
            for image in images.values()
        ]
        selected = greedy_diverse_views(
            view_records,
            limit=selected_view_count,
            minimum_center_distance=minimum_baseline,
        )
    else:
        selected = list(selected_views)
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

    track_images = [
        {int(value) for value in point.get("tree_image_ids", ())}
        for point in tracks
    ]
    for view_index, view in enumerate(selected):
        image_id = int(view["image_id"])
        observed = np.asarray(
            [image_id in values for values in track_images], dtype=bool
        )
        components, component_instance = _component_instance_map(
            view, anchor_xyz, anchor_instances, observed
        )
        u, v, depth, valid = _project(centers, view)
        mask_height, mask_width = components.shape
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
        component = components[rows, cols]
        matched_instance = (
            component_instance[component] == center_instances
        ) & (component > 0)

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
                anchor_xyz[observed], view
            )
            observed_instances = anchor_instances[observed]
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
            valid
            & ~occluded
            & (
                ((component == 0) & ~posterior_found)
                | free
            )
        )
        ambiguous = valid & ~(supported | confirmed_negative)
        positive += supported.astype(np.int16)
        depth_support += compatible.astype(np.int16)
        negative += confirmed_negative.astype(np.int16)
        unknown += (ambiguous | occluded).astype(np.int16)
        free_space_violations += free.astype(np.int16)
        support_matrix[:, view_index] = supported
        likelihood_indices = np.nonzero(posterior_found)[0]
        depth_nll_sum[likelihood_indices] += (
            residual[likelihood_indices]
            / np.maximum(posterior_sigma[likelihood_indices], 1e-6)
        ) ** 2
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
    maximum_baseline = np.zeros(count, dtype=np.float32)
    maximum_angle = np.zeros(count, dtype=np.float32)
    for sequence in sorted({view["sequence_id"] for view in selected}):
        columns = [
            index
            for index, view in enumerate(selected)
            if view["sequence_id"] == sequence
        ]
        sequence_count += support_matrix[:, columns].any(axis=1)
    for first in range(len(selected)):
        center_a = np.asarray(selected[first]["camera_center"])
        for second in range(first + 1, len(selected)):
            pair = support_matrix[:, first] & support_matrix[:, second]
            indices = np.nonzero(pair)[0]
            if not len(indices):
                continue
            center_b = np.asarray(selected[second]["camera_center"])
            baseline = float(np.linalg.norm(center_a - center_b))
            ray_a = centers[indices] - center_a[None]
            ray_b = centers[indices] - center_b[None]
            cosine = np.sum(ray_a * ray_b, axis=1) / np.maximum(
                np.linalg.norm(ray_a, axis=1)
                * np.linalg.norm(ray_b, axis=1),
                1e-8,
            )
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            maximum_baseline[indices] = np.maximum(
                maximum_baseline[indices], baseline
            )
            maximum_angle[indices] = np.maximum(
                maximum_angle[indices], angle
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
        raise RuntimeError(
            "No instance-aware ray/depth-supported canopy voxel survived"
        )
    accepted_information = information[keep]
    weak = np.linalg.det(accepted_information) <= 1e-12
    accepted_information[weak] += (
        np.eye(3)[None] / max(voxel_size * voxel_size, 1e-6)
    )
    position_covariance = np.linalg.inv(accepted_information)
    nearest = nearest_anchor[keep]
    return {
        "centers": centers[keep].astype(np.float32),
        "colors": anchor_rgb[nearest].astype(np.float32) / 255.0,
        "scales": np.full(
            (int(keep.sum()), 3), voxel_size * 0.65, dtype=np.float32
        ),
        "opacities": np.full(
            (int(keep.sum()), 1), 0.012, dtype=np.float32
        ),
        "quaternions": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (int(keep.sum()), 1),
        ),
        "primitive_role": np.full(
            int(keep.sum()), 0, dtype=np.int8
        ),
        "layer_role": np.full(int(keep.sum()), 0, dtype=np.int8),
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
            [tracks[index]["error"] for index in nearest],
            dtype=np.float32,
        ),
        "track_linearity": np.zeros(int(keep.sum()), dtype=np.float32),
        "selected_views": selected,
        "candidate_count": count,
        "tree_instance_count": int(center_instances.max()) + 1,
        "rigid_depth_view_count": len(rigid_depth_maps),
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
