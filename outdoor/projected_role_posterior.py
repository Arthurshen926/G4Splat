"""Project stable multiview roles into every fixed Cambridge camera.

The sparse MASt3R track archive contains observations for only the cameras
used to construct its correspondence graph.  A stable 3D track nevertheless
remains positive geometric evidence in every calibrated database camera in
which it projects.  This module materializes that evidence once, outside the
Gaussian lifecycle, and records only disagreements with the single-frame
object/sky/tree masks.

Geometry support and current-image visibility are deliberately separate.
The former may protect a persistent background surface behind an occluder;
the latter alone is allowed to restore RGB supervision in that camera.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from matcha.cambridge_masks import CambridgeMaskLookup
from outdoor.scene_contract import sha256_file


PROJECTED_RIGID_POSTERIOR_VERSION = (
    "cambridge-all-camera-projected-rigid-conflict-posterior-v2-"
    "tree-occlusion-geometry-with-exact-observation-visibility"
)
# ``LazyCamera`` uses this fixed far plane when the Cambridge camera contract
# does not provide one. A target beyond it can never receive renderer alpha or
# a depth gradient, so it is not a valid optimization observation.
PROJECTED_RIGID_MAX_CAMERA_DEPTH = 100.0


def _exact_track_ids_by_camera(
    archive: Any,
    *,
    track_id: np.ndarray,
    stable: np.ndarray,
) -> dict[str, np.ndarray]:
    """Index exact track observations without inventing camera visibility.

    Older/minimal archives may not contain the ragged observation table.  In
    that case the conservative answer is an empty map: projected geometry is
    still useful behind an occluder, but no projected row may claim that the
    current image actually observes it.
    """
    required = {
        "observation_offsets",
        "observation_camera_indices",
        "camera_names",
    }
    if not required.issubset(set(archive.files)):
        return {}
    offsets = np.asarray(archive["observation_offsets"], dtype=np.int64)
    observation_camera = np.asarray(
        archive["observation_camera_indices"], dtype=np.int64
    )
    camera_names = np.asarray(archive["camera_names"]).astype(str)
    if offsets.shape != (len(track_id) + 1,):
        raise RuntimeError("Track observation offsets do not align")
    observation_tracks = np.repeat(
        np.arange(len(track_id), dtype=np.int64), np.diff(offsets)
    )
    if len(observation_tracks) != len(observation_camera):
        raise RuntimeError("Track observation camera table does not align")
    if len(observation_camera) and (
        observation_camera.min() < 0
        or observation_camera.max() >= len(camera_names)
    ):
        raise RuntimeError("Track observation camera index is out of range")
    keep = stable[observation_tracks]
    result: dict[str, np.ndarray] = {}
    for camera_index in np.unique(observation_camera[keep]):
        camera_rows = keep & (observation_camera == camera_index)
        ids = np.unique(track_id[observation_tracks[camera_rows]])
        result[Path(camera_names[int(camera_index)]).stem] = ids
    return result


def stable_rigid_track_probability(archive: Any) -> np.ndarray:
    """Return a continuous, cross-sequence rigid stability posterior."""
    role = np.asarray(archive["role_probabilities"], dtype=np.float64)
    observations = np.asarray(
        archive["valid_observation_count"], dtype=np.float64
    )
    sequences = np.asarray(archive["sequence_count"], dtype=np.float64)
    reprojection = np.asarray(
        archive["reprojection_error"], dtype=np.float64
    )
    angle = np.asarray(
        archive["triangulation_angle_median"], dtype=np.float64
    )
    cycle = np.asarray(archive["cycle_consistency"], dtype=np.float64)
    cross_sequence = np.clip(sequences - 1.0, 0.0, 1.0)
    view_precision = 1.0 - np.exp(
        -np.maximum(observations - 1.0, 0.0) / 2.0
    )
    reprojection_precision = np.exp(
        -np.square(np.maximum(reprojection, 0.0) / 1.5)
    )
    baseline_precision = 1.0 - np.exp(
        -np.maximum(angle, 0.0) / 3.0
    )
    cycle_precision = np.clip((cycle - 0.50) / 0.35, 0.0, 1.0)
    stability = np.sqrt(
        np.clip(
            view_precision
            * reprojection_precision
            * baseline_precision
            * cycle_precision,
            0.0,
            1.0,
        )
    )
    return np.asarray(
        np.clip(role[:, 0], 0.0, 1.0)
        * cross_sequence
        * stability,
        dtype=np.float32,
    )


def _fit_image_affine(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a bounded robust per-channel affine appearance calibration."""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    scale = np.ones(3, dtype=np.float64)
    bias = np.zeros(3, dtype=np.float64)
    if len(source) < 32:
        return scale, bias
    weight = np.ones(len(source), dtype=np.float64)
    for _ in range(2):
        design = np.stack([source, np.ones_like(source)], axis=-1)
        root_weight = np.sqrt(np.maximum(weight, 1.0e-6))
        for channel in range(3):
            lhs = design[:, channel] * root_weight[:, None]
            rhs = target[:, channel] * root_weight
            solution = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            scale[channel] = np.clip(solution[0], 0.5, 1.5)
            bias[channel] = np.clip(solution[1], -0.25, 0.25)
        prediction = np.clip(source * scale + bias, 0.0, 1.0)
        error = np.mean(np.abs(prediction - target), axis=1)
        weight = np.exp(-np.square(error / 0.20))
    return scale, bias


def _load_target_rgb(
    rgb_root: Path,
    dataset: Path,
    image_name: str,
    size: tuple[int, int],
) -> np.ndarray:
    candidates = (
        Path(rgb_root) / image_name,
        Path(dataset) / "images" / image_name,
    )
    path = next((value for value in candidates if value.is_file()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if image.size != tuple(map(int, size)):
            image = image.resize(tuple(map(int, size)), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def build_projected_rigid_conflict_posterior(
    track_archive: Path,
    scene_contract: Path,
    dataset: Path,
    tree_mask_pickle: Path,
    output: Path,
    *,
    rgb_root: Path | None = None,
    appearance_sigma: float = 0.20,
) -> dict[str, Any]:
    """Persist a ragged all-camera posterior for mask-conflicting rigid rays."""
    track_archive = Path(track_archive).resolve()
    scene_contract = Path(scene_contract).resolve()
    dataset = Path(dataset).resolve()
    tree_mask_pickle = Path(tree_mask_pickle).resolve()
    output = Path(output).resolve()
    rgb_root = (
        Path(rgb_root).resolve()
        if rgb_root is not None
        else (dataset / "images").resolve()
    )
    if appearance_sigma <= 0:
        raise ValueError("appearance_sigma must be positive")
    scene = json.loads(scene_contract.read_text(encoding="utf-8"))
    records = list(scene["records"])
    lookup = CambridgeMaskLookup(
        dataset, tree_mask_pickle, mask_indices=[0, 1, 2, 3]
    )

    with np.load(track_archive, allow_pickle=False) as archive:
        required = {
            "xyz",
            "rgb",
            "track_id",
            "role_probabilities",
            "valid_observation_count",
            "sequence_count",
            "reprojection_error",
            "triangulation_angle_median",
            "cycle_consistency",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(
                "Projected rigid posterior lacks track fields: "
                + ", ".join(missing)
            )
        probability = stable_rigid_track_probability(archive)
        stable = probability > 0.0
        all_track_id = np.asarray(archive["track_id"], dtype=np.int64)
        exact_track_ids = _exact_track_ids_by_camera(
            archive, track_id=all_track_id, stable=stable
        )
        xyz = np.asarray(archive["xyz"], dtype=np.float64)[stable]
        rgb = np.asarray(archive["rgb"], dtype=np.float32)[stable] / 255.0
        track_id = all_track_id[stable]
        probability = probability[stable]

    camera_names: list[str] = []
    camera_sizes: list[tuple[int, int]] = []
    pixels_parts: list[np.ndarray] = []
    support_parts: list[np.ndarray] = []
    visible_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    track_parts: list[np.ndarray] = []
    offsets = [0]
    views_with_conflicts = 0
    tree_occlusion_count = 0
    tree_occlusion_exact_observation_count = 0
    tree_occlusion_geometry_mass = 0.0
    tree_occlusion_visible_mass = 0.0
    for record in records:
        image_name = str(record["image_name"])
        key = lookup.source_name_for(image_name)
        channels = lookup.masks[key]
        object_keep, sky_keep, distortion_keep, tree_keep = [
            value.detach().to(device="cpu", dtype=bool).numpy()
            for value in channels[:4]
        ]
        height, width = object_keep.shape
        target_rgb = _load_target_rgb(
            rgb_root, dataset, image_name, (width, height)
        )
        transform = np.asarray(
            record["T_world_to_camera"], dtype=np.float64
        )
        camera_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        depth = camera_xyz[:, 2]
        camera = record["camera"]
        scale_x = width / float(camera["width"])
        scale_y = height / float(camera["height"])
        focal_x = float(camera["fx"]) * scale_x
        focal_y = float(camera["fy"]) * scale_y
        center_x = float(camera["cx"]) * scale_x
        center_y = float(camera["cy"]) * scale_y
        columns = np.rint(
            focal_x * camera_xyz[:, 0] / np.maximum(depth, 1.0e-8)
            + center_x
        ).astype(np.int64)
        rows = np.rint(
            focal_y * camera_xyz[:, 1] / np.maximum(depth, 1.0e-8)
            + center_y
        ).astype(np.int64)
        valid = (
            (depth > 0.05)
            & (depth < PROJECTED_RIGID_MAX_CAMERA_DEPTH)
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        valid_rows = np.flatnonzero(valid)
        if len(valid_rows):
            linear = rows[valid_rows] * width + columns[valid_rows]
            order = np.lexsort((depth[valid_rows], linear))
            ordered_linear = linear[order]
            nearest = np.r_[
                True, ordered_linear[1:] != ordered_linear[:-1]
            ]
            chosen = valid_rows[order[nearest]]
        else:
            chosen = np.empty(0, dtype=np.int64)

        chosen_rows = rows[chosen]
        chosen_columns = columns[chosen]
        known_rigid = (
            object_keep[chosen_rows, chosen_columns]
            & sky_keep[chosen_rows, chosen_columns]
            & distortion_keep[chosen_rows, chosen_columns]
            & tree_keep[chosen_rows, chosen_columns]
        )
        scale, bias = _fit_image_affine(
            rgb[chosen][known_rigid],
            target_rgb[chosen_rows[known_rigid], chosen_columns[known_rigid]],
        )
        corrected_rgb = np.clip(rgb[chosen] * scale + bias, 0.0, 1.0)
        appearance_error = np.mean(
            np.abs(
                corrected_rgb
                - target_rgb[chosen_rows, chosen_columns]
            ),
            axis=1,
        )
        visible_probability = probability[chosen] * np.exp(
            -np.square(appearance_error / float(appearance_sigma))
        )
        object_or_sky_conflict = (
            (~object_keep[chosen_rows, chosen_columns])
            | (~sky_keep[chosen_rows, chosen_columns])
        ) & distortion_keep[chosen_rows, chosen_columns]
        # A cross-sequence rigid track projected behind a tree is positive
        # evidence that persistent background geometry exists, but it is not
        # evidence that the current camera sees that geometry.  Only a real
        # observation of this exact track in this exact camera can grant RGB
        # visibility through a (possibly coarse) tree mask.
        tree_occlusion_conflict = (
            object_keep[chosen_rows, chosen_columns]
            & sky_keep[chosen_rows, chosen_columns]
            & (~tree_keep[chosen_rows, chosen_columns])
            & distortion_keep[chosen_rows, chosen_columns]
        )
        camera_exact_tracks = exact_track_ids.get(
            Path(image_name).stem, np.empty(0, dtype=np.int64)
        )
        exact_observation = np.isin(
            track_id[chosen], camera_exact_tracks, assume_unique=False
        )
        visible_probability[tree_occlusion_conflict] *= exact_observation[
            tree_occlusion_conflict
        ]
        conflict = object_or_sky_conflict | tree_occlusion_conflict
        tree_rows = tree_occlusion_conflict & conflict
        tree_occlusion_count += int(tree_rows.sum())
        tree_occlusion_exact_observation_count += int(
            (tree_rows & exact_observation).sum()
        )
        tree_occlusion_geometry_mass += float(
            probability[chosen][tree_rows].sum(dtype=np.float64)
        )
        tree_occlusion_visible_mass += float(
            visible_probability[tree_rows].sum(dtype=np.float64)
        )
        chosen = chosen[conflict]
        chosen_rows = chosen_rows[conflict]
        chosen_columns = chosen_columns[conflict]
        visible_probability = visible_probability[conflict]
        camera_names.append(image_name)
        camera_sizes.append((width, height))
        pixels_parts.append(
            np.column_stack([chosen_columns, chosen_rows]).astype(np.uint16)
        )
        support_parts.append(probability[chosen].astype(np.float16))
        visible_parts.append(visible_probability.astype(np.float16))
        depth_parts.append(depth[chosen].astype(np.float32))
        track_parts.append(track_id[chosen].astype(np.int64))
        offsets.append(offsets[-1] + len(chosen))
        views_with_conflicts += int(len(chosen) > 0)

    payload = {
        "schema_version": np.asarray(PROJECTED_RIGID_POSTERIOR_VERSION),
        "camera_names": np.asarray(camera_names),
        "camera_image_sizes": np.asarray(camera_sizes, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "pixels": np.concatenate(pixels_parts, axis=0),
        "geometry_support_probability": np.concatenate(
            support_parts, axis=0
        ),
        "visible_observation_probability": np.concatenate(
            visible_parts, axis=0
        ),
        "camera_depth": np.concatenate(depth_parts, axis=0),
        "source_track_id": np.concatenate(track_parts, axis=0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = {
        "schema_version": PROJECTED_RIGID_POSTERIOR_VERSION,
        "output": str(output),
        "track_archive": str(track_archive),
        "scene_contract": str(scene_contract),
        "tree_mask_pickle": str(tree_mask_pickle),
        "rgb_root": str(rgb_root),
        "input_hashes": {
            "track_archive": sha256_file(track_archive),
            "scene_contract": sha256_file(scene_contract),
            "tree_mask_pickle": sha256_file(tree_mask_pickle),
        },
        "producer_implementation_sha256": sha256_file(Path(__file__)),
        "camera_count": len(camera_names),
        "views_with_mask_conflicts": views_with_conflicts,
        "stable_rigid_track_count": int(len(xyz)),
        "projected_conflict_observation_count": int(offsets[-1]),
        "tree_occlusion_observation_count": int(tree_occlusion_count),
        "tree_occlusion_exact_observation_count": int(
            tree_occlusion_exact_observation_count
        ),
        "tree_occlusion_geometry_support_mass": float(
            tree_occlusion_geometry_mass
        ),
        "tree_occlusion_visible_observation_mass": float(
            tree_occlusion_visible_mass
        ),
        "geometry_support_mass": float(
            np.asarray(payload["geometry_support_probability"], dtype=np.float64).sum()
        ),
        "visible_observation_mass": float(
            np.asarray(payload["visible_observation_probability"], dtype=np.float64).sum()
        ),
        "appearance_sigma": float(appearance_sigma),
        "camera_depth_interval": [
            0.05,
            float(PROJECTED_RIGID_MAX_CAMERA_DEPTH),
        ],
        "contract": (
            "cross_sequence_track_geometry_projects_to_all_fixed_cameras__"
            "nearest_stable_depth_per_pixel__object_sky_or_tree_conflicts__"
            "geometry_support_separate_from_current_rgb_visibility__"
            "tree_occluded_geometry_has_zero_current_rgb_visibility_unless_"
            "the_exact_track_is_observed_in_the_current_camera"
        ),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
