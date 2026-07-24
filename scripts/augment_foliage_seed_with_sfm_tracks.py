#!/usr/bin/env python
"""Add compact real SfM tree-track 3DGS to a canopy visual-hull seed state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-state", type=Path, required=True)
    parser.add_argument("--semantic-tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-scale", type=float, default=0.012)
    parser.add_argument("--maximum-cross-scale", type=float, default=0.045)
    parser.add_argument("--maximum-long-scale", type=float, default=0.12)
    parser.add_argument("--opacity", type=float, default=0.035)
    return parser.parse_args()


def _track_frames(xyz, minimum, maximum_cross, maximum_long):
    count = len(xyz)
    neighbor_count = min(10, count)
    distances, neighbors = cKDTree(xyz).query(xyz, k=neighbor_count)
    local = xyz[neighbors] - xyz[:, None]
    covariance = np.einsum("nki,nkj->nij", local, local) / max(
        neighbor_count - 1, 1
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, axis=1)[:, ::-1]
    eigenvalues = np.take_along_axis(eigenvalues, order, axis=1)
    eigenvectors = np.take_along_axis(
        eigenvectors, order[:, None, :], axis=2
    )
    # Rows are local Gaussian axes under the rasterizer's R^T S^2 R
    # covariance convention.
    frames = np.transpose(eigenvectors, (0, 2, 1))
    negative = np.linalg.det(frames) < 0
    frames[negative, 2] *= -1
    xyzw = Rotation.from_matrix(frames).as_quat().astype(np.float32)
    quaternion = np.column_stack(
        [xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]]
    )

    nearest = distances[:, 1].clip(minimum, maximum_cross)
    spread = np.sqrt(np.maximum(eigenvalues, 1e-10))
    linearity = spread[:, 0] / np.maximum(spread[:, 1], 1e-6)
    long_axis = np.where(
        linearity >= 2.0,
        np.maximum(spread[:, 0], nearest),
        nearest,
    ).clip(minimum, maximum_long)
    cross_a = np.maximum(spread[:, 1] * 0.45, nearest * 0.5).clip(
        minimum, maximum_cross
    )
    cross_b = np.maximum(spread[:, 2] * 0.45, nearest * 0.4).clip(
        minimum, maximum_cross
    )
    scales = np.column_stack([long_axis, cross_a, cross_b]).astype(np.float32)
    return scales, quaternion, linearity.astype(np.float32)


def main():
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        state = torch.load(
            args.volume_state.resolve(), map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(args.volume_state.resolve(), map_location="cpu")
    if state.get("version") != "independent_sfm_semantic_canopy_volume_v1":
        raise RuntimeError("Unsupported canopy volume state")
    tracks = np.load(args.semantic_tracks.resolve())
    xyz = tracks["xyz"].astype(np.float32)
    rgb = tracks["rgb"].astype(np.float32) / 255.0
    scales, quaternions, linearity = _track_frames(
        xyz,
        args.minimum_scale,
        args.maximum_cross_scale,
        args.maximum_long_scale,
    )
    base_count = len(state["centers"])
    track_count = len(xyz)
    nearest_volume = cKDTree(state["centers"].numpy()).query(xyz, k=1)[1]
    nearest_volume = torch.from_numpy(nearest_volume).long()

    payload = dict(state)
    payload["centers"] = torch.cat(
        [state["centers"], torch.from_numpy(xyz)], dim=0
    )
    payload["colors"] = torch.cat(
        [state["colors"], torch.from_numpy(rgb)], dim=0
    )
    payload["scales"] = torch.cat(
        [state["scales"], torch.from_numpy(scales)], dim=0
    )
    payload["quaternions"] = torch.cat(
        [state["quaternions"], torch.from_numpy(quaternions)], dim=0
    )
    payload["opacities"] = torch.cat(
        [
            state["opacities"],
            torch.full((track_count, 1), float(args.opacity)),
        ],
        dim=0,
    )
    # Posterior/risk metadata remains aligned by copying the closest accepted
    # visual-hull voxel, then replacing support counts with the track's direct
    # observations where those are stronger.
    for key, value in list(state.items()):
        if not isinstance(value, torch.Tensor) or len(value) != base_count:
            continue
        if key in {
            "centers",
            "colors",
            "scales",
            "quaternions",
            "opacities",
        }:
            continue
        payload[key] = torch.cat([value, value[nearest_volume]], dim=0)
    payload["support_view_count"][base_count:] = torch.from_numpy(
        tracks["tree_observation_count"]
    ).to(payload["support_view_count"].dtype)
    payload["support_sequence_count"][base_count:] = torch.from_numpy(
        tracks["tree_sequence_count"]
    ).to(payload["support_sequence_count"].dtype)
    payload["primitive_role"] = torch.cat(
        [
            torch.zeros(base_count, dtype=torch.int8),
            torch.ones(track_count, dtype=torch.int8),
        ]
    )
    payload["track_linearity"] = torch.cat(
        [
            torch.zeros(base_count, dtype=torch.float32),
            torch.from_numpy(linearity),
        ]
    )
    payload["audit"] = {
        **state.get("audit", {}),
        "high_frequency_track_augmentation": {
            "source": str(args.semantic_tracks.resolve()),
            "track_count": track_count,
            "legacy_surfel_xyz_used": False,
            "minimum_scale": args.minimum_scale,
            "maximum_cross_scale": args.maximum_cross_scale,
            "maximum_long_scale": args.maximum_long_scale,
            "opacity": args.opacity,
            "geometry": (
                "local PCA ellipsoids centered only at multi-sequence real "
                "SfM tree tracks"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "visual_hull_count": base_count,
        "sfm_track_count": track_count,
        "total_count": len(payload["centers"]),
        "scale_min": scales.min(axis=0).tolist(),
        "scale_median": np.median(scales, axis=0).tolist(),
        "scale_max": scales.max(axis=0).tolist(),
        "linearity_median": float(np.median(linearity)),
        "linearity_p95": float(np.quantile(linearity, 0.95)),
        "initialization_from_legacy_surfel_xyz": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
