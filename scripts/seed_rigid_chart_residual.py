#!/usr/bin/env python
"""Append uncovered Chart/SfM points as trainable rigid 2DGS residual seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


SH_C0 = 0.28209479177387814


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fields(vertices, prefix):
    return sorted(
        (name for name in vertices.dtype.names if name.startswith(prefix)),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def _voxel_representatives(points, voxel_size):
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-structural-ply", type=Path, required=True)
    parser.add_argument("--chart-points-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--minimum-uncovered-distance", type=float, default=0.02)
    parser.add_argument("--base-coverage-scale", type=float, default=0.75)
    parser.add_argument("--maximum-base-coverage", type=float, default=0.12)
    parser.add_argument("--normal-neighbors", type=int, default=12)
    parser.add_argument("--maximum-neighbor-radius", type=float, default=0.20)
    parser.add_argument("--scale-multiplier", type=float, default=1.5)
    parser.add_argument("--minimum-scale", type=float, default=0.004)
    parser.add_argument("--maximum-scale", type=float, default=0.08)
    parser.add_argument("--initial-opacity", type=float, default=0.02)
    parser.add_argument("--maximum-seeds", type=int, default=120_000)
    args = parser.parse_args()
    if args.output_sidecar is None:
        args.output_sidecar = args.output_ply.with_suffix(
            ".residual_seed.pth.npz"
        )
    if args.normal_neighbors < 4:
        parser.error("--normal-neighbors must be at least four")
    if not 0 < args.initial_opacity < 1:
        parser.error("--initial-opacity must be in (0,1)")
    return args


def main():
    args = _parse_args()
    base_path = args.base_structural_ply.resolve()
    chart_path = args.chart_points_ply.resolve()
    output_path = args.output_ply.resolve()
    sidecar_path = args.output_sidecar.resolve()
    for path in (base_path, chart_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output_path, sidecar_path):
        if path.exists():
            raise FileExistsError(path)

    base_ply = PlyData.read(str(base_path))
    base = np.asarray(base_ply["vertex"].data)
    chart = np.asarray(PlyData.read(str(chart_path))["vertex"].data)
    required_chart = {"x", "y", "z", "red", "green", "blue"}
    if not required_chart.issubset(chart.dtype.names):
        raise RuntimeError("Chart point cloud must contain XYZ and RGB")
    points = np.stack([chart[name] for name in ("x", "y", "z")], axis=1)
    colors = np.stack(
        [chart[name] for name in ("red", "green", "blue")], axis=1
    ).astype(np.float64) / 255.0
    finite = np.isfinite(points).all(axis=1)
    source_indices = np.flatnonzero(finite)
    points = points[finite].astype(np.float64)
    colors = colors[finite]
    representatives = _voxel_representatives(points, args.voxel_size)
    points = points[representatives]
    colors = colors[representatives]
    source_indices = source_indices[representatives]

    base_xyz = np.stack(
        [base[name] for name in ("x", "y", "z")], axis=1
    ).astype(np.float64)
    base_scales = np.exp(
        np.stack([base["scale_0"], base["scale_1"]], axis=1)
        .astype(np.float64)
        .clip(-20, 20)
    ).max(axis=1)
    base_tree = cKDTree(base_xyz)
    distance, nearest = base_tree.query(points, k=1, workers=-1)
    coverage = np.maximum(
        float(args.minimum_uncovered_distance),
        np.minimum(
            float(args.maximum_base_coverage),
            float(args.base_coverage_scale) * base_scales[nearest],
        ),
    )
    uncovered = distance > coverage
    points = points[uncovered]
    colors = colors[uncovered]
    source_indices = source_indices[uncovered]
    distance = distance[uncovered]
    coverage = coverage[uncovered]

    residual_tree = cKDTree(points)
    neighbor_distance, neighbor_index = residual_tree.query(
        points, k=args.normal_neighbors, workers=-1
    )
    locally_supported = (
        neighbor_distance[:, -1] <= float(args.maximum_neighbor_radius)
    )
    points = points[locally_supported]
    colors = colors[locally_supported]
    source_indices = source_indices[locally_supported]
    distance = distance[locally_supported]
    coverage = coverage[locally_supported]
    neighbor_distance = neighbor_distance[locally_supported]
    neighbor_index = neighbor_index[locally_supported]
    if len(points) > args.maximum_seeds:
        # Prefer confidently supported holes just beyond existing coverage;
        # extreme distances are more likely coordinate/outlier failures.
        score = distance / np.maximum(coverage, 1e-8)
        selected = np.argsort(score)[: args.maximum_seeds]
        points = points[selected]
        colors = colors[selected]
        source_indices = source_indices[selected]
        neighbor_distance = neighbor_distance[selected]
        neighbor_index = neighbor_index[selected]

    # Indices refer to the pre-filter residual point array. Rebuild the local
    # tree after caps so every PCA neighbourhood has a valid compact index.
    residual_tree = cKDTree(points)
    neighbor_distance, neighbor_index = residual_tree.query(
        points, k=min(args.normal_neighbors, len(points)), workers=-1
    )
    neighborhoods = points[neighbor_index]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(
        centered.shape[1] - 1, 1
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, :, 0]
    tangent_x = eigenvectors[:, :, 2]
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y, axis=1, keepdims=True).clip(
        1e-8
    )
    tangent_x = np.cross(tangent_y, normal)
    matrices = np.stack([tangent_x, tangent_y, normal], axis=2)
    quaternion_xyzw = Rotation.from_matrix(matrices).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    scales = np.sqrt(
        np.maximum(eigenvalues[:, [2, 1]], args.minimum_scale**2)
    ) * float(args.scale_multiplier)
    scales = np.clip(
        scales, float(args.minimum_scale), float(args.maximum_scale)
    )

    seeds = np.repeat(base[:1], len(points), axis=0)
    for axis, name in enumerate(("x", "y", "z")):
        seeds[name] = points[:, axis].astype(seeds[name].dtype)
    seeds["scale_0"] = np.log(scales[:, 0]).astype(
        seeds["scale_0"].dtype
    )
    seeds["scale_1"] = np.log(scales[:, 1]).astype(
        seeds["scale_1"].dtype
    )
    for column, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        seeds[name] = quaternion_wxyz[:, column].astype(seeds[name].dtype)
    logit = np.log(args.initial_opacity / (1 - args.initial_opacity))
    seeds["opacity"] = np.asarray(logit, dtype=seeds["opacity"].dtype)
    dc_names = _fields(base, "f_dc_")
    rest_names = _fields(base, "f_rest_")
    if len(dc_names) != 3:
        raise RuntimeError("Structural PLY must have exactly three DC fields")
    sh_dc = (colors - 0.5) / SH_C0
    for channel, name in enumerate(dc_names):
        seeds[name] = sh_dc[:, channel].astype(seeds[name].dtype)
    for name in rest_names:
        seeds[name] = 0
    if "primitive_class" in base.dtype.names:
        seeds["primitive_class"] = 0
    if "source_type" in base.dtype.names:
        seeds["source_type"] = 0
    if "geometry_confidence" in base.dtype.names:
        planarity = 1.0 - eigenvalues[:, 0] / np.maximum(
            eigenvalues.sum(axis=1), 1e-8
        )
        seeds["geometry_confidence"] = np.clip(
            planarity, 0, 1
        ).astype(seeds["geometry_confidence"].dtype)
    if "protected_flag" in base.dtype.names:
        seeds["protected_flag"] = 0
    if "block_id" in base.dtype.names:
        seeds["block_id"] = -1
    if "mip_filter" in base.dtype.names:
        seeds["mip_filter"] = 0

    output = np.concatenate([base.copy(), seeds])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(output, "vertex")],
        text=False,
        byte_order=base_ply.byte_order,
    ).write(str(output_path))
    np.savez_compressed(
        sidecar_path,
        version=np.asarray("rigid_chart_residual_seed_v1"),
        base_count=np.asarray(len(base), dtype=np.int64),
        source_point_index=source_indices.astype(np.int64),
        local_eigenvalues=eigenvalues.astype(np.float32),
        chart_point_path=np.asarray(str(chart_path)),
    )
    manifest = {
        "version": "rigid_chart_residual_seed_v1",
        "base_structural_ply": str(base_path),
        "base_structural_ply_sha256": _sha256(base_path),
        "chart_points_ply": str(chart_path),
        "chart_points_ply_sha256": _sha256(chart_path),
        "output_ply": str(output_path),
        "output_ply_sha256": _sha256(output_path),
        "input_base_count": int(len(base)),
        "input_chart_point_count": int(len(chart)),
        "residual_seed_count": int(len(seeds)),
        "output_count": int(len(output)),
        "policy": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (int, float, bool))
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
