#!/usr/bin/env python3
"""Export non-destructive clean variants of a Gaussian PLY.

The native output preserves every remaining 2DGS field and is suitable for
rendering.  The optional visualization PLY converts SH DC color to ordinary
RGB and adds opacity/scale/source-index scalar fields for point-cloud viewers
that otherwise display every Gaussian center with equal weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    positive = value >= 0
    output = np.empty_like(value)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _required_fields(vertices: np.ndarray) -> None:
    names = set(vertices.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"Gaussian PLY is missing fields: {missing}")


def clean_vertices(
    vertices: np.ndarray,
    *,
    opacity_min: float,
    structural_class: int = 0,
    drop_nonstructural: bool = False,
    isolation_distance: float | None = None,
    isolation_scale_ratio: float = 4.0,
    isolation_opacity_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return retained vertices, retained source indices and an audit."""
    _required_fields(vertices)
    if not 0.0 <= opacity_min < 1.0:
        raise ValueError("opacity_min must lie in [0, 1)")
    if isolation_distance is not None and isolation_distance <= 0:
        raise ValueError("isolation_distance must be positive")
    if isolation_scale_ratio <= 0:
        raise ValueError("isolation_scale_ratio must be positive")
    if not 0.0 <= isolation_opacity_max <= 1.0:
        raise ValueError("isolation_opacity_max must lie in [0, 1]")

    count = len(vertices)
    xyz = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float64, copy=False)
    opacity = _sigmoid(vertices["opacity"])
    maximum_scale = np.exp(
        np.maximum(vertices["scale_0"], vertices["scale_1"]).astype(
            np.float64, copy=False
        )
    )
    finite = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(opacity)
        & np.isfinite(maximum_scale)
    )
    keep = finite.copy()
    reason_counts = {
        "nonfinite": int((~finite).sum()),
        "nonstructural": 0,
        "opacity": 0,
        "isolated": 0,
    }

    if drop_nonstructural:
        if "primitive_class" not in (vertices.dtype.names or ()):
            raise RuntimeError(
                "--drop-nonstructural requires primitive_class metadata"
            )
        structural = (
            np.asarray(vertices["primitive_class"]).reshape(-1).astype(np.int64)
            == int(structural_class)
        )
        reason_counts["nonstructural"] = int((keep & ~structural).sum())
        keep &= structural

    opacity_keep = opacity >= float(opacity_min)
    reason_counts["opacity"] = int((keep & ~opacity_keep).sum())
    keep &= opacity_keep

    nearest_distance = None
    normalized_isolation = None
    if isolation_distance is not None:
        from scipy.spatial import cKDTree

        finite_indices = np.flatnonzero(finite)
        if len(finite_indices) < 2:
            raise RuntimeError("At least two finite Gaussian centers are required")
        distances, _ = cKDTree(xyz[finite_indices]).query(
            xyz[finite_indices], k=2, workers=-1
        )
        nearest_distance = np.full(count, np.inf, dtype=np.float64)
        nearest_distance[finite_indices] = distances[:, 1]
        normalized_isolation = nearest_distance / np.maximum(
            maximum_scale, 5e-3
        )
        isolated = (
            (nearest_distance > float(isolation_distance))
            & (normalized_isolation > float(isolation_scale_ratio))
            & (opacity <= float(isolation_opacity_max))
        )
        reason_counts["isolated"] = int((keep & isolated).sum())
        keep &= ~isolated

    retained_indices = np.flatnonzero(keep)
    removed = ~keep
    audit = {
        "input_count": int(count),
        "retained_count": int(keep.sum()),
        "removed_count": int(removed.sum()),
        "retained_fraction": float(keep.mean()),
        "parameters": {
            "opacity_min": float(opacity_min),
            "structural_class": int(structural_class),
            "drop_nonstructural": bool(drop_nonstructural),
            "isolation_distance": (
                None
                if isolation_distance is None
                else float(isolation_distance)
            ),
            "isolation_scale_ratio": float(isolation_scale_ratio),
            "isolation_opacity_max": float(isolation_opacity_max),
        },
        "exclusive_removal_reason_counts": reason_counts,
        "retained_opacity_quantiles": (
            np.quantile(opacity[keep], [0.0, 0.01, 0.5, 0.99, 1.0]).tolist()
            if keep.any()
            else []
        ),
        "retained_maximum_scale_quantiles": (
            np.quantile(
                maximum_scale[keep], [0.0, 0.01, 0.5, 0.99, 1.0]
            ).tolist()
            if keep.any()
            else []
        ),
    }
    if nearest_distance is not None and normalized_isolation is not None:
        audit["retained_nearest_distance_quantiles"] = np.quantile(
            nearest_distance[keep], [0.0, 0.5, 0.9, 0.99, 1.0]
        ).tolist()
        audit["retained_normalized_isolation_quantiles"] = np.quantile(
            normalized_isolation[keep], [0.0, 0.5, 0.9, 0.99, 1.0]
        ).tolist()
    if "primitive_class" in (vertices.dtype.names or ()):
        classes, class_counts = np.unique(
            np.asarray(vertices["primitive_class"])[keep].astype(np.int64),
            return_counts=True,
        )
        audit["retained_primitive_class_counts"] = {
            str(int(key)): int(value)
            for key, value in zip(classes, class_counts)
        }
    return vertices[keep].copy(), retained_indices, audit


def visualization_vertices(
    vertices: np.ndarray, source_indices: np.ndarray
) -> np.ndarray:
    """Convert native Gaussian rows into an ordinary colored point cloud."""
    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("opacity", "<f4"),
        ("maximum_scale", "<f4"),
        ("source_index", "<i4"),
        ("primitive_class", "<i2"),
    ]
    output = np.empty(len(vertices), dtype=dtype)
    for name in ("x", "y", "z"):
        output[name] = vertices[name]
    dc = np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    )
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    output["red"] = np.rint(255.0 * rgb[:, 0]).astype(np.uint8)
    output["green"] = np.rint(255.0 * rgb[:, 1]).astype(np.uint8)
    output["blue"] = np.rint(255.0 * rgb[:, 2]).astype(np.uint8)
    output["opacity"] = _sigmoid(vertices["opacity"]).astype(np.float32)
    output["maximum_scale"] = np.exp(
        np.maximum(vertices["scale_0"], vertices["scale_1"])
    ).astype(np.float32)
    output["source_index"] = source_indices.astype(np.int32)
    if "primitive_class" in (vertices.dtype.names or ()):
        output["primitive_class"] = np.asarray(
            vertices["primitive_class"]
        ).astype(np.int16)
    else:
        output["primitive_class"] = 0
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--visualization-ply", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--opacity-min", type=float, default=0.0)
    parser.add_argument("--structural-class", type=int, default=0)
    parser.add_argument("--drop-nonstructural", action="store_true")
    parser.add_argument("--isolation-distance", type=float)
    parser.add_argument("--isolation-scale-ratio", type=float, default=4.0)
    parser.add_argument("--isolation-opacity-max", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_ply = args.input_ply.resolve()
    output_ply = args.output_ply.resolve()
    visualization_ply = (
        None
        if args.visualization_ply is None
        else args.visualization_ply.resolve()
    )
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_ply.with_suffix(".manifest.json")
    )
    if not input_ply.is_file():
        raise FileNotFoundError(input_ply)
    for path in (output_ply, visualization_ply, manifest_path):
        if path is not None and path.exists():
            raise FileExistsError(path)

    source = PlyData.read(str(input_ply))
    if len(source.elements) != 1 or source.elements[0].name != "vertex":
        raise RuntimeError("Expected exactly one vertex element")
    vertices = np.asarray(source["vertex"].data)
    retained, source_indices, audit = clean_vertices(
        vertices,
        opacity_min=args.opacity_min,
        structural_class=args.structural_class,
        drop_nonstructural=args.drop_nonstructural,
        isolation_distance=args.isolation_distance,
        isolation_scale_ratio=args.isolation_scale_ratio,
        isolation_opacity_max=args.isolation_opacity_max,
    )
    if not len(retained):
        raise RuntimeError("Cleaning removed every Gaussian")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(retained, "vertex")],
        text=False,
        byte_order=source.byte_order,
    ).write(str(output_ply))
    if visualization_ply is not None:
        visualization_ply.parent.mkdir(parents=True, exist_ok=True)
        PlyData(
            [
                PlyElement.describe(
                    visualization_vertices(retained, source_indices), "vertex"
                )
            ],
            text=False,
            byte_order=source.byte_order,
        ).write(str(visualization_ply))

    manifest = {
        "protocol": "gaussian_point_cloud_clean_v1",
        "input": {
            "ply": str(input_ply),
            "sha256": _sha256(input_ply),
        },
        "output": {
            "native_ply": str(output_ply),
            "native_ply_sha256": _sha256(output_ply),
            "visualization_ply": (
                None if visualization_ply is None else str(visualization_ply)
            ),
            "visualization_ply_sha256": (
                None
                if visualization_ply is None
                else _sha256(visualization_ply)
            ),
        },
        "audit": audit,
        "rendering_contract": (
            "native_ply preserves every field and value of retained rows; "
            "visualization_ply is a derived xyz/RGB/scalar-field point cloud"
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
