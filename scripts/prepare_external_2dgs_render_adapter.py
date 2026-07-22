#!/usr/bin/env python3
"""Adapt a local 2DGS PLY for a clean external gsplat 2DGS renderer.

The local G4Splat 2DGS and clean ULFLoc/STDLoc 2DGS PLY layouts share RGB,
position, opacity, scale and quaternion fields.  The external implementation
additionally requires at least one ``loc_*`` field even when RGB-only
rendering never reads it.  This adapter appends one zero-valued, explicitly
unused field without changing any source Gaussian parameter.  It enables a
renderer-only evaluation of the *same trained local PLY*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_vertex_for_external_2dgs(vertex: np.ndarray) -> np.ndarray:
    """Append the one external-only location-feature field losslessly."""
    names = vertex.dtype.names
    if names is None:
        raise RuntimeError("Expected a structured PLY vertex array")
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"Input does not have the expected local 2DGS fields: {missing}")
    existing_loc = [name for name in names if name.startswith("loc_")]
    if existing_loc:
        raise RuntimeError(
            "Input already contains external location-feature fields; refusing to "
            f"rewrite its parameter schema: {existing_loc[:3]}"
        )

    dtype = list(vertex.dtype.descr) + [("loc_0", "<f4")]
    adapted = np.empty(vertex.shape, dtype=dtype)
    for name in names:
        adapted[name] = vertex[name]
    adapted["loc_0"] = 0.0
    return adapted


def build_adapter(input_ply: Path, output_ply: Path) -> dict:
    input_ply = input_ply.resolve()
    output_ply = output_ply.resolve()
    if not input_ply.is_file():
        raise FileNotFoundError(f"Local 2DGS PLY does not exist: {input_ply}")
    if output_ply.exists():
        raise FileExistsError(f"Refusing to overwrite renderer-adapter PLY: {output_ply}")

    source = PlyData.read(str(input_ply))
    if len(source.elements) != 1 or source.elements[0].name != "vertex":
        raise RuntimeError("Expected exactly one vertex element in the local 2DGS PLY")
    adapted_vertex = adapt_vertex_for_external_2dgs(source["vertex"].data)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(adapted_vertex, "vertex")], text=False).write(str(output_ply))

    return {
        "protocol": "local_2dgs_ply_to_clean_external_2dgs_renderer_v1",
        "source_ply": str(input_ply),
        "source_ply_sha256": _sha256(input_ply),
        "output_ply": str(output_ply),
        "output_ply_sha256": _sha256(output_ply),
        "gaussian_count": int(len(adapted_vertex)),
        "copied_parameter_fields": list(source["vertex"].data.dtype.names or ()),
        "added_external_only_fields": ["loc_0"],
        "added_field_value": 0.0,
        "rgb_only_contract": (
            "clean external 2dgs renderer receives the exact source position, RGB, "
            "opacity, scale, and quaternion fields; loc_0 exists only to satisfy "
            "its PLY loader and is not consumed by rgb_only rendering"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON provenance record; defaults beside --output-ply.",
    )
    args = parser.parse_args()
    manifest = build_adapter(args.input_ply, args.output_ply)
    manifest_path = (args.manifest or args.output_ply.with_suffix(".adapter_manifest.json")).resolve()
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite adapter manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
