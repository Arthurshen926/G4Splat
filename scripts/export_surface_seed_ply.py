#!/usr/bin/env python
"""Export a role-aware surface seed as a native 2DGS PLY.

This is an audit/bridging utility for the rigid-first Cambridge stage.  It
does not read COLMAP points or a historical Gaussian model: every exported
parameter comes from the supplied MASt3R/MAtCha surface seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "2d-gaussian-splatting"))

from scene.gaussian_model import GaussianModel  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-seed", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument(
        "--scene-extent",
        type=float,
        default=1.0,
        help=(
            "Only initializes the model's optimizer scale metadata; physical "
            "XYZ/scales written to the PLY are unchanged."
        ),
    )
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()
    if args.scene_extent <= 0:
        parser.error("--scene-extent must be positive")
    return args


def main() -> None:
    args = _parse_args()
    source = args.surface_seed.resolve()
    output = args.output_ply.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    for path in (output, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    with np.load(source, allow_pickle=False) as seed:
        required = {
            "xyz",
            "scales",
            "quaternions",
            "rgb",
            "initial_opacity",
            "source_type",
            "track_id",
            "geometry_confidence",
        }
        missing = sorted(required - set(seed.files))
        if missing:
            raise RuntimeError(f"Surface seed is missing fields: {missing}")
        xyz = torch.from_numpy(seed["xyz"]).float().cuda()
        scales = torch.from_numpy(seed["scales"]).float().cuda()
        quaternions = torch.from_numpy(seed["quaternions"]).float().cuda()
        rgb = torch.from_numpy(seed["rgb"]).float().cuda()
        opacity = torch.from_numpy(seed["initial_opacity"]).float().cuda()
        source_type = torch.from_numpy(seed["source_type"]).to(
            device="cuda", dtype=torch.int16
        )
        track_id = torch.from_numpy(seed["track_id"]).to(
            device="cuda", dtype=torch.int64
        )
        confidence = torch.from_numpy(seed["geometry_confidence"]).float().cuda()
        seed_version = (
            str(seed["version"].item()) if "version" in seed else "unknown"
        )

    count = int(len(xyz))
    for name, value in {
        "scales": scales,
        "quaternions": quaternions,
        "rgb": rgb,
        "initial_opacity": opacity,
        "source_type": source_type,
        "track_id": track_id,
        "geometry_confidence": confidence,
    }.items():
        if len(value) != count:
            raise RuntimeError(
                f"{name} row count {len(value)} does not match XYZ {count}"
            )
    if count == 0:
        raise RuntimeError("Cannot export an empty surface seed")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (xyz, scales, quaternions, rgb, opacity, confidence)
    ):
        raise RuntimeError("Surface seed contains non-finite parameters")
    if bool((scales <= 0).any()):
        raise RuntimeError("Surface seed scales must be positive")
    if bool(((opacity <= 0) | (opacity >= 1)).any()):
        raise RuntimeError("Surface seed opacities must lie in (0, 1)")

    surface = GaussianModel(args.sh_degree)
    surface.create_from_parameters(
        xyz,
        scales,
        quaternions,
        rgb,
        float(args.scene_extent),
    )
    surface._opacity = torch.nn.Parameter(
        torch.logit(opacity[:, None].clamp(1e-6, 1 - 1e-6))
    )
    surface._source_type = source_type
    surface._track_id = track_id
    surface._geometry_confidence = confidence
    surface._primitive_class.fill_(GaussianModel.PRIMITIVE_STRUCTURAL)
    surface._protected_flag.zero_()
    surface._block_id.fill_(-1)

    output.parent.mkdir(parents=True, exist_ok=True)
    surface.save_ply(str(output))
    manifest = {
        "protocol": "role-aware-surface-seed-to-native-2dgs-v1",
        "surface_seed": str(source),
        "surface_seed_sha256": _sha256(source),
        "surface_seed_version": seed_version,
        "output_ply": str(output),
        "output_ply_sha256": _sha256(output),
        "point_count": count,
        "sh_degree": int(args.sh_degree),
        "historical_gaussian_input_used": False,
        "colmap_points_or_tracks_used": False,
        "parameter_mapping": {
            "xyz": "exact",
            "tangent_scales": "exact",
            "quaternion_wxyz": "exact",
            "rgb_to_sh_dc": "GaussianModel.create_from_parameters",
            "opacity": "exact_logit",
            "higher_order_sh": "zero",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
