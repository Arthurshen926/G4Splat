#!/usr/bin/env python3
"""Conservatively edit appended warm-start Gaussians without touching a baseline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814


def constrain_warmstart_ply(
    baseline_ply: Path,
    candidate_ply: Path,
    output_ply: Path,
    max_opacity: float = 0.15,
    max_scale: float = 0.30,
    diffuse_only: bool = True,
    clamp_dc: bool = True,
) -> dict:
    baseline = PlyData.read(baseline_ply)
    candidate = PlyData.read(candidate_ply)
    baseline_vertex = baseline["vertex"].data
    candidate_vertex = candidate["vertex"].data
    baseline_count = len(baseline_vertex)
    if len(candidate_vertex) < baseline_count:
        raise ValueError("Candidate has fewer Gaussians than the baseline")
    if baseline_vertex.dtype.names != candidate_vertex.dtype.names:
        raise ValueError("Baseline and candidate PLY schemas differ")
    if not 0.0 < max_opacity < 1.0:
        raise ValueError("max_opacity must be in (0, 1)")
    if max_scale <= 0.0:
        raise ValueError("max_scale must be positive")

    data = candidate_vertex.copy()
    for name in data.dtype.names:
        data[name][:baseline_count] = baseline_vertex[name]

    suffix = slice(baseline_count, None)
    opacity_before = 1.0 / (1.0 + np.exp(-data["opacity"][suffix].astype(np.float64)))
    scale_before = np.stack(
        [
            np.exp(data["scale_0"][suffix].astype(np.float64)),
            np.exp(data["scale_1"][suffix].astype(np.float64)),
        ],
        axis=1,
    )
    data["opacity"][suffix] = np.minimum(
        data["opacity"][suffix], np.log(max_opacity / (1.0 - max_opacity))
    )
    max_log_scale = np.log(max_scale)
    data["scale_0"][suffix] = np.minimum(data["scale_0"][suffix], max_log_scale)
    data["scale_1"][suffix] = np.minimum(data["scale_1"][suffix], max_log_scale)

    rest_names = [name for name in data.dtype.names if name.startswith("f_rest_")]
    if diffuse_only:
        for name in rest_names:
            data[name][suffix] = 0.0
    if clamp_dc:
        dc_limit = 0.5 / SH_C0
        for name in ("f_dc_0", "f_dc_1", "f_dc_2"):
            data[name][suffix] = np.clip(data[name][suffix], -dc_limit, dc_limit)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    output = PlyData(
        [PlyElement.describe(data, "vertex")],
        text=candidate.text,
        byte_order=candidate.byte_order,
        comments=candidate.comments,
        obj_info=candidate.obj_info,
    )
    output.write(output_ply)
    report = {
        "baseline_count": baseline_count,
        "candidate_count": len(data),
        "appended_count": len(data) - baseline_count,
        "max_opacity": max_opacity,
        "max_scale": max_scale,
        "diffuse_only": diffuse_only,
        "clamp_dc": clamp_dc,
        "opacity_clamped_count": int((opacity_before > max_opacity).sum()),
        "scale_clamped_count": int((scale_before.max(axis=1) > max_scale).sum()),
    }
    output_ply.with_suffix(".constraints.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_ply", type=Path, required=True)
    parser.add_argument("--candidate_model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--max_opacity", type=float, default=0.15)
    parser.add_argument("--max_scale", type=float, default=0.30)
    parser.add_argument("--keep_sh_rest", action="store_true")
    parser.add_argument("--no_clamp_dc", action="store_true")
    args = parser.parse_args()

    candidate_ply = (
        args.candidate_model
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )
    output_ply = (
        args.output_model
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )
    report = constrain_warmstart_ply(
        baseline_ply=args.baseline_ply,
        candidate_ply=candidate_ply,
        output_ply=output_ply,
        max_opacity=args.max_opacity,
        max_scale=args.max_scale,
        diffuse_only=not args.keep_sh_rest,
        clamp_dc=not args.no_clamp_dc,
    )
    cfg_args = args.candidate_model / "cfg_args"
    if cfg_args.is_file():
        args.output_model.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_args, args.output_model / "cfg_args")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
