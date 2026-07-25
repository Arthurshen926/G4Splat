#!/usr/bin/env python
"""Bake a layered foliage UV ownership state into a clean structural 2DGS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from outdoor.surfel_uv_bake import bake_surfel_uv_ownership  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_dict_to_numpy(payload: dict) -> dict[str, np.ndarray]:
    return {
        name: (
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
        )
        for name, value in payload.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-ply", type=Path, required=True)
    parser.add_argument("--layered-state", type=Path, required=True)
    parser.add_argument("--replacement-audit", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument(
        "--mode",
        choices=("uv-equivalent", "rigid-clean"),
        required=True,
    )
    parser.add_argument("--rigid-threshold", type=float, default=0.05)
    parser.add_argument("--rigid-dominance", type=float, default=1.0)
    parser.add_argument(
        "--global-canopy-threshold", type=float, default=0.45
    )
    parser.add_argument(
        "--global-rigid-threshold", type=float, default=0.20
    )
    parser.add_argument(
        "--minimum-global-support-views", type=int, default=3
    )
    parser.add_argument("--maximum-bake-radius", type=float, default=4096)
    parser.add_argument(
        "--minimum-bake-camera-depth", type=float, default=1.0
    )
    parser.add_argument(
        "--child-sigma-over-spacing", type=float, default=0.55
    )
    parser.add_argument(
        "--minimum-child-peak-alpha",
        type=float,
        default=1.0 / 255.0,
    )
    parser.add_argument(
        "--keep-nonstructural",
        action="store_true",
        help=(
            "Keep historical primitive_class!=structural rows in rigid-clean "
            "mode. By default they are removed from the clean base."
        ),
    )
    args = parser.parse_args()
    if args.output_sidecar is None:
        args.output_sidecar = args.output_ply.with_suffix(
            ".provenance.pth"
        )
    if args.output_manifest is None:
        args.output_manifest = args.output_ply.with_suffix(".manifest.json")
    return args


def main() -> None:
    args = _parse_args()
    structural_path = args.structural_ply.resolve()
    state_path = args.layered_state.resolve()
    audit_path = args.replacement_audit.resolve()
    for path in (structural_path, state_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.output_ply.resolve(),
        args.output_sidecar.resolve(),
        args.output_manifest.resolve(),
    ):
        if path.exists():
            raise FileExistsError(path)

    ply = PlyData.read(str(structural_path))
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise RuntimeError("Expected a single vertex PLY element")
    vertices = np.asarray(ply.elements[0].data)
    state = torch.load(state_path, map_location="cpu")
    audit = torch.load(audit_path, map_location="cpu")
    if state.get("protocol") != "layered_dynamic_foliage_surfel_uv_replace_v7":
        raise RuntimeError("Layered state does not contain intrinsic UV gates")
    state_candidates = state["legacy_candidate_indices"].long()
    audit_candidates = audit["candidate_indices"].long()
    if not torch.equal(state_candidates, audit_candidates):
        raise RuntimeError(
            "Layered state and replacement audit candidate order differ"
        )
    result = bake_surfel_uv_ownership(
        vertices,
        state_candidates.numpy(),
        state["surfel_uv_gate_atlas"].numpy(),
        state["surfel_uv_evidence_sum"].numpy(),
        state["surfel_uv_evidence_weight"].numpy(),
        _tensor_dict_to_numpy(audit["statistics"]),
        mode=args.mode,
        rigid_threshold=args.rigid_threshold,
        rigid_dominance=args.rigid_dominance,
        global_canopy_threshold=args.global_canopy_threshold,
        global_rigid_threshold=args.global_rigid_threshold,
        minimum_global_support_views=args.minimum_global_support_views,
        maximum_bake_radius=args.maximum_bake_radius,
        minimum_bake_camera_depth=args.minimum_bake_camera_depth,
        child_sigma_over_spacing=args.child_sigma_over_spacing,
        minimum_child_peak_alpha=args.minimum_child_peak_alpha,
        remove_nonstructural=not args.keep_nonstructural,
    )
    output_ply = args.output_ply.resolve()
    output_sidecar = args.output_sidecar.resolve()
    output_manifest = args.output_manifest.resolve()
    for path in (output_ply, output_sidecar, output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(result.vertices, "vertex")],
        text=False,
        byte_order=ply.byte_order,
    ).write(str(output_ply))
    torch.save(
        {
            "version": "surfel_uv_bake_provenance_v1",
            "mode": args.mode,
            "source_parent_index": torch.from_numpy(
                result.source_parent_index
            ),
            "source_uv_x": torch.from_numpy(result.source_uv_x),
            "source_uv_y": torch.from_numpy(result.source_uv_y),
            "ownership_role": torch.from_numpy(result.ownership_role),
            "audit": result.audit,
        },
        output_sidecar,
    )
    manifest = {
        "version": "surfel_uv_baked_structure_v1",
        "mode": args.mode,
        "input": {
            "structural_ply": str(structural_path),
            "structural_ply_sha256": _sha256(structural_path),
            "layered_state": str(state_path),
            "layered_state_sha256": _sha256(state_path),
            "replacement_audit": str(audit_path),
            "replacement_audit_sha256": _sha256(audit_path),
        },
        "output": {
            "structural_ply": str(output_ply),
            "structural_ply_sha256": _sha256(output_ply),
            "provenance": str(output_sidecar),
            "provenance_sha256": _sha256(output_sidecar),
        },
        "audit": result.audit,
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result.audit, indent=2))


if __name__ == "__main__":
    main()
