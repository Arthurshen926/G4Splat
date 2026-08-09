#!/usr/bin/env python3
"""Upgrade a complete foliage seed without rebuilding its ray evidence.

The expensive visual-hull/DAV2/temporal producer is immutable evidence.  This
migration repairs a historical schema defect in which strong tracked branch
rows kept the generic crown role and every merged track received zero static
skeleton confidence.  It never invents points: only existing cross-sequence,
multi-view tracks are relabelled, and local replacement groups are rebuilt
after the ownership change.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "2d-gaussian-splatting")]

from outdoor.hybrid_gaussian_renderer import (  # noqa: E402
    LAYER_CANONICAL_CROWN,
    LAYER_STATIC_SKELETON,
)
from outdoor.role_aware_initialization import (  # noqa: E402
    _local_replacement_groups,
)
from outdoor.runtime_provenance import collect_runtime_provenance  # noqa: E402
from outdoor.scene_contract import sha256_file  # noqa: E402


UPGRADE_VERSION = "static-foliage-track-skeleton-upgrade-v1"


def _promotion_and_confidence(
    payload: dict,
    *,
    minimum_views: int,
    minimum_sequences: int,
    maximum_reprojection_error: float,
    minimum_occupancy: float,
    minimum_linearity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = torch.as_tensor(payload["layer_role"], dtype=torch.int8).cpu()
    track_id = torch.as_tensor(payload["track_id"], dtype=torch.int64).cpu()
    views = torch.as_tensor(payload["support_view_count"]).float().cpu()
    sequences = torch.as_tensor(
        payload["support_sequence_count"]
    ).float().cpu()
    reprojection = torch.as_tensor(
        payload["reprojection_error"]
    ).float().cpu()
    occupancy = torch.as_tensor(
        payload["occupancy_probability"]
    ).float().cpu()
    linearity = torch.as_tensor(payload["track_linearity"]).float().cpu()
    finite = (
        torch.isfinite(reprojection)
        & torch.isfinite(occupancy)
        & torch.isfinite(linearity)
    )
    promote = (
        (layer == LAYER_CANONICAL_CROWN)
        & (track_id >= 0)
        & (views >= int(minimum_views))
        & (sequences >= int(minimum_sequences))
        & (reprojection <= float(maximum_reprojection_error))
        & (occupancy >= float(minimum_occupancy))
        & (linearity >= float(minimum_linearity))
        & finite
    )
    view_confidence = ((views - 1.0) / 4.0).clamp(0.0, 1.0)
    sequence_confidence = ((sequences - 1.0) / 2.0).clamp(0.0, 1.0)
    reprojection_confidence = torch.exp(
        -reprojection.clamp_min(0.0)
        / max(float(maximum_reprojection_error), 1.0e-6)
    )
    line_confidence = (
        (linearity - 1.0) / max(float(minimum_linearity), 1.01)
    ).clamp(0.0, 1.0)
    confidence = torch.sqrt(
        (
            view_confidence
            * sequence_confidence
            * reprojection_confidence
            * occupancy.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
    ) * (0.35 + 0.65 * line_confidence)
    confidence[~finite] = 0.0
    return promote, confidence


def upgrade_payload(
    payload: dict,
    *,
    minimum_views: int = 3,
    minimum_sequences: int = 2,
    maximum_reprojection_error: float = 1.5,
    minimum_occupancy: float = 0.65,
    minimum_linearity: float = 1.8,
    replacement_distance: float = 0.36,
) -> tuple[dict, dict]:
    result = dict(payload)
    layer = torch.as_tensor(payload["layer_role"], dtype=torch.int8).cpu().clone()
    skeleton_before = layer == LAYER_STATIC_SKELETON
    promote, confidence = _promotion_and_confidence(
        payload,
        minimum_views=minimum_views,
        minimum_sequences=minimum_sequences,
        maximum_reprojection_error=maximum_reprojection_error,
        minimum_occupancy=minimum_occupancy,
        minimum_linearity=minimum_linearity,
    )
    layer[promote] = LAYER_STATIC_SKELETON
    skeleton_after = layer == LAYER_STATIC_SKELETON
    previous_confidence = torch.as_tensor(
        payload.get(
            "static_skeleton_confidence",
            torch.zeros(len(layer), dtype=torch.float32),
        )
    ).float().cpu()
    confidence = torch.maximum(previous_confidence, confidence)
    confidence[~skeleton_after] = 0.0
    opacity = torch.as_tensor(payload["opacities"]).float().cpu().clone()
    # A tracked branch should be optically present at initialization, but it
    # remains far from opaque and still has to earn mass from real RGB/rays.
    opacity[promote] = torch.maximum(
        opacity[promote], torch.full_like(opacity[promote], 0.04)
    )
    groups, group_audit = _local_replacement_groups(
        torch.as_tensor(payload["centers"]).cpu().numpy(),
        layer.numpy(),
        torch.as_tensor(payload["tree_instance_id"]).cpu().numpy(),
        maximum_center_distance=float(replacement_distance),
    )
    result["layer_role"] = layer
    result["static_skeleton_confidence"] = confidence
    result["opacities"] = opacity
    result["replacement_group"] = torch.from_numpy(groups)
    nonzero_confidence = skeleton_after & (confidence > 0)
    audit = {
        "protocol": UPGRADE_VERSION,
        "contract": (
            "existing_cross_sequence_multiview_track_only__no_new_xyz__"
            "rebuild_local_replacement_groups_after_role_change"
        ),
        "skeleton_before": int(skeleton_before.sum()),
        "promoted_track_rows": int(promote.sum()),
        "skeleton_after": int(skeleton_after.sum()),
        "skeleton_nonzero_confidence": int(nonzero_confidence.sum()),
        "minimum_views": int(minimum_views),
        "minimum_sequences": int(minimum_sequences),
        "maximum_reprojection_error": float(maximum_reprojection_error),
        "minimum_occupancy": float(minimum_occupancy),
        "minimum_linearity": float(minimum_linearity),
        "replacement_distance": float(replacement_distance),
        "replacement_groups": group_audit,
    }
    result["audit"] = dict(payload.get("audit", {}))
    result["audit"]["static_track_skeleton_upgrade"] = audit
    return result, audit


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source = args.source_initialization.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        if not (output / "surface_seed.npz").is_file():
            raise RuntimeError("refusing to replace an unrelated directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    source_manifest_path = source / "initialization_manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_foliage = Path(manifest["foliage_seed"]).resolve()
    payload = torch.load(source_foliage, map_location="cpu", weights_only=False)
    upgraded, audit = upgrade_payload(payload)
    foliage_path = output / "foliage_seed_gaussians.pth"
    torch.save(upgraded, foliage_path)
    for name in ("surface_seed.npz", "surface_seed.json"):
        _link_or_copy(source / name, output / name)
    foliage_json_source = source / "foliage_seed_gaussians.json"
    foliage_json = json.loads(foliage_json_source.read_text(encoding="utf-8"))
    foliage_json["static_track_skeleton_upgrade"] = audit
    foliage_json["static_skeleton"] = audit["skeleton_after"]
    (output / "foliage_seed_gaussians.json").write_text(
        json.dumps(foliage_json, indent=2) + "\n", encoding="utf-8"
    )
    manifest["surface_seed"] = str((output / "surface_seed.npz").resolve())
    manifest["foliage_seed"] = str(foliage_path.resolve())
    manifest["foliage"] = dict(manifest.get("foliage", {}))
    manifest["foliage"]["static_track_skeleton_upgrade"] = audit
    manifest["foliage"]["static_skeleton"] = audit["skeleton_after"]
    contract = dict(manifest.get("initialization_contract", {}))
    contract["static_track_skeleton_upgrade"] = {
        **audit,
        "source_initialization": str(source),
        "source_foliage_sha256": sha256_file(source_foliage),
        "upgraded_foliage_sha256": sha256_file(foliage_path),
    }
    if isinstance(contract.get("foliage_reuse"), dict):
        contract["foliage_reuse"] = dict(contract["foliage_reuse"])
        contract["foliage_reuse"]["foliage_seed_sha256"] = sha256_file(
            foliage_path
        )
    manifest["initialization_contract"] = contract
    manifest["runtime_provenance"] = collect_runtime_provenance(
        ROOT,
        python_modules=(
            "scripts.upgrade_static_foliage_seed",
            "outdoor.role_aware_initialization",
        ),
    )
    manifest["migrated_from"] = {
        "initialization_manifest": str(source_manifest_path),
        "initialization_manifest_sha256": sha256_file(source_manifest_path),
        "foliage_seed_sha256": sha256_file(source_foliage),
        "migration_protocol": UPGRADE_VERSION,
    }
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
