#!/usr/bin/env python
"""Add one all-camera rigid posterior to an otherwise exact Evidence Store.

This produces a causal paired store: every existing artifact remains byte
identical and only the projected mask-conflict posterior plus its audit
sidecar are appended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.projected_role_posterior import (  # noqa: E402
    PROJECTED_RIGID_POSTERIOR_VERSION,
)
from outdoor.scene_contract import sha256_file  # noqa: E402


ARTIFACT_NAMES = frozenset(
    {
        "projected_rigid_conflict_posterior",
        "projected_rigid_conflict_posterior_summary",
    }
)


def _canonical_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def derive_projected_rigid_manifest(
    parent: dict[str, Any], posterior: Path
) -> dict[str, Any]:
    posterior = Path(posterior).resolve()
    summary = posterior.with_suffix(".json")
    if not posterior.is_file() or not summary.is_file():
        raise FileNotFoundError(posterior if not posterior.is_file() else summary)
    with np.load(posterior, allow_pickle=False) as archive:
        schema = str(archive["schema_version"].item())
        camera_count = len(archive["camera_names"])
        observation_count = int(archive["offsets"][-1])
    if schema != PROJECTED_RIGID_POSTERIOR_VERSION:
        raise ValueError(f"Unsupported projected posterior {schema!r}")
    artifacts = [dict(row) for row in parent.get("artifacts", ())]
    duplicate = ARTIFACT_NAMES & {str(row.get("name")) for row in artifacts}
    if duplicate:
        raise ValueError(
            "Parent already contains projected rigid artifacts: "
            + ", ".join(sorted(duplicate))
        )
    artifacts.extend(
        [
            {
                "name": "projected_rigid_conflict_posterior",
                "source_type": "mast3r_all_fixed_camera_rigid_projection",
                "path": str(posterior),
                "sha256": sha256_file(posterior),
                "bytes": posterior.stat().st_size,
                "measurement": (
                    "ragged all-camera stable rigid geometry and current-RGB "
                    "visibility posterior at object/sky/tree mask conflicts"
                ),
                "coordinate_frame": "fixed_camera_mask_raster_pixels",
                "covariance": (
                    "track_reprojection_baseline_cycle_stability_and_"
                    "robust_appearance_consistency"
                ),
                "camera_scope": "fixed_calibrated_database",
                "sequence_scope": "cross_sequence_tracks_projected_all_views",
                "semantic_role": (
                    "positive_rigid_rescue_and_occluded_background_support"
                ),
                "validity": (
                    "geometry support distinct from current-image visibility"
                ),
            },
            {
                "name": "projected_rigid_conflict_posterior_summary",
                "source_type": "mast3r_all_fixed_camera_rigid_projection",
                "path": str(summary),
                "sha256": sha256_file(summary),
                "bytes": summary.stat().st_size,
                "measurement": "projection coverage and immutable input audit",
                "coordinate_frame": "metadata",
                "covariance": "documented_in_summary",
                "camera_scope": "fixed_calibrated_database",
                "sequence_scope": "cross_sequence",
                "semantic_role": "audit",
                "validity": f"{camera_count} fixed cameras",
            },
        ]
    )
    manifest = dict(parent)
    manifest.pop("evidence_hash", None)
    manifest["artifacts"] = sorted(artifacts, key=lambda row: row["name"])
    manifest["ownership_posterior_upgrade"] = {
        "protocol": (
            "exact-parent-plus-all-camera-rigid-conflict-posterior-v2-"
            "tree-occlusion-geometry-with-exact-observation-visibility"
        ),
        "parent_evidence_hash": str(parent["evidence_hash"]),
        "only_added_artifacts": sorted(ARTIFACT_NAMES),
        "all_parent_artifact_hashes_identical": True,
        "camera_count": int(camera_count),
        "projected_conflict_observation_count": observation_count,
        "semantic_masks_are_priors_not_negative_geometry_authority": True,
        "geometry_support_separate_from_current_rgb_visibility": True,
        "tree_occluded_geometry_preserved": True,
        "tree_occluded_rgb_requires_exact_current_camera_observation": True,
    }
    manifest["evidence_hash"] = _canonical_json_digest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-evidence-store", type=Path, required=True)
    parser.add_argument("--projected-rigid-posterior", type=Path, required=True)
    parser.add_argument("--output-evidence-store", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    parent = load_evidence_store(args.parent_evidence_store)
    output = args.output_evidence_store.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    manifest = derive_projected_rigid_manifest(
        parent, args.projected_rigid_posterior
    )
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    load_evidence_store(output)
    print(
        json.dumps(
            {
                "evidence_store": str(output),
                "evidence_hash": manifest["evidence_hash"],
                "parent_evidence_hash": parent["evidence_hash"],
                "artifact_count": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
