#!/usr/bin/env python
"""Derive an exact paired MASt3R-only Evidence Store ablation.

All metric/weak/semantic artifacts remain content-identical to the parent;
only optional SfM coverage tracks and their authority metadata are removed.
This is stricter than rebuilding the store because it prevents cache, mask,
or Chart drift from becoming an unreported second A/B variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import (  # noqa: E402
    MAST3R_ONLY_GEOMETRY,
    MAST3R_PRIMARY_SFM_COVERAGE,
    load_evidence_store,
)


SFM_ARTIFACTS = frozenset({"colmap_tracks", "colmap_tracks_summary"})


def _canonical_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def derive_mast3r_only_manifest(parent: dict[str, Any]) -> dict[str, Any]:
    """Return the single-variable no-SfM partner of a coverage store."""
    if parent.get("geometry_source") != MAST3R_PRIMARY_SFM_COVERAGE:
        raise ValueError(
            "Parent Evidence Store must use mast3r_primary_sfm_coverage"
        )
    artifacts = [dict(row) for row in parent.get("artifacts", ())]
    removed = [row for row in artifacts if row.get("name") in SFM_ARTIFACTS]
    if {row.get("name") for row in removed} != SFM_ARTIFACTS:
        raise ValueError("Parent Evidence Store lacks complete SfM artifacts")
    manifest = dict(parent)
    manifest.pop("evidence_hash", None)
    manifest["geometry_source"] = MAST3R_ONLY_GEOMETRY
    manifest["geometry_authority"] = "mast3r_matcha_primary"
    manifest["sfm_track_usage_mode"] = "disabled"
    manifest["colmap_points_or_tracks_used"] = False
    manifest["camera_container"] = (
        "cameras.bin/images.bin are serialization only; no points3D.bin or "
        "COLMAP track geometry is consumed"
    )
    manifest["artifacts"] = [
        row for row in artifacts if row.get("name") not in SFM_ARTIFACTS
    ]
    manifest["geometry_ablation"] = {
        "protocol": "exact-paired-mast3r-only-evidence-ablation-v1",
        "parent_evidence_hash": str(parent["evidence_hash"]),
        "only_changed_authority": "optional_sfm_coverage_tracks_removed",
        "removed_artifacts": [
            {
                "name": str(row["name"]),
                "sha256": str(row["sha256"]),
            }
            for row in sorted(removed, key=lambda value: value["name"])
        ],
        "all_other_artifact_hashes_identical": True,
        "camera_mask_chart_pointmap_dav2_plane_contract_identical": True,
    }
    manifest["evidence_hash"] = _canonical_json_digest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-evidence-store", type=Path, required=True)
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
    manifest = derive_mast3r_only_manifest(parent)
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
                "geometry_source": manifest["geometry_source"],
                "artifact_count": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
