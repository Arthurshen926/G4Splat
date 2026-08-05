from copy import deepcopy

import pytest

from scripts.derive_mast3r_only_evidence import (
    derive_mast3r_only_manifest,
)


def _parent():
    return {
        "evidence_hash": "parent-hash",
        "geometry_source": "mast3r_primary_sfm_coverage",
        "geometry_authority": "mast3r_matcha_primary",
        "sfm_track_usage_mode": "coverage_only",
        "colmap_points_or_tracks_used": True,
        "artifacts": [
            {"name": "mast3r_tracks", "sha256": "mast3r"},
            {"name": "chart_geometry", "sha256": "chart"},
            {"name": "colmap_tracks", "sha256": "sfm"},
            {"name": "colmap_tracks_summary", "sha256": "summary"},
        ],
    }


def test_mast3r_only_ablation_changes_only_sfm_authority():
    parent = _parent()
    result = derive_mast3r_only_manifest(deepcopy(parent))

    assert result["geometry_source"] == "mast3r_only"
    assert result["sfm_track_usage_mode"] == "disabled"
    assert result["colmap_points_or_tracks_used"] is False
    assert [row["name"] for row in result["artifacts"]] == [
        "mast3r_tracks",
        "chart_geometry",
    ]
    assert result["geometry_ablation"]["parent_evidence_hash"] == (
        "parent-hash"
    )
    assert result["geometry_ablation"][
        "all_other_artifact_hashes_identical"
    ] is True
    assert result["evidence_hash"] != parent["evidence_hash"]


def test_mast3r_only_ablation_requires_complete_coverage_parent():
    parent = _parent()
    parent["artifacts"].pop()
    with pytest.raises(ValueError, match="complete SfM artifacts"):
        derive_mast3r_only_manifest(parent)
