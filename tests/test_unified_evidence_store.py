import json
from pathlib import Path

import pytest

from outdoor.evidence_store import (
    EVIDENCE_STORE_VERSION,
    EvidenceStoreBuilder,
    MAST3R_PRIMARY_SFM_COVERAGE,
    artifact_path,
    load_evidence_store,
    mast3r_is_geometry_authority,
    sfm_coverage_tracks_enabled,
)


def test_evidence_store_hashes_every_registered_source(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    scene = tmp_path / "scene.json"
    semantic = tmp_path / "semantic.json"
    tracks = tmp_path / "tracks.npz"
    scene.write_text('{"camera":"fixed"}\n')
    semantic.write_text('{"roles":["rigid","canopy"]}\n')
    tracks.write_bytes(b"source-aware-track-archive")
    root = tmp_path / "evidence"
    builder = EvidenceStoreBuilder(
        root,
        dataset=dataset,
        scene_contract=scene,
        semantic_contract=semantic,
    )
    builder.add_file(
        "tracks",
        "colmap",
        tracks,
        measurement="xyz plus tracks",
    )
    payload = builder.write()
    assert payload["schema_version"] == EVIDENCE_STORE_VERSION
    assert payload["fusion_policy"]["source_measurements_retained"]
    loaded = load_evidence_store(root)
    assert loaded["evidence_hash"] == payload["evidence_hash"]
    assert artifact_path(loaded, "tracks") == tracks


def test_evidence_store_detects_mutated_artifact(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    scene = tmp_path / "scene.json"
    semantic = tmp_path / "semantic.json"
    source = tmp_path / "source.bin"
    scene.write_text("{}")
    semantic.write_text("{}")
    source.write_bytes(b"before")
    root = tmp_path / "evidence"
    builder = EvidenceStoreBuilder(
        root,
        dataset=dataset,
        scene_contract=scene,
        semantic_contract=semantic,
    )
    builder.add_file(
        "source",
        "chart",
        source,
        measurement="pointmap",
    )
    builder.write()
    source.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="changed after registration"):
        load_evidence_store(root)


def test_sfm_coverage_preserves_mast3r_authority_and_camera_independence(
    tmp_path,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    scene = tmp_path / "scene.json"
    semantic = tmp_path / "semantic.json"
    tracks = tmp_path / "colmap_tracks.npz"
    scene.write_text("{}")
    semantic.write_text("{}")
    tracks.write_bytes(b"raw-sfm-track-evidence")
    builder = EvidenceStoreBuilder(
        tmp_path / "evidence",
        dataset=dataset,
        scene_contract=scene,
        semantic_contract=semantic,
        geometry_source=MAST3R_PRIMARY_SFM_COVERAGE,
    )
    builder.add_file(
        "colmap_tracks",
        "raw_sfm_coverage_tracks",
        tracks,
        measurement="coverage only",
    )
    payload = builder.write()
    assert mast3r_is_geometry_authority(payload)
    assert sfm_coverage_tracks_enabled(payload)
    assert payload["geometry_authority"] == "mast3r_matcha_primary"
    assert payload["sfm_track_usage_mode"] == "coverage_only"
    assert payload["colmap_points_or_tracks_used"]
    assert "never controls camera admission" in payload["camera_container"]
