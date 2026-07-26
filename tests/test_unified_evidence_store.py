import json
from pathlib import Path

import pytest

from outdoor.evidence_store import (
    EVIDENCE_STORE_VERSION,
    EvidenceStoreBuilder,
    artifact_path,
    load_evidence_store,
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

