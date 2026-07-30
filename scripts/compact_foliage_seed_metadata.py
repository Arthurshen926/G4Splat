#!/usr/bin/env python
"""Migrate a valid initialization to compact foliage camera metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import sha256_file  # noqa: E402
from outdoor.role_aware_initialization import (  # noqa: E402
    INITIALIZATION_VERSION,
    _compact_padded_camera_metadata,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source_initialization.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    source_manifest_path = source / "initialization_manifest.json"
    source_foliage_path = source / "foliage_seed_gaussians.pth"
    source_summary_path = source / "foliage_seed_gaussians.json"
    for path in (
        source_manifest_path,
        source_foliage_path,
        source_summary_path,
        source / "surface_seed.npz",
        source / "surface_seed.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    output.mkdir(parents=True)
    for name in ("surface_seed.npz", "surface_seed.json"):
        shutil.copy2(source / name, output / name)

    payload = torch.load(source_foliage_path, map_location="cpu")
    numpy_metadata = {
        key: payload[key].numpy()
        for key in (
            "support_camera_ids",
            "support_view_count",
            "observation_camera_ids",
            "observation_uv",
            "observation_depth",
        )
    }
    storage = _compact_padded_camera_metadata(numpy_metadata)
    for key in (
        "support_camera_ids",
        "observation_camera_ids",
        "observation_uv",
        "observation_depth",
    ):
        payload[key] = torch.from_numpy(numpy_metadata[key])
    payload["geometry_version"] = (
        str(payload.get("geometry_version", "unknown"))
        + "_compact_camera_slots_v1"
    )
    payload.setdefault("audit", {})["protocol"] = INITIALIZATION_VERSION
    payload["audit"]["camera_metadata_storage"] = storage
    foliage_path = output / "foliage_seed_gaussians.pth"
    torch.save(payload, foliage_path)

    summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    summary["protocol"] = INITIALIZATION_VERSION
    summary["camera_metadata_storage"] = storage
    summary["migrated_from"] = {
        "initialization": str(source),
        "foliage_seed_sha256": sha256_file(source_foliage_path),
        "migration": "stable_rowwise_padding_removal_only",
    }
    (output / "foliage_seed_gaussians.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    manifest["version"] = INITIALIZATION_VERSION
    manifest["surface_seed"] = str(output / "surface_seed.npz")
    manifest["foliage_seed"] = str(foliage_path)
    manifest["foliage"] = summary
    manifest.setdefault("initialization_contract", {})[
        "camera_metadata_storage"
    ] = "stable_rowwise_compact_valid_slots_v1"
    manifest["migrated_from"] = {
        "initialization": str(source),
        "manifest_sha256": sha256_file(source_manifest_path),
        "foliage_seed_sha256": sha256_file(source_foliage_path),
        "render_and_training_equivalent": True,
    }
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "version": INITIALIZATION_VERSION,
                "foliage_seed_sha256": sha256_file(foliage_path),
                **storage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
