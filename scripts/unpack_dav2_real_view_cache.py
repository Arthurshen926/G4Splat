#!/usr/bin/env python
"""Materialize the prepared packed DAV2 cache as restartable per-view arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.packed_cache.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(source, map_location="cpu")
    names = payload.get("image_names")
    depths = payload.get("depths")
    if (
        payload.get("version") != 1
        or not isinstance(names, list)
        or not isinstance(depths, list)
        or len(names) != len(depths)
        or not names
    ):
        raise RuntimeError("Unsupported packed DAV2 real-view cache")
    completed = 0
    for name, depth in zip(names, depths):
        destination = output / f"{Path(str(name)).stem}.npy"
        if destination.is_file():
            try:
                value = np.load(destination, mmap_mode="r")
                if value.ndim == 2 and value.size:
                    completed += 1
                    continue
            except (OSError, ValueError):
                pass
        value = (
            depth.detach().cpu().numpy()
            if torch.is_tensor(depth)
            else np.asarray(depth)
        )
        value = np.asarray(value, dtype=np.float16).squeeze()
        if value.ndim != 2 or not np.isfinite(value).any():
            raise RuntimeError(f"Invalid DAV2 depth for {name}")
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value)
        temporary.replace(destination)
        completed += 1
    manifest = {
        "version": "dav2-real-view-cache-v2-unpacked",
        "packed_cache": str(source),
        "image_count": len(names),
        "completed_count": completed,
        "ordinal_only": True,
    }
    (output / "dav2_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
