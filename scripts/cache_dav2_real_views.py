#!/usr/bin/env python
"""Build a restartable Depth Anything V2 ordinal cache for all real views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dav2-repository",
        type=Path,
        default=Path("/root/MAtCha/Depth-Anything-V2"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/root/MAtCha/Depth-Anything-V2/checkpoints/"
            "depth_anything_v2_vitl.pth"
        ),
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repository = args.dav2_repository.resolve()
    sys.path.insert(0, str(repository))
    from depth_anything_v2.dpt import DepthAnythingV2

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path
        for path in (args.dataset.resolve() / "images").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    configuration = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    }
    model = DepthAnythingV2(**configuration)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")
    )
    model = model.to(args.device).eval()
    completed = 0
    with torch.inference_mode():
        for image_path in tqdm(images, desc="DAV2 real-view cache"):
            destination = output / f"{image_path.stem}.npy"
            if destination.is_file():
                try:
                    cached = np.load(destination, mmap_mode="r")
                    if cached.ndim == 2 and cached.size:
                        completed += 1
                        continue
                except (OSError, ValueError):
                    pass
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read {image_path}")
            depth = model.infer_image(image, args.input_size)
            temporary = destination.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, np.asarray(depth, dtype=np.float16))
            temporary.replace(destination)
            completed += 1
    manifest = {
        "version": "dav2-real-view-cache-v1",
        "dataset": str(args.dataset.resolve()),
        "image_count": len(images),
        "completed_count": completed,
        "encoder": configuration,
        "input_size": args.input_size,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "ordinal_only": True,
    }
    (output / "dav2_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
