#!/usr/bin/env python3
"""Build a read-only-input adapter for clean Cambridge external controls.

The released ULF-Loc and STDLoc Cambridge scripts expect nested camera names
and a ``<images>/masks.pkl`` file.  Our strict all-train G4 staging has exactly
the 1,487 requested cameras but deliberately flattens the filenames and keeps
the external mask pickle outside ``images/`` so regular G4 input audits cannot
mistake it for an RGB frame.  This tool creates a *new* control input tree:

* every staged RGB frame and sparse model is linked, not copied by default;
* the mask pickle is reduced to exactly the staged cameras and re-keyed from
  the original nested name to the staged COLMAP filename;
* a manifest records the input/mask provenance.

For a strict cross-renderer control, ``--canonical-size`` instead writes the
same quantized bilinear RGB targets for every implementation.  This avoids
silently comparing G4's PIL resize path against the released ULF/STDLoc
``torch.interpolate(..., bilinear, align_corners=False)`` path.

No ULF-Loc or STDLoc source file is modified.  The resulting directory may be
passed to either clean repository with ``--images images``; because it has no
``dataset_test.txt``, all staged cameras are treated as training cameras.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def name_set_sha256(names: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def _source_images(source: Path) -> set[str]:
    image_dir = source / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing staged images directory: {image_dir}")
    names = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not names:
        raise RuntimeError(f"No RGB frames in {image_dir}")
    return names


def _read_mapping(source: Path) -> dict[str, str]:
    path = source / "name_mapping.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing staged name mapping: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"Invalid or empty name mapping: {path}")
    mapping = {str(staged): str(original) for staged, original in raw.items()}
    if len(mapping) != len(raw):
        raise RuntimeError(f"Duplicate staged name in mapping: {path}")
    return mapping


def _mask_shape(value: object) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        raise RuntimeError(f"Mask channel has no HxW shape: {type(value)!r}")
    return int(shape[-2]), int(shape[-1])


def _link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite adapter path: {destination}")
    destination.symlink_to(source)


def _write_canonical_bilinear_rgb(
    source: Path,
    destination: Path,
    *,
    size: tuple[int, int],
) -> tuple[int, int]:
    """Write ULF/STDLoc's bilinear target on a shared 8-bit image grid.

    The released external camera loader first converts an RGB image to a
    float tensor and then calls ``F.interpolate`` with
    ``align_corners=False``.  A shared PNG necessarily quantizes that float
    result once, but it makes the *stored target bytes* identical for G4,
    ULF-Loc, and STDLoc.  That is preferable to leaving their image-resize
    kernels as an uncontrolled experimental variable.
    """
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        source_size = tuple(int(value) for value in image.size)
        values = np.asarray(image, dtype=np.uint8).copy()
    rgb = torch.from_numpy(values).permute(2, 0, 1).float().div_(255.0)
    width, height = size
    resized = F.interpolate(
        rgb[None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    encoded = (
        resized.mul(255.0)
        .round()
        .clamp_(0.0, 255.0)
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(encoded, mode="RGB").save(destination)
    return source_size


def build_external_control(
    *,
    staged_dataset: Path,
    source_mask_pickle: Path,
    output: Path,
    canonical_size: tuple[int, int] | None = None,
) -> dict:
    """Create the immutable external-control input and return its manifest."""
    staged_dataset = staged_dataset.resolve()
    source_mask_pickle = source_mask_pickle.resolve()
    output = output.resolve()
    if canonical_size is not None:
        if len(canonical_size) != 2 or any(int(value) <= 0 for value in canonical_size):
            raise ValueError(f"canonical_size must be positive (width, height), got {canonical_size}")
        canonical_size = tuple(int(value) for value in canonical_size)
    if output.exists():
        raise FileExistsError(f"Refusing to mix an adapter with existing path: {output}")
    if not source_mask_pickle.is_file():
        raise FileNotFoundError(f"Mask pickle does not exist: {source_mask_pickle}")
    sparse = staged_dataset / "sparse"
    if not sparse.is_dir():
        raise FileNotFoundError(f"Missing staged sparse model: {sparse}")

    staged_names = _source_images(staged_dataset)
    mapping = _read_mapping(staged_dataset)
    if set(mapping) != staged_names:
        raise RuntimeError(
            "The staged RGB filenames and name_mapping.json disagree: "
            f"images={len(staged_names)}, mapping={len(mapping)}"
        )

    with source_mask_pickle.open("rb") as handle:
        original_masks = pickle.load(handle)
    if not isinstance(original_masks, dict):
        raise RuntimeError(f"Mask pickle is not a dictionary: {source_mask_pickle}")

    selected_masks: dict[str, tuple] = {}
    raw_shapes: Counter[tuple[int, int]] = Counter()
    missing: list[str] = []
    invalid: list[str] = []
    for staged_name in sorted(staged_names):
        original_name = mapping[staged_name]
        channels = original_masks.get(original_name)
        if channels is None:
            missing.append(f"{staged_name}->{original_name}")
            continue
        if not isinstance(channels, (tuple, list)) or len(channels) < 3:
            invalid.append(f"{staged_name}->{original_name}")
            continue
        first_three = tuple(channels[:3])
        try:
            raw_shapes.update(_mask_shape(channel) for channel in first_three)
        except RuntimeError:
            invalid.append(f"{staged_name}->{original_name}")
            continue
        selected_masks[staged_name] = first_three
    if missing or invalid or len(selected_masks) != len(staged_names):
        raise RuntimeError(
            "Cannot construct a complete all-train external mask adapter: "
            f"selected={len(selected_masks)}, expected={len(staged_names)}, "
            f"missing={missing[:3]}, invalid={invalid[:3]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        images = temp_parent / "images"
        images.mkdir()
        source_image_sizes: Counter[tuple[int, int]] = Counter()
        for staged_name in sorted(staged_names):
            source_image = staged_dataset / "images" / staged_name
            destination_image = images / staged_name
            if canonical_size is None:
                _link(source_image, destination_image)
            else:
                source_image_sizes.update(
                    [_write_canonical_bilinear_rgb(
                        source_image,
                        destination_image,
                        size=canonical_size,
                    )]
                )
        with (images / "masks.pkl").open("wb") as handle:
            pickle.dump(selected_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        _link(sparse, temp_parent / "sparse")
        _link(staged_dataset / "name_mapping.json", temp_parent / "name_mapping.json")
        for optional_name in ("source_split.txt", "qc_manifest.json"):
            optional = staged_dataset / optional_name
            if optional.is_file():
                _link(optional, temp_parent / optional_name)

        manifest = {
            "contract": "clean_external_cambridge_all_train_mask_adapter_v2",
            "staged_dataset": str(staged_dataset),
            "source_mask_pickle": str(source_mask_pickle),
            "source_mask_pickle_sha256": file_sha256(source_mask_pickle),
            "staged_rgb_count": len(staged_names),
            "staged_rgb_name_set_sha256": name_set_sha256(staged_names),
            "selected_mask_count": len(selected_masks),
            "selected_mask_channels": [0, 1, 2],
            "original_mask_key_count": len(original_masks),
            "raw_mask_shape_counts_per_channel": {
                f"{height}x{width}": count
                for (height, width), count in sorted(raw_shapes.items())
            },
            "all_train_split": "no_dataset_test_txt",
            "rgb_target_storage": (
                "linked_staged_rgb"
                if canonical_size is None
                else "shared_uint8_torch_bilinear_align_corners_false"
            ),
            "canonical_image_size_wh": (
                None if canonical_size is None else list(canonical_size)
            ),
            "canonical_source_size_counts_wh": {
                f"{width}x{height}": count
                for (width, height), count in sorted(source_image_sizes.items())
            },
            "canonical_quantization": (
                None
                if canonical_size is None
                else "round(float_rgb_times_255)_uint8_png; max_per_channel_error<=0.5/255"
            ),
        }
        (temp_parent / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temp_parent.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-dataset", type=Path, required=True)
    parser.add_argument("--source-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--canonical-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help=(
            "Write shared ULF/STDLoc-style bilinear RGB targets at this size "
            "instead of linking the staged source RGB frames."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_external_control(
                staged_dataset=args.staged_dataset,
                source_mask_pickle=args.source_mask_pickle,
                output=args.output,
                canonical_size=None if args.canonical_size is None else tuple(args.canonical_size),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
