#!/usr/bin/env python
"""Visualize independent foliage-volume support on selected Cambridge views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup
from outdoor.foliage_geometry import (
    _project,
    _camera_record,
    read_cameras_binary,
    read_images_binary,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--volume-npz", type=Path, required=True)
    parser.add_argument("--reference-render-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-indices", type=int, nargs="+", required=True)
    return parser.parse_args()


def _tree_visual(image, tree):
    array = np.asarray(image).copy()
    array[tree] = (
        0.4 * array[tree] + 0.6 * np.asarray([20, 220, 80])
    ).astype(np.uint8)
    return Image.fromarray(array)


def _seed_visual(image, u, v, valid, inside, source_width, source_height):
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    scale_x = image.width / source_width
    scale_y = image.height / source_height
    for column, row, accepted in zip(u[valid], v[valid], inside[valid]):
        x, y = float(column * scale_x), float(row * scale_y)
        color = (30, 240, 90, 105) if accepted else (255, 40, 30, 165)
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=color)
    return overlay.convert("RGB")


def _label(image, text):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 24), fill=(0, 0, 0))
    draw.text((7, 6), text, fill=(255, 255, 255))
    return result


def main():
    args = _parse_args()
    source = args.source_path.resolve()
    sparse = source / "sparse" / "0"
    cameras = read_cameras_binary(sparse / "cameras.bin")
    images = read_images_binary(sparse / "images.bin")
    image_by_name = {image["name"]: image for image in images.values()}
    names = sorted(path.name for path in (source / "images").iterdir() if path.is_file())
    masks = CambridgeMaskLookup(source, args.tree_mask_pickle.resolve())
    volume = np.load(args.volume_npz.resolve())
    centers = volume["centers"].astype(np.float64)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    report = []

    for index in args.image_indices:
        image_name = names[index]
        camera_record = _camera_record(
            image_by_name[image_name],
            cameras[image_by_name[image_name]["camera_id"]],
            masks,
        )
        source_image = Image.open(source / "images" / image_name).convert("RGB")
        display = source_image.resize(
            (source_image.width // 2, source_image.height // 2),
            Image.Resampling.LANCZOS,
        )
        keep = camera_record["tree_keep_mask"]
        tree = np.asarray(
            Image.fromarray((~keep).astype(np.uint8) * 255).resize(
                display.size, Image.Resampling.NEAREST
            )
        ) > 0
        u, v, depth, valid = _project(centers, camera_record)
        safe_u = np.where(valid, u, 0)
        safe_v = np.where(valid, v, 0)
        mask_rows = np.clip(
            np.floor(safe_v / camera_record["height"] * keep.shape[0]).astype(int),
            0,
            keep.shape[0] - 1,
        )
        mask_columns = np.clip(
            np.floor(safe_u / camera_record["width"] * keep.shape[1]).astype(int),
            0,
            keep.shape[1] - 1,
        )
        inside = valid & (~keep[mask_rows, mask_columns])
        precision = float(inside.sum() / max(valid.sum(), 1))
        reference = (
            Image.open(args.reference_render_dir / f"{index:05d}.png").convert("RGB")
            if args.reference_render_dir
            else display
        )
        reference = reference.resize(display.size, Image.Resampling.LANCZOS)
        panels = [
            _label(display, f"GT {index:05d} {image_name}"),
            _label(reference, "stable 68k render"),
            _label(_tree_visual(display, tree), "tree semantic region (green)"),
            _label(
                _seed_visual(
                    display,
                    u,
                    v,
                    valid,
                    inside,
                    camera_record["width"],
                    camera_record["height"],
                ),
                f"3D seed centers: green=in tree, red=outside ({precision:.3f})",
            ),
        ]
        row = Image.new("RGB", (display.width * len(panels), display.height))
        for panel_index, panel in enumerate(panels):
            row.paste(panel, (panel_index * display.width, 0))
        row.save(output / f"{index:05d}_projection.png")
        rows.append(row)
        report.append(
            {
                "index": int(index),
                "image_name": image_name,
                "visible_seed_centers": int(valid.sum()),
                "tree_seed_centers": int(inside.sum()),
                "center_inside_tree_fraction": precision,
                "minimum_visible_depth": float(depth[valid].min()) if valid.any() else None,
                "median_visible_depth": float(np.median(depth[valid])) if valid.any() else None,
            }
        )
    contact = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
    offset = 0
    for row in rows:
        contact.paste(row, (0, offset))
        offset += row.height
    contact.save(output / "00408_00410_foliage_projection_contact.png")
    (output / "projection_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
