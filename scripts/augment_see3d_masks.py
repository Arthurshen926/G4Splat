#!/usr/bin/env python3
"""Conservatively add no-reference artifact regions to See3D edit masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
import torch


def load_detector(detector_repo: Path):
    detector_repo = detector_repo.expanduser().resolve()
    if not detector_repo.is_dir():
        raise FileNotFoundError(detector_repo)
    if str(detector_repo) not in sys.path:
        sys.path.append(str(detector_repo))
    from valid_support_mask import NoReferenceValidSupportMaskBuilder

    return NoReferenceValidSupportMaskBuilder()


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def read_resized_mask(path: Path, size: tuple[int, int], *, invert: bool = False) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    mask = np.asarray(image, dtype=np.uint8) > 127
    return ~mask if invert else mask


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.uint8(mask) * 255, mode="L").save(path)


def filter_components(
    mask: np.ndarray,
    *,
    min_area_fraction: float,
    max_component_fraction: float,
) -> tuple[np.ndarray, list[dict]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(mask), connectivity=8
    )
    total = int(mask.size)
    output = np.zeros_like(mask, dtype=bool)
    components = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        fraction = area / total
        accepted = min_area_fraction <= fraction <= max_component_fraction
        if accepted:
            output[labels == label] = True
        components.append(
            {
                "area": area,
                "area_fraction": fraction,
                "bbox_xywh": [x, y, width, height],
                "accepted": accepted,
            }
        )
    components.sort(key=lambda item: item["area"], reverse=True)
    return output, components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector_repo", type=Path, default=Path("/root/STDLoc"))
    parser.add_argument(
        "--external_mask_dir",
        type=Path,
        default=None,
        help="Optional masks for these exact rendered frames; combined with detector output.",
    )
    parser.add_argument(
        "--external_mask_pattern",
        default="{stem}.valid_mask.png",
        help="Filename template under external_mask_dir. Available field: {stem}.",
    )
    parser.add_argument(
        "--external_mask_is_valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat white external pixels as valid (default); invert for white=artifact masks.",
    )
    parser.add_argument("--detector_scale", type=float, default=0.5)
    parser.add_argument("--min_component_fraction", type=float, default=0.002)
    parser.add_argument("--max_component_fraction", type=float, default=0.15)
    parser.add_argument("--artifact_dilate_radius", type=int, default=3)
    parser.add_argument("--max_edit_fraction", type=float, default=0.35)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and args.replace:
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    detector = load_detector(args.detector_repo)

    render_paths = sorted(input_root.glob("ori_warp_frame*.png"))
    if not render_paths:
        raise FileNotFoundError(f"No ori_warp_frame*.png in {input_root}")
    records = []
    for render_path in render_paths:
        stem = render_path.stem.removeprefix("ori_warp_frame")
        original_known = read_mask(input_root / f"mask_frame{stem}.png")
        render = read_rgb(render_path)
        height, width = render.shape[:2]
        scaled = cv2.resize(
            render,
            (
                max(32, int(round(width * args.detector_scale))),
                max(32, int(round(height * args.detector_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        result = detector.build(torch.from_numpy(scaled.copy()).permute(2, 0, 1))
        detector_valid = cv2.resize(
            np.uint8(result.valid_mask.detach().cpu().numpy()),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        score = cv2.resize(
            result.invalid_score.detach().cpu().numpy().astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        external_path = None
        external_artifact = np.zeros_like(original_known)
        if args.external_mask_dir is not None:
            external_path = args.external_mask_dir.expanduser().resolve() / args.external_mask_pattern.format(
                stem=stem
            )
            if not external_path.is_file():
                raise FileNotFoundError(
                    f"Missing external mask for frame {stem}: {external_path}. "
                    "External masks must correspond to these exact camera renders."
                )
            external_artifact = read_resized_mask(
                external_path,
                (width, height),
                invert=bool(args.external_mask_is_valid),
            ) & original_known

        detector_artifact = original_known & ~detector_valid
        candidate = detector_artifact | external_artifact
        artifact, components = filter_components(
            candidate,
            min_area_fraction=float(args.min_component_fraction),
            max_component_fraction=float(args.max_component_fraction),
        )
        if args.artifact_dilate_radius > 0 and np.any(artifact):
            radius = int(args.artifact_dilate_radius)
            kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
            artifact = cv2.dilate(np.uint8(artifact), kernel).astype(bool) & original_known

        proposed_known = original_known & ~artifact
        proposed_edit_fraction = float((~proposed_known).mean())
        accepted = proposed_edit_fraction <= float(args.max_edit_fraction)
        known = proposed_known if accepted else original_known
        artifact = artifact if accepted else np.zeros_like(artifact)
        condition = render.copy()
        condition[~known] = 0

        Image.fromarray(render, mode="RGB").save(output_root / f"ori_warp_frame{stem}.png")
        Image.fromarray(condition, mode="RGB").save(output_root / f"warp_frame{stem}.png")
        save_mask(output_root / f"mask_frame{stem}.png", known)
        save_mask(output_root / f"original_known_frame{stem}.png", original_known)
        save_mask(output_root / f"artifact_edit_frame{stem}.png", artifact)
        save_mask(output_root / f"external_artifact_frame{stem}.png", external_artifact)
        Image.fromarray(np.uint8(np.clip(score, 0.0, 1.0) * 255), mode="L").save(
            output_root / f"artifact_score_frame{stem}.png"
        )
        records.append(
            {
                "index": int(stem),
                "original_edit_fraction": float((~original_known).mean()),
                "artifact_edit_fraction": float(artifact.mean()),
                "detector_artifact_fraction": float(detector_artifact.mean()),
                "external_artifact_fraction": float(external_artifact.mean()),
                "external_mask_path": str(external_path) if external_path is not None else None,
                "combined_edit_fraction": float((~known).mean()),
                "artifact_gate_accepted": accepted,
                "detector_summary": result.summary,
                "components": components,
            }
        )

    report = {
        "version": 2,
        "mode": "visibility_plus_explicit_artifact_edit_mask",
        "input": str(input_root),
        "detector_repo": str(args.detector_repo.expanduser().resolve()),
        "views": records,
    }
    (output_root / "repair_mask_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
