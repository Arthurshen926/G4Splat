#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def load_rgb_crop(
    path: Path, crop_box: tuple[int, int, int, int], size: tuple[int, int]
) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").crop(crop_box).resize(
            size, resample=Image.Resampling.LANCZOS
        )


def tree_crop_box(tree_keep_mask: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Choose a fixed-aspect crop with the greatest tree-pixel coverage."""
    width, height = image_size
    tree_keep_mask = np.asarray(tree_keep_mask).squeeze().astype(bool)
    tree = Image.fromarray((~tree_keep_mask).astype(np.uint8) * 255).resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    tree_array = np.asarray(tree) > 0
    if not tree_array.any():
        raise ValueError("Cannot create a tree crop from an empty tree mask")

    crop_width = max(1, round(width * 0.55))
    crop_height = max(1, round(height * 0.55))
    integral = np.pad(tree_array.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    x_candidates = sorted(
        set(np.linspace(0, width - crop_width, num=17, dtype=np.int64).tolist())
    )
    y_candidates = sorted(
        set(np.linspace(0, height - crop_height, num=11, dtype=np.int64).tolist())
    )
    best_score = -1
    best_box = (0, 0, crop_width, crop_height)
    for top in y_candidates:
        bottom = top + crop_height
        for left in x_candidates:
            right = left + crop_width
            score = (
                integral[bottom, right]
                - integral[top, right]
                - integral[bottom, left]
                + integral[top, left]
            )
            if score > best_score:
                best_score = int(score)
                best_box = (int(left), int(top), int(right), int(bottom))
    return best_box


def error_image(render: Image.Image, target: Image.Image) -> Image.Image:
    render_array = np.asarray(render, dtype=np.float32) / 255.0
    target_array = np.asarray(target, dtype=np.float32) / 255.0
    error = np.abs(render_array - target_array).mean(axis=2)
    error = np.clip(error * 4.0, 0.0, 1.0)
    heat = np.stack([error, np.sqrt(error), 1.0 - error], axis=2)
    return Image.fromarray(np.uint8(np.clip(heat, 0.0, 1.0) * 255.0), mode="RGB")


def select_views(
    baseline_metrics: dict,
    candidate_metrics: dict,
    count: int,
    strategy: str = "worst",
) -> list[str]:
    common = sorted(set(baseline_metrics["per_view"]) & set(candidate_metrics["per_view"]))
    tree_strategy = strategy.startswith("tree_")
    if tree_strategy:
        common = [
            name
            for name in common
            if "tree_static"
            in baseline_metrics["per_view"][name].get("tree_stratified", {})
            and "tree_static"
            in candidate_metrics["per_view"][name].get("tree_stratified", {})
        ]
        if not common:
            raise ValueError("No common views contain non-empty tree-static pixels")

    def view_psnr(metrics: dict, name: str) -> float:
        if tree_strategy:
            return metrics["per_view"][name]["tree_stratified"]["tree_static"]["psnr"]
        return metrics["per_view"][name]["psnr"]

    if strategy in {"best_gain", "tree_best_gain"}:
        return sorted(
            common,
            key=lambda name: (
                view_psnr(candidate_metrics, name)
                - view_psnr(baseline_metrics, name),
                name,
            ),
            reverse=True,
        )[:count]
    if strategy not in {"worst", "tree_worst"}:
        raise ValueError(f"Unknown comparison selection strategy: {strategy}")

    worst = sorted(common, key=lambda name: view_psnr(candidate_metrics, name))
    regressions = sorted(
        common,
        key=lambda name: (
            view_psnr(candidate_metrics, name)
            - view_psnr(baseline_metrics, name)
        ),
    )
    selected = []
    worst_quota = (count + 1) // 2
    for name in worst:
        if name not in selected:
            selected.append(name)
        if len(selected) >= worst_quota:
            break
    for name in regressions:
        if name not in selected:
            selected.append(name)
        if len(selected) >= count:
            return selected
    for name in worst:
        if name not in selected:
            selected.append(name)
        if len(selected) >= count:
            return selected
    return selected


def make_comparison(
    baseline_dir: Path,
    candidate_dir: Path,
    output: Path,
    count: int,
    baseline_label: str = "MAtCha retained_v2",
    candidate_label: str = "G4Splat",
    selection: str = "worst",
    tree_crop: bool = False,
    tree_mask_pickle: Optional[Path] = None,
) -> None:
    baseline_metrics = json.loads((baseline_dir / "rgb_metrics.json").read_text())
    candidate_metrics = json.loads((candidate_dir / "rgb_metrics.json").read_text())
    selected = select_views(baseline_metrics, candidate_metrics, count, strategy=selection)
    tree_masks = None
    if tree_crop:
        if tree_mask_pickle is None:
            raise ValueError("--tree-crop requires --tree-mask-pickle")
        with tree_mask_pickle.open("rb") as handle:
            tree_masks = pickle.load(handle)

    tile_size = (480, 270)
    label_height = 42
    headers = ("GT", baseline_label, candidate_label, f"{candidate_label} absolute error x4")
    sheet = Image.new(
        "RGB",
        (
            tile_size[0] * len(headers),
            label_height + (tile_size[1] + label_height) * len(selected),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, header in enumerate(headers):
        draw.text((column * tile_size[0] + 8, 12), header, fill="black")

    for row, name in enumerate(selected, start=1):
        y = label_height + (row - 1) * (tile_size[1] + label_height)
        source_paths = (
            candidate_dir / "gt" / name,
            baseline_dir / "renders" / name,
            candidate_dir / "renders" / name,
        )
        if tree_masks is None:
            gt, baseline, candidate = (
                load_rgb(path, tile_size) for path in source_paths
            )
        else:
            source_name = candidate_metrics["per_view"][name]["source_image"]
            tree_mask = tree_masks[source_name][3]
            if hasattr(tree_mask, "detach"):
                tree_mask = tree_mask.detach().cpu().numpy()
            with Image.open(source_paths[0]) as source_gt:
                crop_box = tree_crop_box(tree_mask, source_gt.size)
            gt, baseline, candidate = (
                load_rgb_crop(path, crop_box, tile_size)
                for path in source_paths
            )
        images = (gt, baseline, candidate, error_image(candidate, gt))
        for column, image in enumerate(images):
            sheet.paste(image, (column * tile_size[0], y))

        metric_scope = (
            ("tree_stratified", "tree_static")
            if selection.startswith("tree_")
            else None
        )
        if metric_scope is None:
            baseline_psnr = baseline_metrics["per_view"][name]["psnr"]
            candidate_psnr = candidate_metrics["per_view"][name]["psnr"]
            scope_label = "full"
        else:
            baseline_psnr = baseline_metrics["per_view"][name][metric_scope[0]][
                metric_scope[1]
            ]["psnr"]
            candidate_psnr = candidate_metrics["per_view"][name][metric_scope[0]][
                metric_scope[1]
            ]["psnr"]
            scope_label = "tree-static"
        delta = candidate_psnr - baseline_psnr
        label = (
            f"{name} [{scope_label}]  {baseline_label} {baseline_psnr:.2f}  "
            f"{candidate_label} {candidate_psnr:.2f}  delta {delta:+.2f} dB"
        )
        draw.text((8, y + tile_size[1] + 10), label, fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an objective MAtCha/G4 heldout comparison sheet.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--baseline-label", default="MAtCha retained_v2")
    parser.add_argument("--candidate-label", default="G4Splat")
    parser.add_argument(
        "--selection",
        choices=("worst", "best_gain", "tree_worst", "tree_best_gain"),
        default="worst",
        help=(
            "Choose full-image or tree-static lowest-quality/regressed views, "
            "or the corresponding largest PSNR gains."
        ),
    )
    parser.add_argument(
        "--tree-crop",
        action="store_true",
        help="Crop every selected view around the densest tree-mask region.",
    )
    parser.add_argument(
        "--tree-mask-pickle",
        type=Path,
        default=None,
        help="Extended Cambridge mask pickle used by --tree-crop.",
    )
    args = parser.parse_args()
    make_comparison(
        args.baseline,
        args.candidate,
        args.output,
        max(args.count, 1),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        selection=args.selection,
        tree_crop=args.tree_crop,
        tree_mask_pickle=args.tree_mask_pickle,
    )


if __name__ == "__main__":
    main()
