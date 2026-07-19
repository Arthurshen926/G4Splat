#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


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
    if strategy == "best_gain":
        return sorted(
            common,
            key=lambda name: (
                candidate_metrics["per_view"][name]["psnr"]
                - baseline_metrics["per_view"][name]["psnr"],
                name,
            ),
            reverse=True,
        )[:count]
    if strategy != "worst":
        raise ValueError(f"Unknown comparison selection strategy: {strategy}")

    worst = sorted(common, key=lambda name: candidate_metrics["per_view"][name]["psnr"])
    regressions = sorted(
        common,
        key=lambda name: (
            candidate_metrics["per_view"][name]["psnr"]
            - baseline_metrics["per_view"][name]["psnr"]
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
) -> None:
    baseline_metrics = json.loads((baseline_dir / "rgb_metrics.json").read_text())
    candidate_metrics = json.loads((candidate_dir / "rgb_metrics.json").read_text())
    selected = select_views(baseline_metrics, candidate_metrics, count, strategy=selection)

    tile_size = (480, 270)
    label_height = 42
    headers = ("GT", baseline_label, candidate_label, f"{candidate_label} absolute error x4")
    sheet = Image.new(
        "RGB",
        (tile_size[0] * len(headers), (tile_size[1] + label_height) * (len(selected) + 1)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for column, header in enumerate(headers):
        draw.text((column * tile_size[0] + 8, 12), header, fill="black")

    for row, name in enumerate(selected, start=1):
        y = row * (tile_size[1] + label_height)
        gt = load_rgb(candidate_dir / "gt" / name, tile_size)
        baseline = load_rgb(baseline_dir / "renders" / name, tile_size)
        candidate = load_rgb(candidate_dir / "renders" / name, tile_size)
        images = (gt, baseline, candidate, error_image(candidate, gt))
        for column, image in enumerate(images):
            sheet.paste(image, (column * tile_size[0], y))

        baseline_psnr = baseline_metrics["per_view"][name]["psnr"]
        candidate_psnr = candidate_metrics["per_view"][name]["psnr"]
        delta = candidate_psnr - baseline_psnr
        label = (
            f"{name}  {baseline_label} {baseline_psnr:.2f}  "
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
        choices=("worst", "best_gain"),
        default="worst",
        help="Choose the lowest-quality/regressed views or the largest PSNR gains.",
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
    )


if __name__ == "__main__":
    main()
