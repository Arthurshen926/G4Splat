#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import combine_masks, load_mask_dict


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return tf.to_tensor(Image.open(path).convert("RGB")).to(device)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def global_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match the compact metric used by the retained_v2 Cambridge reports."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = prediction.mean(), target.mean()
    sigma_x = ((prediction - mu_x) ** 2).mean()
    sigma_y = ((target - mu_y) ** 2).mean()
    sigma_xy = ((prediction - mu_x) * (target - mu_y)).mean()
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )


def metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    valid_ratio = 1.0
    if mask is not None:
        if tuple(mask.shape) != tuple(prediction.shape[-2:]):
            raise RuntimeError(
                f"Mask shape {tuple(mask.shape)} does not match image {tuple(prediction.shape[-2:])}"
            )
        if not bool(mask.any()):
            raise RuntimeError("Semantic mask removed every pixel")
        valid_ratio = float(mask.float().mean().cpu())
        prediction = prediction[:, mask]
        target = target[:, mask]

    residual = prediction - target
    return {
        "psnr": float(psnr(prediction, target).cpu()),
        "ssim": float(global_ssim(prediction, target).cpu()),
        "mae": float(residual.abs().mean().cpu()),
        "rmse": float(residual.square().mean().sqrt().cpu()),
        "valid_pixel_ratio": valid_ratio,
    }


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = ("psnr", "ssim", "mae", "rmse", "valid_pixel_ratio")
    return {
        key: float(torch.tensor([item[key] for item in items]).mean())
        for key in keys
    }


def render_source_mapping(dataset_path: Path, render_names: list[str]) -> dict[str, str]:
    mapping_path = dataset_path / "name_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    staged_names = sorted(mapping, key=lambda name: Path(name).stem)
    if len(staged_names) != len(render_names):
        raise RuntimeError(
            f"Render count {len(render_names)} differs from dataset count {len(staged_names)}"
        )
    return {
        render_name: mapping[staged_names[index]]
        for index, render_name in enumerate(sorted(render_names))
    }


def evaluate(
    render_output: Path,
    dataset_path: Path,
    mask_pickle: Path,
    device: torch.device,
) -> dict:
    renders_dir, gt_dir = render_output / "renders", render_output / "gt"
    names = sorted(
        path.name
        for path in renders_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not names:
        raise RuntimeError(f"No rendered images found in {renders_dir}")

    masks = load_mask_dict(mask_pickle)
    source_by_render = render_source_mapping(dataset_path, names)
    mask_sets = {
        "dynamic_valid": [0, 2],
        "static_valid": [0, 1, 2],
    }
    raw_items: list[dict[str, float]] = []
    masked_items = {name: [] for name in mask_sets}
    per_view: dict[str, dict] = {}

    for name in tqdm(names, desc="RGB metrics"):
        prediction = load_rgb(renders_dir / name, device)
        target = load_rgb(gt_dir / name, device)
        if prediction.shape != target.shape:
            raise RuntimeError(f"Shape mismatch for {name}: {prediction.shape} vs {target.shape}")

        raw = metrics(prediction, target)
        raw_items.append(raw)
        source_name = source_by_render[name]
        view_result = {**raw, "source_image": source_name, "masked": {}}
        for set_name, indices in mask_sets.items():
            keep = combine_masks(masks[source_name], indices, prediction.shape[-2:], device)
            masked = metrics(prediction, target, keep)
            masked_items[set_name].append(masked)
            view_result["masked"][set_name] = masked
        per_view[name] = view_result

    raw_mean = mean_metrics(raw_items)
    raw_mean.pop("valid_pixel_ratio")
    return {
        "num_images": len(names),
        "mean": raw_mean,
        "masked": {
            name: {
                "mask_indices": mask_sets[name],
                "mean": mean_metrics(items),
            }
            for name, items in masked_items.items()
        },
        "per_view": per_view,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Cambridge render directory.")
    parser.add_argument("render_output", type=Path)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = evaluate(args.render_output, args.dataset_path, args.mask_pickle, device)
    output = args.output or args.render_output / "rgb_metrics.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "raw": result["mean"],
        **{name: value["mean"] for name, value in result["masked"].items()},
    }, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
