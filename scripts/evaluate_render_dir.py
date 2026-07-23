#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
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

from matcha.cambridge_masks import combine_masks, load_mask_dict, tensor_to_resized_mask
from matcha.cambridge_training import ulfloc_masked_supervision


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
    valid_ratio = prediction.new_tensor(1.0)
    if mask is not None:
        if tuple(mask.shape) != tuple(prediction.shape[-2:]):
            raise RuntimeError(
                f"Mask shape {tuple(mask.shape)} does not match image {tuple(prediction.shape[-2:])}"
            )
        if not bool(mask.any()):
            raise RuntimeError("Semantic mask removed every pixel")
        valid_ratio = mask.float().mean()
        prediction = prediction[:, mask]
        target = target[:, mask]

    residual = prediction - target
    # Transfer all scalar statistics together.  A full Cambridge tree audit
    # computes six metric strata per view; synchronising CUDA separately for
    # every scalar otherwise dominates the 1,487-view evaluation.
    values = torch.stack(
        (
            psnr(prediction, target),
            global_ssim(prediction, target),
            residual.abs().mean(),
            residual.square().mean().sqrt(),
            valid_ratio,
        )
    ).detach().cpu().tolist()
    return dict(zip(("psnr", "ssim", "mae", "rmse", "valid_pixel_ratio"), values))


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = ("psnr", "ssim", "mae", "rmse", "valid_pixel_ratio")
    return {
        key: float(torch.tensor([item[key] for item in items]).mean())
        for key in keys
    }


def render_source_mapping(dataset_path: Path, render_names: list[str]) -> dict[str, str]:
    mapping_path = dataset_path / "name_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if len(render_names) != len(set(render_names)):
        raise RuntimeError("Render directory contains duplicate output filenames")

    # Normal evaluation writes the staged camera filename directly, including
    # when only a diagnostic subset was rendered.
    direct = {name: mapping[name] for name in render_names if name in mapping}
    unresolved = [name for name in render_names if name not in direct]
    if not unresolved:
        return direct

    # ``render.py --rgb_only`` deliberately streams the full Cambridge set
    # without retaining every camera tensor in memory.  Its output filenames
    # are five-digit *sorted-camera indices*, not source image stems.  Decode
    # those indices explicitly.  In particular, do not use the old
    # same-cardinality positional guess: an incomplete or foreign directory
    # could otherwise be silently associated with the wrong real RGB/mask.
    staged_names = sorted(mapping, key=lambda name: Path(name).stem)
    indexed: dict[str, str] = {}
    indexed_values: set[int] = set()
    for name in unresolved:
        stem = Path(name).stem
        if len(stem) != 5 or not stem.isascii() or not stem.isdecimal():
            indexed = {}
            break
        output_index = int(stem)
        if output_index in indexed_values or output_index >= len(staged_names):
            indexed = {}
            break
        indexed_values.add(output_index)
        indexed[name] = mapping[staged_names[output_index]]
    if indexed and len(indexed) == len(unresolved):
        direct.update(indexed)
        return direct

    by_stem: dict[str, str] = {}
    for staged_name, source_name in mapping.items():
        stem = Path(staged_name).stem
        if stem in by_stem:
            raise RuntimeError(f"Ambiguous staged image stem in name_mapping.json: {stem}")
        by_stem[stem] = source_name
    for name in list(unresolved):
        source_name = by_stem.get(Path(name).stem)
        if source_name is not None:
            direct[name] = source_name
    unresolved = [name for name in render_names if name not in direct]
    if not unresolved:
        return direct
    raise RuntimeError(
        "Could not map render filename(s) to staged cameras without guessing: "
        f"unresolved={unresolved[:3]}, renders={len(render_names)}, dataset={len(staged_names)}"
    )


def staged_mask_key_by_source(dataset_path: Path) -> dict[str, str]:
    """Build the unambiguous source-name -> staged-name side of the adapter map.

    Cambridge has two legitimate mask-pickle layouts.  The original
    ULF-Loc/STDLoc pickle is keyed by nested source names such as
    ``seq4/frame00037.png``.  The immutable all-train adapter deliberately
    reduces and rekeys that pickle by staged COLMAP names such as
    ``seq4__frame00037.png``.  Render metrics always retain the nested source
    name as their stable cross-renderer identity, so mask lookup must bridge
    these two namespaces rather than assuming that the metric identity is
    also the pickle key.
    """
    mapping_path = dataset_path / "name_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if not isinstance(mapping, dict):
        raise RuntimeError(f"Expected a mapping dictionary in {mapping_path}")
    by_source: dict[str, str] = {}
    for staged_name, source_name in mapping.items():
        staged_name = str(staged_name)
        source_name = str(source_name)
        previous = by_source.get(source_name)
        if previous is not None and previous != staged_name:
            raise RuntimeError(
                "name_mapping.json maps one source image to multiple staged names: "
                f"{source_name!r} -> {previous!r}, {staged_name!r}"
            )
        by_source[source_name] = staged_name
    return by_source


def mask_key_for_source(
    masks: dict,
    *,
    source_name: str,
    staged_by_source: dict[str, str],
    mask_pickle: Path,
) -> tuple[str, str]:
    """Resolve a rendered source identity to a raw or adapter mask key.

    Returns both the actual pickle key and the resolution mode so the output
    report can prove whether it evaluated raw nested masks or an equivalent
    staged adapter.  Prefer a direct source key when both layouts happen to
    be present; this avoids changing historical raw-pickle evaluation.
    """
    source_name = str(source_name)
    if source_name in masks:
        return source_name, "source_key"
    staged_name = staged_by_source.get(source_name)
    if staged_name is not None and staged_name in masks:
        return staged_name, "staged_key"
    raise KeyError(
        f"{source_name!r} from rendered data is missing from {mask_pickle}; "
        "tried the nested source key and its staged adapter key "
        f"{staged_name!r}"
    )


def mask_resolution_audit(
    *,
    raw_mask_shape_counts: Counter,
    rendered_shape_counts: Counter,
    views_requiring_resampling: int,
) -> dict:
    """Make metric-protocol resolution changes explicit in the result JSON."""
    return {
        "raw_mask_shape_counts_per_channel": {
            f"{height}x{width}": count
            for (height, width), count in sorted(raw_mask_shape_counts.items())
        },
        "rendered_image_shape_counts": {
            f"{height}x{width}": count
            for (height, width), count in sorted(rendered_shape_counts.items())
        },
        "views_requiring_mask_resampling": int(views_requiring_resampling),
        "native_pixel_geometry": views_requiring_resampling == 0,
        "contract": (
            "native_mask_pixels"
            if views_requiring_resampling == 0
            else "nearest_resampled_mask_pixels"
        ),
    }


def tree_static_strata(
    static_keep: torch.Tensor,
    tree_keep: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split the ordinary static metric domain into building and tree pixels.

    Cambridge's historical ``static_valid`` domain excludes moving objects,
    sky, and the distortion rim, but trees are normally labelled as *stuff*.
    They can therefore dominate a seemingly architectural metric.  Keep the
    original metric unchanged and expose this diagnostic split alongside it.
    """
    if static_keep.shape != tree_keep.shape:
        raise RuntimeError(
            "Static and tree masks must have the same shape, got "
            f"{tuple(static_keep.shape)} and {tuple(tree_keep.shape)}"
        )
    return {
        "non_tree_static": static_keep & tree_keep,
        "tree_static": static_keep & (~tree_keep),
    }


def evaluate(
    render_output: Path,
    dataset_path: Path,
    mask_pickle: Path,
    device: torch.device,
    tree_mask_pickle: Optional[Path] = None,
    tree_mask_index: int = 3,
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
    tree_masks = load_mask_dict(tree_mask_pickle) if tree_mask_pickle is not None else None
    source_by_render = render_source_mapping(dataset_path, names)
    staged_by_source = staged_mask_key_by_source(dataset_path)
    mask_sets = {
        "dynamic_valid": [0, 2],
        "static_valid": [0, 1, 2],
    }
    raw_items: list[dict[str, float]] = []
    masked_items = {name: [] for name in mask_sets}
    ulfloc_legacy_items: list[dict[str, float]] = []
    tree_stratified_items = {
        "non_tree_static": [],
        "tree_static": [],
    }
    tree_empty_view_counts = Counter()
    per_view: dict[str, dict] = {}
    raw_mask_shape_counts: Counter = Counter()
    rendered_shape_counts: Counter = Counter()
    views_requiring_mask_resampling = 0
    mask_key_resolution_counts: Counter = Counter()
    tree_mask_key_resolution_counts: Counter = Counter()

    for name in tqdm(names, desc="RGB metrics"):
        prediction = load_rgb(renders_dir / name, device)
        target = load_rgb(gt_dir / name, device)
        if prediction.shape != target.shape:
            raise RuntimeError(f"Shape mismatch for {name}: {prediction.shape} vs {target.shape}")
        rendered_shape = tuple(int(size) for size in prediction.shape[-2:])
        rendered_shape_counts[rendered_shape] += 1

        raw = metrics(prediction, target)
        raw_items.append(raw)
        source_name = source_by_render[name]
        view_result = {**raw, "source_image": source_name, "masked": {}}
        mask_key, mask_key_mode = mask_key_for_source(
            masks,
            source_name=source_name,
            staged_by_source=staged_by_source,
            mask_pickle=mask_pickle,
        )
        mask_key_resolution_counts[mask_key_mode] += 1
        mask_tuple = masks[mask_key]
        for set_name, indices in mask_sets.items():
            keep = combine_masks(mask_tuple, indices, prediction.shape[-2:], device)
            masked = metrics(prediction, target, keep)
            masked_items[set_name].append(masked)
            view_result["masked"][set_name] = masked
        raw_mask_tuple = mask_tuple
        if len(raw_mask_tuple) >= 3:
            mask_shapes = [
                tuple(int(size) for size in raw_mask_tuple[index].shape[-2:])
                for index in (0, 1, 2)
            ]
            raw_mask_shape_counts.update(mask_shapes)
            if any(shape != rendered_shape for shape in mask_shapes):
                views_requiring_mask_resampling += 1
            object_mask, sky_mask, distortion_mask = (
                tensor_to_resized_mask(raw_mask_tuple[index], prediction.shape[-2:], device)
                for index in (0, 1, 2)
            )
            legacy_prediction, legacy_target, _ = ulfloc_masked_supervision(
                prediction,
                target,
                object_mask=object_mask,
                sky_mask=sky_mask,
                distortion_mask=distortion_mask,
            )
            legacy = metrics(legacy_prediction, legacy_target)
            ulfloc_legacy_items.append(legacy)
            view_result["ulfloc_legacy"] = legacy
        if tree_masks is not None:
            tree_key, tree_key_mode = mask_key_for_source(
                tree_masks,
                source_name=source_name,
                staged_by_source=staged_by_source,
                mask_pickle=tree_mask_pickle,
            )
            tree_mask_key_resolution_counts[tree_key_mode] += 1
            tree_tuple = tree_masks[tree_key]
            if tree_mask_index < 0 or tree_mask_index >= len(tree_tuple):
                raise RuntimeError(
                    f"Tree mask index {tree_mask_index} is outside tuple length "
                    f"{len(tree_tuple)} for {source_name}"
                )
            static_keep = combine_masks(
                mask_tuple, mask_sets["static_valid"], prediction.shape[-2:], device
            )
            tree_keep = tensor_to_resized_mask(
                tree_tuple[tree_mask_index], prediction.shape[-2:], device
            )
            view_result["tree_stratified"] = {}
            for stratum_name, stratum_mask in tree_static_strata(
                static_keep, tree_keep
            ).items():
                if bool(stratum_mask.any()):
                    stratum_metrics = metrics(prediction, target, stratum_mask)
                    tree_stratified_items[stratum_name].append(stratum_metrics)
                    view_result["tree_stratified"][stratum_name] = stratum_metrics
                else:
                    tree_empty_view_counts[stratum_name] += 1
        per_view[name] = view_result

    raw_mean = mean_metrics(raw_items)
    raw_mean.pop("valid_pixel_ratio")
    resolution_audit = mask_resolution_audit(
        raw_mask_shape_counts=raw_mask_shape_counts,
        rendered_shape_counts=rendered_shape_counts,
        views_requiring_resampling=views_requiring_mask_resampling,
    )
    result = {
        "num_images": len(names),
        "mean": raw_mean,
        "masked": {
            name: {
                "mask_indices": mask_sets[name],
                "mean": mean_metrics(items),
            }
            for name, items in masked_items.items()
        },
        "protocol": {
            "ulfloc_legacy": {
                "description": (
                    "ULF-Loc object/distortion masking followed by white sky target; "
                    "this records the legacy operation order, while exact upstream "
                    "pixel geometry additionally requires native mask resolution."
                ),
                "mean": mean_metrics(ulfloc_legacy_items),
            }
        },
        "mask_resolution_audit": resolution_audit,
        "mask_key_resolution": {
            "mask_pickle": str(mask_pickle),
            "counts": dict(sorted(mask_key_resolution_counts.items())),
        },
        "per_view": per_view,
    }
    if tree_masks is not None:
        result["tree_stratified"] = {
            "tree_mask_pickle": str(tree_mask_pickle),
            "tree_mask_index": int(tree_mask_index),
            "mask_key_resolution_counts": dict(
                sorted(tree_mask_key_resolution_counts.items())
            ),
            **{
                stratum_name: {
                    "mean": (
                        mean_metrics(items) if items else None
                    ),
                    "evaluated_view_count": len(items),
                    "empty_view_count": int(tree_empty_view_counts[stratum_name]),
                }
                for stratum_name, items in tree_stratified_items.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Cambridge render directory.")
    parser.add_argument("render_output", type=Path)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument(
        "--tree-mask-pickle",
        type=Path,
        default=None,
        help="Optional extended tree mask for tree/non-tree static metric strata.",
    )
    parser.add_argument("--tree-mask-index", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = evaluate(
        args.render_output,
        args.dataset_path,
        args.mask_pickle,
        device,
        tree_mask_pickle=args.tree_mask_pickle,
        tree_mask_index=args.tree_mask_index,
    )
    output = args.output or args.render_output / "rgb_metrics.json"
    output.write_text(json.dumps(result, indent=2))
    summary = {
        "raw": result["mean"],
        **{name: value["mean"] for name, value in result["masked"].items()},
        "ulfloc_legacy": result["protocol"]["ulfloc_legacy"]["mean"],
    }
    if "tree_stratified" in result:
        summary["tree_stratified"] = {
            name: value["mean"]
            for name, value in result["tree_stratified"].items()
            if isinstance(value, dict) and "mean" in value
        }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
