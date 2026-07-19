#!/usr/bin/env python3
"""Measure whether warm-start reseed Gaussians can affect a repair region."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "2d-gaussian-splatting"
for root in (REPO_ROOT, GS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from arguments import PipelineParams  # noqa: E402
from artifact_guided_repair import CameraGeometry  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from guidance.cam_utils import MiniCam  # noqa: E402
from scene import GaussianModel  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def camera_from_geometry(geometry: CameraGeometry) -> MiniCam:
    return MiniCam(
        geometry.c2w.astype(np.float32),
        geometry.width,
        geometry.height,
        2.0 * math.atan(geometry.height / (2.0 * geometry.fy)),
        2.0 * math.atan(geometry.width / (2.0 * geometry.fx)),
    )


def package_arrays(package: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "rgb": package["render"].detach().cpu().permute(1, 2, 0).numpy(),
        "alpha": package["rend_alpha"][0].detach().cpu().numpy(),
        "depth": package["surf_depth"][0].detach().cpu().numpy(),
        "radii": package["radii"].detach().cpu().numpy(),
    }


def render_arrays(camera: MiniCam, gaussians: GaussianModel, pipe: Any, background: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        return package_arrays(render(camera, gaussians, pipe, background))


def masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values)[mask]
    return {
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p95": float(np.quantile(selected, 0.95)),
        "max": float(np.max(selected)),
    }


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 1e-6)
    if valid_mask is not None:
        valid &= valid_mask
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        log_depth = np.log(np.maximum(depth, 1e-6))
        low, high = np.quantile(log_depth[valid], [0.02, 0.98])
        high = max(float(high), float(low) + 1e-6)
        normalized[valid] = np.uint8(
            np.clip((log_depth[valid] - low) / (high - low), 0.0, 1.0) * 255.0
        )
    output = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[~valid] = 0
    return output


def rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5), cv2.COLOR_RGB2BGR)


def make_tile(image: np.ndarray, title: str, size: tuple[int, int] = (320, 180)) -> np.ndarray:
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((size[1] + 32, size[0], 3), 245, dtype=np.uint8)
    canvas[32:] = image
    cv2.putText(canvas, title[:54], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
    return canvas


def save_sheet(
    output: Path,
    full: dict[str, np.ndarray],
    prefix: dict[str, np.ndarray],
    suffix: dict[str, np.ndarray],
    clean_depth: np.ndarray,
    repair_mask: np.ndarray,
) -> None:
    rgb_delta = np.mean(np.abs(full["rgb"] - prefix["rgb"]), axis=2)
    alpha_delta = np.abs(full["alpha"] - prefix["alpha"])
    tiles = [
        make_tile(rgb_uint8(full["rgb"]), "full model"),
        make_tile(rgb_uint8(prefix["rgb"]), "frozen prefix only"),
        make_tile(rgb_uint8(suffix["rgb"]), "261-point suffix only"),
        make_tile(cv2.applyColorMap(np.uint8(np.clip(rgb_delta / 0.05, 0.0, 1.0) * 255), cv2.COLORMAP_TURBO), "|full-prefix| RGB (scale=.05)"),
        make_tile(cv2.applyColorMap(np.uint8(np.clip(alpha_delta / 0.01, 0.0, 1.0) * 255), cv2.COLORMAP_TURBO), "|full-prefix| alpha (scale=.01)"),
        make_tile(colorize_depth(full["depth"]), "full rendered depth"),
        make_tile(colorize_depth(prefix["depth"]), "prefix rendered depth"),
        make_tile(colorize_depth(suffix["depth"]), "suffix rendered depth"),
        make_tile(colorize_depth(clean_depth, repair_mask), "clean support depth in repair mask"),
        make_tile(cv2.cvtColor(np.uint8(repair_mask) * 255, cv2.COLOR_GRAY2BGR), f"repair mask {repair_mask.mean():.2%}"),
    ]
    columns = 5
    blank = np.full_like(tiles[0], 245)
    while len(tiles) % columns:
        tiles.append(blank)
    sheet = np.vstack([np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)])
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ply", type=Path, required=True)
    parser.add_argument("--base-count", type=int, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--pseudo-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    pipeline = PipelineParams(parser)
    args = parser.parse_args()
    pipe = pipeline.extract(args)
    manifest = read_json(args.stage_manifest)
    view = next(
        item for item in manifest["pseudo_views"] if int(item["pseudo_view_index"]) == int(args.pseudo_index)
    )
    geometry = CameraGeometry.from_json(view["pseudo_camera"])
    camera = camera_from_geometry(geometry)
    repair_mask = np.asarray(
        Image.open(args.stage_root / "select-gs" / f"clean_support_depth_mask_frame{args.pseudo_index:06d}.png").convert("L")
    ) > 127
    clean_depth = cv2.imread(
        str(args.stage_root / "select-gs" / f"clean_support_depth_frame{args.pseudo_index:06d}.tiff"),
        cv2.IMREAD_UNCHANGED,
    ).astype(np.float32)
    if repair_mask.shape != (geometry.height, geometry.width):
        raise ValueError("Repair mask and pseudo camera dimensions differ")

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(args.model_ply))
    total_count = int(gaussians.get_xyz.shape[0])
    base_count = int(args.base_count)
    if not 0 < base_count < total_count:
        raise ValueError(f"base-count must split the model, got {base_count}/{total_count}")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    opacity = gaussians._opacity.detach().clone()
    hidden_logit = torch.full_like(opacity[:1], -30.0)

    gaussians._opacity.data.copy_(opacity)
    full = render_arrays(camera, gaussians, pipe, background)
    gaussians._opacity.data[base_count:] = hidden_logit
    prefix = render_arrays(camera, gaussians, pipe, background)
    gaussians._opacity.data.copy_(opacity)
    gaussians._opacity.data[:base_count] = hidden_logit
    suffix = render_arrays(camera, gaussians, pipe, background)
    gaussians._opacity.data.copy_(opacity)

    valid_depth = repair_mask & np.isfinite(clean_depth) & (clean_depth > 1e-6)
    camera_package = render(camera, gaussians, pipe, background)
    depth = camera_package["surf_depth"][0]
    mask_tensor = torch.from_numpy(valid_depth).to(device=depth.device)
    clean_tensor = torch.from_numpy(clean_depth).to(device=depth.device, dtype=depth.dtype)
    relative = torch.abs(depth - clean_tensor) / torch.clamp(clean_tensor, min=1e-6)
    loss = relative[mask_tensor].mean()
    opacity_gradient, xyz_gradient = torch.autograd.grad(
        loss,
        (gaussians._opacity, gaussians._xyz),
        allow_unused=True,
    )
    suffix_opacity_gradient = opacity_gradient[base_count:].detach().abs().cpu().numpy().reshape(-1)
    suffix_xyz_gradient = torch.linalg.vector_norm(xyz_gradient[base_count:], dim=1).detach().cpu().numpy()

    rgb_delta = np.mean(np.abs(full["rgb"] - prefix["rgb"]), axis=2)
    alpha_delta = np.abs(full["alpha"] - prefix["alpha"])
    depth_delta = np.abs(full["depth"] - prefix["depth"]) / np.maximum(np.abs(prefix["depth"]), 1e-6)
    report = {
        "version": 1,
        "mode": "warmstart_suffix_visibility_audit",
        "model_ply": str(args.model_ply.resolve()),
        "total_gaussians": total_count,
        "prefix_gaussians": base_count,
        "suffix_gaussians": total_count - base_count,
        "repair_fraction": float(repair_mask.mean()),
        "full_repair_alpha": masked_stats(full["alpha"], repair_mask),
        "prefix_repair_alpha": masked_stats(prefix["alpha"], repair_mask),
        "suffix_only_repair_alpha": masked_stats(suffix["alpha"], repair_mask),
        "effective_suffix_rgb_contribution": masked_stats(rgb_delta, repair_mask),
        "effective_suffix_alpha_contribution": masked_stats(alpha_delta, repair_mask),
        "effective_suffix_relative_depth_change": masked_stats(depth_delta, repair_mask),
        "suffix_only_covered_repair_fraction_alpha_gt_001": float(np.mean(suffix["alpha"][repair_mask] > 0.01)),
        "suffix_in_frustum_count_full_render": int(np.sum(full["radii"][base_count:] > 0)),
        "clean_depth_relative_loss": float(loss.detach().cpu()),
        "suffix_opacity_gradient": {
            **masked_stats(suffix_opacity_gradient, np.ones_like(suffix_opacity_gradient, dtype=bool)),
            "nonzero_gt_1e_10": int(np.sum(suffix_opacity_gradient > 1e-10)),
        },
        "suffix_xyz_gradient_norm": {
            **masked_stats(suffix_xyz_gradient, np.ones_like(suffix_xyz_gradient, dtype=bool)),
            "nonzero_gt_1e_10": int(np.sum(suffix_xyz_gradient > 1e-10)),
        },
    }
    output = args.output.expanduser().resolve()
    write_json(output / "warmstart_visibility_audit.json", report)
    save_sheet(output / "warmstart_visibility_contact_sheet.jpg", full, prefix, suffix, clean_depth, repair_mask)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
