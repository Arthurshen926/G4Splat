#!/usr/bin/env python3
"""Evaluate Gaussian checkpoints on diagnosed artifact components and support views.

This is an experimental, read-only adapter. It compares one or more Gaussian PLY
checkpoints on the exact training-view component, its clean support views, and the
clean-support depth attached to the artifact-directed pseudo camera.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "2d-gaussian-splatting"
for import_root in (REPO_ROOT, GS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from artifact_guided_repair import CameraGeometry  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from guidance.cam_utils import MiniCam  # noqa: E402
from scene import GaussianModel  # noqa: E402


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _make_camera(geometry: CameraGeometry) -> MiniCam:
    return MiniCam(
        geometry.c2w.astype(np.float32),
        geometry.width,
        geometry.height,
        2.0 * math.atan(geometry.height / (2.0 * geometry.fy)),
        2.0 * math.atan(geometry.width / (2.0 * geometry.fx)),
    )


def _read_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value.copy()).permute(2, 0, 1).cuda()


def _save_rgb(path: Path, value: torch.Tensor) -> None:
    image = value.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(image * 255.0 + 0.5), mode="RGB").save(path)


def _psnr(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mse = (rendered - target).square().mean(dim=0)
    return float((-10.0 * torch.log10(mse[mask].mean().clamp_min(1e-12))).item())


def _load_validation_views(
    diagnostic_manifest: dict,
    diagnostics_root: Path,
    proposal: dict,
) -> list[dict]:
    indices = [int(proposal["target_view_index"])]
    indices.extend(int(item["view_index"]) for item in proposal["clean_reference_support"])
    output = []
    for index in dict.fromkeys(indices):
        record = diagnostic_manifest["views"][index]
        geometry = CameraGeometry.from_json(record["camera"])
        semantic = np.asarray(
            Image.open(diagnostics_root / record["files"]["semantic"]).convert("L")
        ) > 127
        labels = np.asarray(Image.open(diagnostics_root / record["files"]["labels"]))
        component = None
        if index == int(proposal["target_view_index"]):
            component = labels == int(proposal["component"]["component_id"])
        output.append(
            {
                "index": index,
                "name": record["image_name"],
                "camera": _make_camera(geometry),
                "gt": _read_rgb(Path(record["image_path"]), (geometry.width, geometry.height)),
                "static_mask": torch.from_numpy(semantic.copy()).cuda(),
                "component_mask": torch.from_numpy(component.copy()).cuda()
                if component is not None
                else None,
            }
        )
    return output


@torch.no_grad()
def _evaluate_validation_views(gaussians, views, pipe, background) -> tuple[list[dict], dict]:
    rows = []
    images = {}
    for view in views:
        image = render(view["camera"], gaussians, pipe, background)["render"].clamp(0.0, 1.0)
        row = {
            "name": view["name"],
            "full_psnr": _psnr(image, view["gt"], torch.ones_like(view["static_mask"])),
            "static_psnr": _psnr(image, view["gt"], view["static_mask"]),
        }
        if view["component_mask"] is not None and view["component_mask"].any():
            row["component_psnr"] = _psnr(image, view["gt"], view["component_mask"])
        rows.append(row)
        images[view["name"]] = image
    return rows, images


def _build_component_context(stage_root: Path, diagnostics_root: Path, diagnostic_manifest: dict, proposal: dict) -> dict:
    pseudo_index = int(proposal["pseudo_view_index"])
    stem = f"{pseudo_index:06d}"
    known = np.asarray(
        Image.open(stage_root / "select-gs" / f"mask_frame{stem}.png").convert("L")
    ) > 127
    clean_depth = np.asarray(Image.open(proposal["clean_support_depth_path"]), dtype=np.float32)
    return {
        "proposal": proposal,
        "pseudo_camera": _make_camera(CameraGeometry.from_json(proposal["pseudo_camera"])),
        "repair_mask": torch.from_numpy((~known).copy()).cuda(),
        "clean_depth": torch.from_numpy(clean_depth.copy()).cuda(),
        "validation_views": _load_validation_views(
            diagnostic_manifest,
            diagnostics_root,
            proposal,
        ),
    }


@torch.no_grad()
def _pseudo_metrics(package: dict, context: dict, reference_rgb: torch.Tensor) -> dict:
    depth = package["surf_depth"][0]
    clean_depth = context["clean_depth"]
    repair = context["repair_mask"] & torch.isfinite(clean_depth) & (clean_depth > 0)
    denominator = torch.maximum(torch.maximum(depth.abs(), clean_depth.abs()), depth.new_tensor(1e-6))
    relative = torch.abs(depth - clean_depth) / denominator
    outside = ~context["repair_mask"]
    return {
        "clean_depth_relative_mean": float(relative[repair].mean().item()),
        "clean_depth_relative_median": float(relative[repair].median().item()),
        "alpha_mean": float(package["rend_alpha"][0][repair].mean().item()),
        "inside_rgb_mae_vs_reference": float(
            torch.abs(package["render"] - reference_rgb)[:, context["repair_mask"]].mean().item()
        ),
        "outside_rgb_mae_vs_reference": float(
            torch.abs(package["render"] - reference_rgb)[:, outside].mean().item()
        ),
    }


def _metric_deltas(current: dict, reference: dict) -> dict:
    reference_pseudo = reference["pseudo"]
    deltas = {
        key: float(value - reference_pseudo[key])
        for key, value in current["pseudo"].items()
        if key in reference_pseudo and isinstance(value, (int, float))
    }
    reference_views = {row["name"]: row for row in reference["validation"]}
    view_deltas = []
    for row in current["validation"]:
        base = reference_views[row["name"]]
        view_deltas.append(
            {
                "name": row["name"],
                **{
                    f"{key}_delta": float(value - base[key])
                    for key, value in row.items()
                    if key != "name" and key in base
                },
            }
        )
    return {"pseudo": deltas, "validation": view_deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage_root", type=Path, required=True)
    parser.add_argument("--diagnostics_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("LABEL", "MODEL_PATH", "ITERATION"),
        required=True,
        help="Repeat in reference-first order, for example: --model baseline path 30000",
    )
    args = parser.parse_args()

    stage_root = args.stage_root.expanduser().resolve()
    diagnostics_root = args.diagnostics_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_manifest = json.loads((stage_root / "artifact_guided_manifest.json").read_text())
    diagnostic_manifest = json.loads((diagnostics_root / "diagnostic_manifest.json").read_text())
    proposals = [
        proposal
        for proposal in stage_manifest["pseudo_views"]
        if proposal["classification"] == "wrong_geometry"
    ]
    if not proposals:
        raise RuntimeError("The stage manifest contains no wrong-geometry pseudo views")
    contexts = [
        _build_component_context(stage_root, diagnostics_root, diagnostic_manifest, proposal)
        for proposal in proposals
    ]

    labels = [entry[0] for entry in args.model]
    if len(labels) != len(set(labels)):
        raise ValueError("Model labels must be unique")
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        depth_ratio=0.0,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    component_results = [
        {
            "pseudo_view_index": int(context["proposal"]["pseudo_view_index"]),
            "target_image_name": context["proposal"]["target_image_name"],
            "component_id": int(context["proposal"]["component"]["component_id"]),
            "models": {},
        }
        for context in contexts
    ]
    reference_pseudo_images: list[torch.Tensor] = []

    for model_index, (label, model_path_value, iteration_value) in enumerate(args.model):
        model_path = Path(model_path_value).expanduser().resolve()
        iteration = int(iteration_value)
        checkpoint = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        gaussians = GaussianModel(3)
        gaussians.load_ply(str(checkpoint))
        safe_label = _safe_name(label)

        for component_index, context in enumerate(contexts):
            validation, validation_images = _evaluate_validation_views(
                gaussians,
                context["validation_views"],
                pipe,
                background,
            )
            pseudo_package = render(context["pseudo_camera"], gaussians, pipe, background)
            if model_index == 0:
                reference_pseudo_images.append(pseudo_package["render"].detach().clone())
            reference_rgb = reference_pseudo_images[component_index]
            pseudo = _pseudo_metrics(pseudo_package, context, reference_rgb)
            component_results[component_index]["models"][label] = {
                "model_path": str(model_path),
                "iteration": iteration,
                "point_count": int(len(gaussians.get_xyz)),
                "validation": validation,
                "pseudo": pseudo,
            }

            component_root = output_root / f"component_{component_index:03d}"
            _save_rgb(component_root / f"pseudo.{safe_label}.png", pseudo_package["render"])
            for name, image in validation_images.items():
                _save_rgb(component_root / f"{_safe_name(name)}.{safe_label}.png", image)

        del gaussians
        torch.cuda.empty_cache()

    reference_label = labels[0]
    for component in component_results:
        reference = component["models"][reference_label]
        for label in labels[1:]:
            component["models"][label]["delta_vs_reference"] = _metric_deltas(
                component["models"][label],
                reference,
            )

    report = {
        "version": 1,
        "mode": "artifact_component_clean_support_evaluation",
        "stage_root": str(stage_root),
        "diagnostics_dir": str(diagnostics_root),
        "reference_label": reference_label,
        "components": component_results,
    }
    report_path = output_root / "artifact_model_comparison.json"
    report_path.write_text(json.dumps(_json_ready(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(report), indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
