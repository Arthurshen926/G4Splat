#!/usr/bin/env python3
"""Export a clean ULF-Loc/STDLoc checkpoint over every staged train camera.

This is deliberately an *adapter-free evaluator*: the supplied clean external
tree is imported read-only and its own ``Scene``, ``GaussianModel`` and gsplat
renderer are used.  It exists because the released ULF-Loc/STDLoc repositories
have no standalone all-camera RGB renderer, while a fair Cambridge audit needs
all 1,487 real training cameras rather than their five-image TensorBoard
samples.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _save_rgb(path: Path, image: torch.Tensor) -> None:
    value = (
        image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5
    ).astype(np.uint8)
    Image.fromarray(value).save(path)


def _psnr(image: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((image - target).square()).item()
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def _load_external_modules(root: Path):
    """Import the requested clean tree before any repository-local renderer."""
    root = root.resolve()
    if not (root / "train.py").is_file():
        raise FileNotFoundError(f"Not an external Gaussian-splatting checkout: {root}")
    # This command starts in G4Splat, whose 2DGS modules have overlapping
    # top-level names.  It imports no such module before this point; removing
    # a stale interactive import makes the contract explicit for reuse.
    for module_name in ("arguments", "gaussian_renderer", "scene"):
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(root))
    from arguments import ModelParams  # type: ignore
    from gaussian_renderer import render_gsplat  # type: ignore
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore

    return ModelParams, render_gsplat, Scene, GaussianModel, GaussianModel_2dgs


def _external_dataset_args(
    *,
    source_path: Path,
    model_path: Path,
    images: str,
    feature_type: str,
    resolution: int,
    gaussian_type: str,
) -> list[str]:
    """Build the clean repository's ModelParams argument contract.

    ``feature_type`` looks unrelated to RGB-only rendering, but clean
    ULFLoc/STDLoc's Cambridge scene reader uses it while constructing
    ``SceneInfo``.  Keeping it in a tested helper prevents an export-only
    invocation from silently diverging from the successful training command.
    """
    return [
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--images",
        str(images),
        "-f",
        str(feature_type),
        "-r",
        str(resolution),
        "--data_device",
        "cpu",
        "-g",
        str(gaussian_type),
    ]


def _gsplat_backend_audit(*, required_version: str | None = None) -> dict[str, str]:
    """Record the exact gsplat module selected by an external renderer.

    ULF-Loc and STDLoc list ``gsplat`` without a version pin.  That is not a
    harmless detail for the 2D renderer: its background-channel contract has
    changed between releases.  An all-camera export must therefore record the
    resolved module, and strict controls can require an exact version instead
    of silently changing renderer behavior with the Python environment.
    """
    gsplat = importlib.import_module("gsplat")
    version = str(getattr(gsplat, "__version__", "unknown"))
    if required_version is not None and version != str(required_version):
        raise RuntimeError(
            "External renderer gsplat version does not match the declared "
            f"strict contract: required={required_version}, resolved={version}, "
            f"module={getattr(gsplat, '__file__', '<unknown>')}"
        )
    module_file = str(getattr(gsplat, "__file__", "<unknown>"))
    return {
        "version": version,
        "module_file": module_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument(
        "--external-root",
        type=Path,
        help="Clean ULF-Loc or STDLoc checkout to import read-only.",
    )
    # Backward compatibility for the initial ULF-only command spelling.
    roots.add_argument("--ulfloc-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument(
        "--feature-type",
        default="sp",
        help=(
            "Feature type expected by the clean external Scene reader. This is "
            "still required for RGB-only export because the reader uses it to "
            "construct its scene metadata."
        ),
    )
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--longest-edge", type=int, default=640)
    parser.add_argument(
        "--gaussian-type",
        choices=("3dgs", "2dgs"),
        default="3dgs",
        help=(
            "Gaussian parameterization used by the clean external checkpoint. "
            "The 2dgs option preserves the external gsplat rasterizer so it can "
            "serve as a renderer-only counterpart to a local 2DGS control."
        ),
    )
    parser.add_argument(
        "--require-gsplat-version",
        default=None,
        help=(
            "Fail unless the imported external gsplat module has this exact "
            "version. Use for strict renderer controls because clean ULF/STD "
            "requirements do not pin gsplat."
        ),
    )
    parser.add_argument("--expected-train-count", type=int, default=1487)
    args = parser.parse_args()

    if args.iteration <= 0:
        parser.error("--iteration must be positive")
    if args.longest_edge <= 0:
        parser.error("--longest-edge must be positive")
    source_path = args.source_path.resolve()
    model_path = args.model_path.resolve()
    output = args.output.resolve()
    ply_path = model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    if not ply_path.is_file():
        raise FileNotFoundError(f"External checkpoint does not exist: {ply_path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix evaluator outputs: {output}")

    external_root = (args.external_root or args.ulfloc_root).resolve()
    (
        ModelParams,
        render_gsplat,
        Scene,
        GaussianModel,
        GaussianModel2D,
    ) = _load_external_modules(external_root)
    gsplat_backend = _gsplat_backend_audit(
        required_version=args.require_gsplat_version,
    )
    external_parser = argparse.ArgumentParser(add_help=False)
    model_params = ModelParams(external_parser)
    external_args = external_parser.parse_args(
        _external_dataset_args(
            source_path=source_path,
            model_path=model_path,
            images=str(args.images),
            feature_type=str(args.feature_type),
            resolution=int(args.resolution),
            gaussian_type=str(args.gaussian_type),
        )
    )
    dataset = model_params.extract(external_args)
    dataset.longest_edge = int(args.longest_edge)

    torch.cuda.set_device(0)
    gaussian_model_type = GaussianModel if args.gaussian_type == "3dgs" else GaussianModel2D
    gaussians = gaussian_model_type(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras()
    if len(views) != int(args.expected_train_count):
        raise RuntimeError(
            "The clean external evaluator would not cover the declared all-train set: "
            f"loaded={len(views)}, expected={args.expected_train_count}"
        )
    output_names = [Path(str(view.image_name)).name for view in views]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError("External output names are not unique after basename normalization")

    output.mkdir(parents=True)
    renders_dir = output / "renders"
    gt_dir = output / "gt"
    renders_dir.mkdir()
    gt_dir.mkdir()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    per_view: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for view, name in tqdm(zip(views, output_names), total=len(views), desc="clean external full train"):
            prediction = render_gsplat(
                view,
                gaussians,
                background,
                rgb_only=True,
                norm_feat_bf_render=dataset.norm_before_render,
                longest_edge=dataset.longest_edge,
                rasterize_mode="antialiased",
            )["render"].clamp(0.0, 1.0)
            target = torch.nn.functional.interpolate(
                view.original_image.cuda().unsqueeze(0),
                size=prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0].clamp(0.0, 1.0)
            per_view[name] = {"psnr": _psnr(prediction, target)}
            _save_rgb(renders_dir / name, prediction)
            _save_rgb(gt_dir / name, target)

    values = [item["psnr"] for item in per_view.values()]
    report = {
        "experiment": f"clean_external_gsplat_{args.gaussian_type}_full_train_export",
        "external_root": str(external_root),
        "source_path": str(source_path),
        "model_path": str(model_path),
        "iteration": int(args.iteration),
        "renderer": f"clean_external_gsplat_{args.gaussian_type}_antialiased",
        "gsplat_backend": gsplat_backend,
        "gaussian_type": str(args.gaussian_type),
        "feature_type": str(args.feature_type),
        "input_resolution": int(args.resolution),
        "longest_edge": int(dataset.longest_edge),
        "real_training_camera_count": int(len(views)),
        "mean_psnr": float(np.mean(values)),
        "per_view": per_view,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_view"}, indent=2))


if __name__ == "__main__":
    main()
