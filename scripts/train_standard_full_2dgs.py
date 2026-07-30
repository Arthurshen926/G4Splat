#!/usr/bin/env python
"""Train a deliberately simple, all-real-view 2DGS reconstruction control.

This runner exists to keep the first Cambridge ablation comparable to the
direct RGB optimization used by standard 3DGS/ULF-Loc style systems.  By
default it does not read Charts, MASt3R depth, planes, See3D images,
per-image colour affine parameters, or QC weights.  The explicitly opt-in
per-image affine branch is a separate appearance ablation: it is learned only
for real training cameras and is the identity for every unknown/test camera.
A semantic mask is used only when an explicitly requested external-
supervision profile requires it.  Every camera in ``sparse/0`` is sampled for
RGB supervision from iteration one.

The output manifest records the exact input camera count and refuses to run if
the staged image directory and COLMAP reconstruction disagree.  That makes a
partial ``train_opt`` copy impossible to mistake for an all-train control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SURFEL_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from outdoor.lazy_scene import LazyScene  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from scene.colmap_loader import read_extrinsics_binary  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup, tensor_to_resized_mask  # noqa: E402
from matcha.cambridge_training import (  # noqa: E402
    apply_per_image_affine_color_correction,
    compute_rgb_loss,
    opacity_reset_due,
    PerImageAffineColorCorrection,
    save_per_image_affine_color_correction,
    ulfloc_masked_supervision,
)


def _staged_names(image_dir: Path) -> set[str]:
    image_suffixes = {".png", ".jpg", ".jpeg"}
    names = {
        path.name
        for path in image_dir.iterdir()
        if (path.is_file() or path.is_symlink())
        and path.suffix.lower() in image_suffixes
    }
    if not names:
        raise RuntimeError(f"No RGB image files found in {image_dir}")
    return names


def _loaded_camera_filenames(staged_names: set[str], cameras: list) -> set[str]:
    """Recover staged filenames without assuming every camera is a PNG.

    2DGS stores ``Camera.image_name`` without the filename extension, whereas
    the all-train audit must compare it to the filenames in COLMAP and
    ``images/``.  The former implementation unconditionally appended
    ``.png``.  That happened to work for Cambridge, but made the audit itself
    format-dependent rather than a check of the camera contract.
    """
    by_stem = {Path(name).stem: name for name in staged_names}
    if len(by_stem) != len(staged_names):
        raise RuntimeError("Staged RGB filenames are ambiguous after stem normalization")
    camera_stems = [Path(str(camera.image_name)).stem for camera in cameras]
    if len(set(camera_stems)) != len(camera_stems):
        raise RuntimeError("Loaded training cameras do not have unique image names")
    unknown = sorted(set(camera_stems) - set(by_stem))
    if unknown:
        raise RuntimeError(
            "Loaded training cameras are absent from staged RGB files: "
            f"{unknown[:3]}"
        )
    return {by_stem[stem] for stem in camera_stems}


def _source_control_manifest_path(source_path: Path) -> Path | None:
    """Locate the adapter provenance manifest, across supported layouts."""
    for candidate in (
        source_path / "input_manifest.json",
        source_path / "external_control_manifest.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _input_audit(source_path: Path, cameras: list) -> dict:
    image_dir = source_path / "images"
    sparse_images = source_path / "sparse" / "0" / "images.bin"
    mapping_path = source_path / "name_mapping.json"
    if not image_dir.is_dir() or not sparse_images.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(
            "A standard full-view control requires images/, sparse/0/images.bin, "
            f"and name_mapping.json under {source_path}"
        )
    staged = _staged_names(image_dir)
    mapping = json.loads(mapping_path.read_text())
    mapping_names = {str(name) for name in mapping}
    colmap = read_extrinsics_binary(str(sparse_images))
    colmap_names = {Path(image.name).name for image in colmap.values()}
    camera_names = _loaded_camera_filenames(staged, cameras)
    if staged != mapping_names or staged != colmap_names or staged != camera_names:
        raise RuntimeError(
            "The direct control would not supervise exactly one RGB image per "
            "staged COLMAP camera. "
            f"images={len(staged)}, mapping={len(mapping_names)}, "
            f"colmap={len(colmap_names)}, loaded={len(camera_names)}"
        )
    source_split = source_path / "source_split.txt"
    source_split_count = None
    if source_split.is_file():
        source_split_count = sum(1 for line in source_split.read_text().splitlines() if line.strip())
        if source_split_count != len(staged):
            raise RuntimeError(
                "source_split.txt does not describe every staged training image: "
                f"split={source_split_count}, images={len(staged)}"
            )
    # The external-control adapter writes ``input_manifest.json``.  Accept
    # the older filename too so a control can fingerprint either layout rather
    # than silently reporting no source-data provenance.
    control_manifest = _source_control_manifest_path(source_path)
    return {
        "source_path": str(source_path),
        "image_count": len(staged),
        "colmap_camera_count": len(colmap_names),
        "loaded_camera_count": len(camera_names),
        "source_split_count": source_split_count,
        "camera_set_sha256_input": _set_digest(staged),
        "name_mapping_sha256": _file_digest(mapping_path),
        "source_control_manifest": None if control_manifest is None else str(control_manifest.resolve()),
        "source_control_manifest_sha256": (
            None if control_manifest is None else _file_digest(control_manifest)
        ),
    }


def _set_digest(names: set[str]) -> str:
    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    """Return a streamed SHA-256 digest without duplicating a PLY in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    """Hash a JSON-compatible value with a canonical representation."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _capture_rng_state() -> dict:
    """Capture every RNG state relevant to a fixed-schedule continuation.

    The strict runner normally draws its camera identities from a precomputed
    schedule, but CUDA kernels and future optional branches can still consume
    Torch's generators.  Saving the generators makes a state fork auditable
    instead of merely a PLY warm start with a fresh optimizer.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _rng_state_byte_tensor_on_cpu(value: object, *, field: str) -> torch.Tensor:
    """Return a serialized Torch RNG state in the device layout its setters need.

    A full training-state payload is loaded with ``map_location='cuda'`` so
    the Gaussian parameters and Adam buffers can be restored without a second
    large device copy.  ``torch.load`` applies that mapping to *every* tensor,
    however, including the byte tensors returned by ``torch.get_rng_state``
    and ``torch.cuda.get_rng_state_all``.  Both ``torch.set_rng_state`` and
    CUDA's RNG setter expect those byte states on CPU.  Normalize only this
    small metadata subset here; the model/optimizer tensors deliberately stay
    on CUDA.
    """
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Training-state RNG field {field!r} is not a tensor")
    if value.dtype != torch.uint8:
        raise RuntimeError(
            f"Training-state RNG field {field!r} must be uint8, got {value.dtype}"
        )
    return value.detach().to(device="cpu").contiguous()


def _restore_rng_state(state: dict) -> None:
    """Restore a state produced by :func:`_capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"Training-state RNG payload is missing keys: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(
        _rng_state_byte_tensor_on_cpu(state["torch_cpu"], field="torch_cpu")
    )
    if "torch_cuda_all" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Training-state payload requires CUDA RNG restoration")
        cuda_states = state["torch_cuda_all"]
        if not isinstance(cuda_states, (list, tuple)):
            raise RuntimeError("Training-state RNG field 'torch_cuda_all' is not a sequence")
        torch.cuda.set_rng_state_all(
            [
                _rng_state_byte_tensor_on_cpu(value, field=f"torch_cuda_all[{index}]")
                for index, value in enumerate(cuda_states)
            ]
        )


def _training_state_contract(
    input_audit: dict,
    camera_schedule_audit: dict,
    *,
    declared_total_iterations: int,
) -> dict:
    """Build the immutable contract for an exact post-checkpoint fork.

    Optimizer hyperparameters are intentionally not part of this contract:
    the point of a fork is to compare one declared post-checkpoint schedule
    (for example non-position LR decay) against another.  Data, renderer,
    code, and the full camera stream must remain identical.
    """
    schedule_keys = (
        "protocol",
        "seed",
        "scheduled_iterations",
        "training_camera_count",
        "sequence_sha256",
    )
    input_keys = (
        "source_path",
        "image_count",
        "colmap_camera_count",
        "loaded_camera_count",
        "source_split_count",
        "camera_set_sha256_input",
        "name_mapping_sha256",
        "source_control_manifest_sha256",
        "scene_sparse_ply_sha256",
        "scene_sparse_ply_bytes",
        "supervision_profile",
        "cambridge_mask_pickle_sha256",
        "cambridge_mask_dataset_path",
        "mask_resolution_audit",
        "ulfloc_native_pixel_protocol",
        "white_background",
        "resolution",
        "cudnn",
        "renderer",
        "mip_filter",
        "per_image_affine_color",
    )
    implementation = input_audit.get("implementation")
    if implementation is None:
        raise RuntimeError("Cannot create a training-state contract without implementation audit")
    return {
        "protocol": "standard_2dgs_training_state_contract_v1",
        "declared_total_iterations": int(declared_total_iterations),
        "input": {key: input_audit.get(key) for key in input_keys},
        "camera_schedule": {
            key: camera_schedule_audit.get(key) for key in schedule_keys
        },
        "implementation_sha256": _json_digest(implementation),
    }


def _contract_mismatches(expected: object, actual: object, prefix: str = "") -> list[str]:
    """Return compact, deterministic differences between two state contracts."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = sorted(set(expected) | set(actual))
        mismatches: list[str] = []
        for key in keys:
            child = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                mismatches.append(f"{child}: unexpected value in checkpoint")
            elif key not in actual:
                mismatches.append(f"{child}: missing from checkpoint")
            else:
                mismatches.extend(_contract_mismatches(expected[key], actual[key], child))
        return mismatches
    if expected != actual:
        return [f"{prefix}: expected {expected!r}, got {actual!r}"]
    return []


def _load_training_state(path: Path) -> dict:
    """Load and minimally validate a full optimizer/RNG continuation state."""
    if not path.is_file():
        raise FileNotFoundError(f"Training-state checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cuda", weights_only=False)
    except TypeError:
        # ``weights_only`` was added after older Torch versions supported by
        # upstream 2DGS; keep the control runner usable in those environments.
        payload = torch.load(path, map_location="cuda")
    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise RuntimeError(f"Unsupported training-state checkpoint: {path}")
    required = {
        "iteration",
        "model_capture",
        "rng_state",
        "training_loop_state",
        "contract",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"Malformed training-state checkpoint {path}: missing {missing}")
    if int(payload["iteration"]) < 1:
        raise RuntimeError(f"Training-state checkpoint has invalid iteration: {path}")
    loop_state = payload["training_loop_state"]
    if not isinstance(loop_state, dict) or "ema" not in loop_state:
        raise RuntimeError(
            f"Malformed training-state checkpoint {path}: training_loop_state.ema is required"
        )
    try:
        float(loop_state["ema"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Malformed training-state checkpoint {path}: training_loop_state.ema is not numeric"
        ) from error
    return payload


def _restore_color_correction_state(
    payload: dict,
    *,
    color_correction: PerImageAffineColorCorrection | None,
    color_correction_optimizer: torch.optim.Optimizer | None,
) -> None:
    """Restore the optional appearance branch at an exact state fork.

    Gaussian Adam moments are included in ``GaussianModel.capture``.  The
    camera-affine optimizer is intentionally separate from that model, so an
    exact continuation must carry both its parameters and moments explicitly.
    """
    checkpoint = payload.get("color_correction")
    if color_correction is None:
        if checkpoint is not None:
            raise RuntimeError(
                "Training-state checkpoint contains per-image affine state, "
                "but this run disabled --use-color-correction"
            )
        return
    if color_correction_optimizer is None:
        raise RuntimeError("Color correction is enabled without its optimizer")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Training-state checkpoint has no per-image affine state; "
            "cannot resume it with --use-color-correction"
        )
    required = {"model_state", "optimizer_state"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            "Malformed per-image affine state in training checkpoint: "
            f"missing {missing}"
        )
    color_correction.load_state_dict(checkpoint["model_state"], strict=True)
    color_correction_optimizer.load_state_dict(checkpoint["optimizer_state"])


def _save_training_state(
    *,
    model_path: Path,
    iteration: int,
    gaussians: GaussianModel,
    training_loop_state: dict,
    contract: dict,
    color_correction: PerImageAffineColorCorrection | None = None,
    color_correction_optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Persist the exact branch point after topology/reset/optimizer updates."""
    if (color_correction is None) != (color_correction_optimizer is None):
        raise RuntimeError(
            "Per-image affine model and optimizer must be present or absent together"
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    destination = model_path / "training_state" / f"iteration_{int(iteration)}.pth"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "iteration": int(iteration),
            "model_capture": gaussians.capture(),
            "rng_state": _capture_rng_state(),
            # EMA has no gradient effect, but preserving it prevents a
            # resumed trace from diverging from an uninterrupted trace at the
            # first logged step.  State forks should be exact in every
            # observable training-loop value, not merely the Gaussian PLY.
            "training_loop_state": dict(training_loop_state),
            # Version 2 intentionally permits this optional field: historical
            # no-affine checkpoints contain ``None``, while an affine branch
            # needs its disjoint Adam state to remain a true state fork.
            "color_correction": (
                None
                if color_correction is None
                else {
                    "model_state": color_correction.state_dict(),
                    "optimizer_state": color_correction_optimizer.state_dict(),
                }
            ),
            "contract": contract,
        },
        destination,
    )
    return destination


def _implementation_audit() -> dict:
    """Fingerprint the code and native kernel used by a standard control.

    A fixed data set and command line do not define a single-variable
    experiment if a dirty working tree changes the training loop, renderer, or
    CUDA extension between two runs.  Keep this intentionally small and
    explicit: only files reached by the all-real 2DGS control are included,
    rather than treating unrelated ROI/See3D edits as a training difference.
    """
    files = [
        REPO_ROOT / "scripts" / "train_standard_full_2dgs.py",
        REPO_ROOT / "matcha" / "cambridge_masks.py",
        REPO_ROOT / "matcha" / "cambridge_training.py",
        REPO_ROOT / "outdoor" / "lazy_scene.py",
        SURFEL_ROOT / "arguments" / "__init__.py",
        SURFEL_ROOT / "gaussian_renderer" / "__init__.py",
        SURFEL_ROOT / "scene" / "__init__.py",
        SURFEL_ROOT / "scene" / "dataset_readers.py",
        SURFEL_ROOT / "scene" / "gaussian_model.py",
        SURFEL_ROOT / "utils" / "camera_utils.py",
        SURFEL_ROOT / "utils" / "intrinsics_utils.py",
        SURFEL_ROOT / "utils" / "loss_utils.py",
        SURFEL_ROOT
        / "submodules"
        / "diff-surfel-rasterization"
        / "diff_surfel_rasterization"
        / "__init__.py",
    ]
    files.extend(
        sorted(
            (
                SURFEL_ROOT
                / "submodules"
                / "diff-surfel-rasterization"
                / "diff_surfel_rasterization"
            ).glob("_C*.so")
        )
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Standard-control implementation audit is missing required files: "
            + ", ".join(missing)
        )
    return {
        "protocol": "standard_2dgs_implementation_contract_v1",
        "files": {
            str(path.relative_to(REPO_ROOT)): {
                "sha256": _file_digest(path),
                "bytes": int(path.stat().st_size),
            }
            for path in files
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
    }


def _write_rigid_surface_handoff(
    *,
    model_path: Path,
    iteration: int,
    point_count: int,
    input_audit: dict,
    init_ply: Path | None,
) -> Path:
    """Write a closed provenance chain for rigid-to-hybrid continuation."""
    surface_ply = (
        model_path
        / "point_cloud"
        / f"iteration_{int(iteration)}"
        / "point_cloud.ply"
    ).resolve()
    if not surface_ply.is_file():
        raise FileNotFoundError(
            f"Final rigid surface PLY was not persisted: {surface_ply}"
        )
    input_manifest = (model_path / "input_manifest.json").resolve()
    camera_contract_path = (
        model_path / "camera_intrinsics_contract.json"
    ).resolve()
    camera_contract = json.loads(
        camera_contract_path.read_text(encoding="utf-8")
    )
    seed_manifest_path = (
        init_ply.with_suffix(".manifest.json").resolve()
        if init_ply is not None
        else None
    )
    seed_manifest = (
        json.loads(seed_manifest_path.read_text(encoding="utf-8"))
        if seed_manifest_path is not None and seed_manifest_path.is_file()
        else {}
    )
    rejection_reasons = []
    if not bool(input_audit.get("camera_container_only")):
        rejection_reasons.append("scene was not opened camera-only")
    if bool(input_audit.get("colmap_points_or_tracks_used", True)):
        rejection_reasons.append("COLMAP point/track geometry was used")
    if input_audit.get("initialization") != "ply_warm_start":
        rejection_reasons.append("rigid run did not start from an explicit PLY")
    if (
        input_audit.get("surface_ownership_contract")
        != "rigid_pixels_tree_sky_transient_excluded"
    ):
        rejection_reasons.append(
            "surface RGB ownership was not rigid-only/tree-excluded"
        )
    if (
        seed_manifest.get("protocol")
        != "role-aware-surface-seed-to-native-2dgs-v1"
    ):
        rejection_reasons.append(
            "initial PLY lacks the role-aware MASt3R/MAtCha export contract"
        )
    if bool(seed_manifest.get("historical_gaussian_input_used", True)):
        rejection_reasons.append("historical Gaussian input is not excluded")
    if bool(seed_manifest.get("colmap_points_or_tracks_used", True)):
        rejection_reasons.append(
            "seed export does not exclude COLMAP point/track geometry"
        )
    handoff = {
        "protocol": "native-rigid-surface-handoff-v1",
        "eligible_for_hybrid_surface_handoff": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "surface_ply": str(surface_ply),
        "surface_ply_sha256": _file_digest(surface_ply),
        "surface_iteration": int(iteration),
        "surface_point_count": int(point_count),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": _file_digest(input_manifest),
        "camera_intrinsics_contract": str(camera_contract_path),
        "camera_intrinsics_contract_sha256": _file_digest(
            camera_contract_path
        ),
        "camera_geometry_sha256": camera_contract[
            "camera_geometry_sha256"
        ],
        "seed_export_manifest": (
            str(seed_manifest_path)
            if seed_manifest_path is not None
            and seed_manifest_path.is_file()
            else None
        ),
        "seed_export_manifest_sha256": (
            _file_digest(seed_manifest_path)
            if seed_manifest_path is not None
            and seed_manifest_path.is_file()
            else None
        ),
        "historical_gaussian_input_used": bool(
            seed_manifest.get("historical_gaussian_input_used", True)
        ),
        "colmap_points_or_tracks_used": bool(
            input_audit.get("colmap_points_or_tracks_used", True)
            or seed_manifest.get("colmap_points_or_tracks_used", True)
        ),
        "all_real_rgb_from_iteration_one": (
            input_audit.get("objective")
            == "all_real_RGB_from_iteration_1"
        ),
        "surface_ownership_contract": input_audit.get(
            "surface_ownership_contract"
        ),
        "rgb_source_control_manifest_sha256": input_audit.get(
            "source_control_manifest_sha256"
        ),
    }
    destination = model_path / "rigid_surface_handoff.json"
    destination.write_text(
        json.dumps(handoff, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def _scene_input_ply_audit(model_path: Path) -> dict:
    """Fingerprint the sparse PLY copied by :class:`Scene` before training.

    ``Scene`` always copies the COLMAP sparse point cloud to ``input.ply``
    before it calls ``create_from_pcd``.  Previously the manifest only
    fingerprinted an explicit warm-start PLY, leaving the most common sparse
    initialization unaudited.  That made a strict comparison rely on a
    manual, after-the-fact hash check.
    """
    input_ply = (model_path / "input.ply").resolve()
    if not input_ply.is_file():
        raise FileNotFoundError(
            "Scene did not materialize its sparse initialization PLY at "
            f"{input_ply}"
        )
    return {
        "scene_sparse_ply": str(input_ply),
        "scene_sparse_ply_sha256": _file_digest(input_ply),
        "scene_sparse_ply_bytes": int(input_ply.stat().st_size),
    }


def _optimization_audit(opt) -> dict:
    """Serialize every optimization value that can affect a strict control.

    ``cfg_args`` preserves the command line, but it is not convenient for a
    machine comparison and earlier manifests omitted the densification
    threshold entirely.  Topology allocation is especially sensitive to that
    threshold, so it must be a first-class audited field rather than inferred
    from a free-form config string.
    """
    return {
        "iterations": int(opt.iterations),
        "learning_rates": {
            "position_init": float(opt.position_lr_init),
            "position_final": float(opt.position_lr_final),
            "position_delay_mult": float(opt.position_lr_delay_mult),
            "position_max_steps": int(opt.position_lr_max_steps),
            "feature": float(opt.feature_lr),
            "opacity": float(opt.opacity_lr),
            "scaling": float(opt.scaling_lr),
            "rotation": float(opt.rotation_lr),
        },
        "non_position_schedule": {
            "decay_from_iter": int(opt.non_position_lr_decay_from),
            "final_multiplier": float(opt.non_position_lr_final_mult),
        },
        "loss": {
            "lambda_dssim": float(opt.lambda_dssim),
            "lambda_dist": float(opt.lambda_dist),
            "lambda_normal": float(opt.lambda_normal),
        },
        "topology": {
            "percent_dense": float(opt.percent_dense),
            "opacity_cull": float(opt.opacity_cull),
            "densify_from_iter": int(opt.densify_from_iter),
            "densify_until_iter": int(opt.densify_until_iter),
            "densification_interval": int(opt.densification_interval),
            "densify_grad_threshold": float(opt.densify_grad_threshold),
            "opacity_reset_interval": int(opt.opacity_reset_interval),
        },
    }


def _fixed_camera_schedule(
    views: list,
    *,
    iterations: int,
    seed: int,
) -> tuple[list[int], dict]:
    """Build a locally seeded, epoch-balanced camera stream for an ablation.

    The stock 2DGS/ULF-Loc sampler draws a random permutation without
    replacement each epoch.  That is a reasonable optimization policy, but a
    strict ablation must not depend on unrelated consumers of Python's global
    RNG.  This helper owns a dedicated RNG and records the complete effective
    sequence digest, so two runs with the same camera set, seed, and iteration
    count have exactly the same RGB-view stream.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive for a fixed camera schedule")
    if not views:
        raise ValueError("Cannot build a camera schedule without training views")

    names = [str(view.image_name) for view in views]
    if len(names) != len(set(names)):
        raise ValueError("Camera image_name values must be unique for a fixed schedule")

    rng = random.Random(int(seed))
    indices: list[int] = []
    epoch_size = len(views)
    while len(indices) < iterations:
        epoch = list(range(epoch_size))
        rng.shuffle(epoch)
        indices.extend(epoch)
    indices = indices[:iterations]
    sequence_names = [names[index] for index in indices]
    counts = Counter(sequence_names)
    payload = "\n".join(sequence_names).encode("utf-8")
    return indices, {
        "protocol": "local_seeded_epoch_permutation_v1",
        "seed": int(seed),
        "scheduled_iterations": int(iterations),
        "training_camera_count": int(epoch_size),
        "full_epochs": int(iterations // epoch_size),
        "tail_length": int(iterations % epoch_size),
        "unique_cameras_seen": int(len(counts)),
        "per_camera_count_min": int(min(counts.values())),
        "per_camera_count_max": int(max(counts.values())),
        "sequence_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _ulfloc_clean_main_camera_schedule(
    views: list,
    *,
    iterations: int,
    seed: int = 0,
) -> tuple[list[int], dict]:
    """Reproduce clean ULF-Loc main's Python-camera sampler without editing it.

    The released main first calls ``safe_state()``, which seeds Python's
    global RNG with zero, then ``Scene(..., shuffle=True)`` shuffles the
    sorted train-camera list. It consumes one otherwise unused ``randint``
    before entering the loop and subsequently pops a random element from a
    persistent per-epoch stack. This is not the same distribution of exact
    sequences as an independently shuffled epoch permutation. Recreate it
    with a local RNG so a G4 control can use the identical camera identities
    while the clean external source remains untouched.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive for a ULF-Loc camera schedule")
    if not views:
        raise ValueError("Cannot build a ULF-Loc camera schedule without training views")

    names = [str(view.image_name) for view in views]
    if len(names) != len(set(names)):
        raise ValueError("Camera image_name values must be unique for a ULF-Loc camera schedule")

    rng = random.Random(int(seed))
    external_scene_indices = list(range(len(views)))
    rng.shuffle(external_scene_indices)
    # ``train.py`` performs this draw immediately after constructing Scene.
    # The selected camera is overwritten before the first optimization step,
    # but the RNG consumption is part of the public main-branch trajectory.
    unused_initial_index = rng.randint(0, len(external_scene_indices) - 1)

    indices: list[int] = []
    stack: list[int] = []
    while len(indices) < iterations:
        if not stack:
            stack = list(external_scene_indices)
        indices.append(stack.pop(rng.randint(0, len(stack) - 1)))

    sequence_names = [names[index] for index in indices]
    counts = Counter(sequence_names)
    payload = "\n".join(sequence_names).encode("utf-8")
    scene_payload = "\n".join(names[index] for index in external_scene_indices).encode("utf-8")
    return indices, {
        "protocol": "clean_ulfloc_main_scene_shuffle_randint_pop_v1",
        "seed": int(seed),
        "scheduled_iterations": int(iterations),
        "training_camera_count": int(len(views)),
        "scene_shuffle": "python_random_shuffle_after_safe_state",
        "preloop_unused_camera_draw": True,
        "preloop_unused_camera_name": names[external_scene_indices[unused_initial_index]],
        "scene_order_sha256": hashlib.sha256(scene_payload).hexdigest(),
        "full_epochs": int(iterations // len(views)),
        "tail_length": int(iterations % len(views)),
        "unique_cameras_seen": int(len(counts)),
        "per_camera_count_min": int(min(counts.values())),
        "per_camera_count_max": int(max(counts.values())),
        "sequence_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _psnr(image: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((image - target).square()).item()
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def _affine_color_audit(
    color_correction: PerImageAffineColorCorrection,
) -> dict:
    """Summarize the learned train-camera appearance branch without RGB data.

    These scalars make it possible to distinguish a modest exposure alignment
    from an unconstrained color transform.  The actual model is saved beside
    the PLY; this summary is deliberately diagnostic rather than a substitute
    for it.
    """
    with torch.no_grad():
        scales = torch.exp(color_correction.log_scales.detach()).reshape(-1, 3)
        biases = color_correction.biases.detach().reshape(-1, 3)
        return {
            "camera_count": int(scales.shape[0]),
            "scale_mean_rgb": [float(value) for value in scales.mean(dim=0).cpu()],
            "scale_std_rgb": [float(value) for value in scales.std(dim=0, unbiased=False).cpu()],
            "scale_min_rgb": [float(value) for value in scales.min(dim=0).values.cpu()],
            "scale_max_rgb": [float(value) for value in scales.max(dim=0).values.cpu()],
            "bias_mean_rgb": [float(value) for value in biases.mean(dim=0).cpu()],
            "bias_std_rgb": [float(value) for value in biases.std(dim=0, unbiased=False).cpu()],
            "bias_min_rgb": [float(value) for value in biases.min(dim=0).values.cpu()],
            "bias_max_rgb": [float(value) for value in biases.max(dim=0).values.cpu()],
            "identity_regularization": float(
                color_correction.identity_regularization().detach().cpu()
            ),
        }


def _save_image(image: torch.Tensor, path: Path) -> None:
    array = (
        image.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255.0
        + 0.5
    ).astype(np.uint8)
    Image.fromarray(array).save(path)


def _target_stems(path: Path | None) -> set[str]:
    if path is None:
        return set()
    names = set()
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        names.add(Path(value.replace("/", "__")).stem)
    return names


def _ulfloc_masks(
    lookup: CambridgeMaskLookup,
    image_name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the three raw channels used by upstream ULF-Loc's RGB protocol."""
    source_name = lookup.source_name_for(image_name)
    masks = lookup.masks[source_name]
    if len(masks) < 3:
        raise RuntimeError(
            "ULF-Loc pixel supervision requires object/sky/distortion channels "
            f"[0, 1, 2], but {source_name} has only {len(masks)} mask channels"
        )
    return tuple(
        tensor_to_resized_mask(masks[index], shape, device)
        for index in (0, 1, 2)
    )


def _rigid_tree_excluded_mask(
    lookup: CambridgeMaskLookup,
    image_name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Return pixels owned exclusively by the native rigid surface.

    Cambridge's first three protocol channels do not classify a static tree
    as invalid.  A model trained with only those channels is a strong standard
    2DGS baseline, but it is not a rigid-only scaffold for a later 3D foliage
    owner.  The fourth channel in ``masks_with_tree.pkl`` is true outside tree
    pixels; intersecting all four channels makes surface ownership explicit
    from iteration one.
    """
    source_name = lookup.source_name_for(image_name)
    masks = lookup.masks[source_name]
    if len(masks) < 4:
        raise RuntimeError(
            "rigid_tree_excluded supervision requires object/sky/distortion/"
            f"tree channels [0,1,2,3], but {source_name} has {len(masks)}"
        )
    keep = [
        tensor_to_resized_mask(masks[index], shape, device)
        for index in (0, 1, 2, 3)
    ]
    return keep[0] & keep[1] & keep[2] & keep[3]


def _mask_resolution_audit(
    lookup: CambridgeMaskLookup,
    views: list,
    *,
    mask_indices: tuple[int, ...],
) -> dict:
    """Describe whether a masked control is pixel-native or resampled.

    The released ULF-Loc loop directly multiplies its loaded RGB tensor by
    ``masks.pkl`` tensors.  It is therefore only a literal upstream-pixel
    reproduction when the raw masks have the same spatial shape as the loaded
    cameras.  Our general Cambridge lookup deliberately supports nearest
    resizing, which is useful for controlled G4 experiments but must not be
    silently labelled as an exact ULF-Loc run.
    """
    if not views:
        raise RuntimeError("Cannot audit mask resolution without train cameras")
    target_shapes = Counter(
        tuple(int(size) for size in view.original_image.shape[-2:])
        for view in views
    )
    raw_shapes = Counter()
    resampled_views = 0
    for view in views:
        source_name = lookup.source_name_for(view.image_name)
        masks = lookup.masks[source_name]
        target_shape = tuple(int(size) for size in view.original_image.shape[-2:])
        selected_shapes = []
        for index in mask_indices:
            if index < 0 or index >= len(masks):
                raise RuntimeError(
                    f"{source_name} does not provide required mask channel {index}"
                )
            raw_shape = tuple(int(size) for size in masks[index].shape[-2:])
            raw_shapes[raw_shape] += 1
            selected_shapes.append(raw_shape)
        if any(shape != target_shape for shape in selected_shapes):
            resampled_views += 1
    return {
        "mask_indices": list(mask_indices),
        "loaded_image_shape_counts": {
            f"{height}x{width}": count
            for (height, width), count in sorted(target_shapes.items())
        },
        "raw_mask_shape_counts_per_channel": {
            f"{height}x{width}": count
            for (height, width), count in sorted(raw_shapes.items())
        },
        "views_requiring_mask_resampling": resampled_views,
        "native_pixel_geometry": resampled_views == 0,
        "contract": (
            "native_mask_pixels"
            if resampled_views == 0
            else "nearest_resampled_mask_pixels"
        ),
    }


def _evaluate(
    *,
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    background: torch.Tensor,
    output_dir: Path,
    target_stems: set[str],
    evaluate_all: bool,
    color_correction: PerImageAffineColorCorrection | None = None,
) -> dict:
    views = scene.getTrainCameras()
    selected = views if evaluate_all else [view for view in views if view.image_name in target_stems]
    missing = target_stems - {view.image_name for view in views}
    if missing:
        raise RuntimeError(f"Requested target images are not all training cameras: {sorted(missing)}")
    # A training-only control is valid without diagnostic target images.  Keep
    # the evaluation directory contract stable in that case because ``main``
    # always writes the final metrics record there.
    output_dir.mkdir(parents=True, exist_ok=True)
    if not selected:
        return {"evaluated_views": 0, "per_view": {}}

    renders_dir = output_dir / "renders"
    gt_dir = output_dir / "gt"
    renders_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    per_view: dict[str, dict] = {}
    total_mse = 0.0
    with torch.no_grad():
        for view in tqdm(selected, desc="evaluate RGB", leave=False):
            raw_image = render(view, gaussians, pipe, background, rgb_only=True)["render"]
            image = apply_per_image_affine_color_correction(
                color_correction,
                raw_image,
                view.image_name,
            ).clamp(0.0, 1.0)
            target = view.original_image.cuda().clamp(0.0, 1.0)
            mse = torch.mean((image - target).square()).item()
            per_view[view.image_name] = {"psnr": _psnr(image, target), "mse": mse}
            total_mse += mse
            # ``selected`` already is the exact requested population: all
            # cameras for ``--evaluate-all`` and only the named diagnostics
            # otherwise.  The former inverted condition saved no full-view
            # renders when the diagnostic list was empty, leaving scalar
            # PSNR without the image corpus required by strict region/SSIM/
            # MAE evaluation.  Persist every selected pair.
            _save_image(image, renders_dir / f"{view.image_name}.png")
            _save_image(target, gt_dir / f"{view.image_name}.png")
    return {
        "evaluated_views": len(selected),
        "mean_mse": total_mse / len(selected),
        "mean_psnr_from_mean_mse": float(-10.0 * np.log10(max(total_mse / len(selected), 1e-12))),
        "appearance_mode": (
            "train_camera_affine" if color_correction is not None else "raw_gaussian"
        ),
        "per_view": per_view,
    }


def _parse_args() -> tuple[argparse.Namespace, object, object, object]:
    parser = argparse.ArgumentParser(
        description="Direct all-real-camera 2DGS control for strict Cambridge ablations."
    )
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    # ULF-Loc and STDLoc misleadingly *declare* 0.05 in OptimizationParams,
    # but their released 2DGS loops actually pass a literal 0.005 to
    # densify_and_prune.  The standard all-real control follows executed
    # reference semantics by default; callers can still override the value
    # explicitly and every run records the effective threshold.
    parser.set_defaults(opacity_cull=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic-cudnn",
        action="store_true",
        help=(
            "Use the deterministic CuDNN configuration set by clean "
            "ULF-Loc/STDLoc main before their safe_state() call. Record the "
            "effective CuDNN flags so this runtime choice cannot remain an "
            "untracked cross-framework difference."
        ),
    )
    parser.add_argument(
        "--camera-schedule-seed",
        type=int,
        default=None,
        help=(
            "Use a separately seeded, recorded per-epoch camera permutation. "
            "This is the strict-ablation mode: the full RGB camera sequence is "
            "independent of Python's global RNG and identical across runs with "
            "the same source cameras, schedule seed, and iteration count."
        ),
    )
    parser.add_argument(
        "--camera-schedule-protocol",
        choices=("local_epoch_permutation", "clean_ulfloc_main"),
        default="local_epoch_permutation",
        help=(
            "Camera sampler for a recorded control. local_epoch_permutation "
            "is the independent strict-ablation schedule; clean_ulfloc_main "
            "reproduces the released ULF-Loc main's seeded Scene shuffle, "
            "unused pre-loop draw, and randint-pop sequence."
        ),
    )
    parser.add_argument(
        "--init-ply",
        type=Path,
        default=None,
        help=(
            "Optional all-real 2DGS warm start. The PLY parameters are loaded "
            "before training; optimizer and densification statistics are "
            "intentionally reinitialized, matching a refinement-stage handoff."
        ),
    )
    parser.add_argument(
        "--camera-only-scene",
        action="store_true",
        help=(
            "Load fixed COLMAP cameras and RGB only, without opening or "
            "copying points3D. This requires --init-ply and is the no-COLMAP-"
            "geometry rigid-first path."
        ),
    )
    parser.add_argument(
        "--resume-training-state",
        type=Path,
        default=None,
        help=(
            "Resume an exact optimizer/RNG/model state written by this runner. "
            "Unlike --init-ply, this is an auditable causal fork and requires "
            "the identical fixed camera schedule and input contract."
        ),
    )
    parser.add_argument(
        "--state-checkpoint-iterations",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Save full model, optimizer, densification, and RNG state after "
            "these iterations. Use with --stop-after-iteration to create an "
            "exact branch point without changing the declared LR horizon."
        ),
    )
    parser.add_argument(
        "--stop-after-iteration",
        type=int,
        default=None,
        help=(
            "End a staging run after this completed iteration while retaining "
            "the declared --iterations horizon. The stop iteration is always "
            "written as a full training-state checkpoint; no final render is "
            "emitted because it would not correspond to a saved PLY snapshot."
        ),
    )
    parser.add_argument("--save-iterations", type=int, nargs="*", default=[])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--continue-opacity-resets-after-densify",
        action="store_true",
        help=(
            "Experimental causal-control switch: continue the declared opacity "
            "reset cadence after densification ends. The default preserves the "
            "native 2DGS/ULF-Loc schedule, where resets stop with densification."
        ),
    )
    parser.add_argument(
        "--normal-weight",
        type=float,
        default=0.0,
        help="Optional native 2DGS normal-consistency weight; zero keeps RGB-only control.",
    )
    parser.add_argument("--normal-from", type=int, default=0)
    parser.add_argument(
        "--distortion-weight",
        type=float,
        default=0.0,
        help="Optional native 2DGS distortion weight; zero keeps RGB-only control.",
    )
    parser.add_argument("--distortion-from", type=int, default=0)
    parser.add_argument(
        "--target-images",
        type=Path,
        default=None,
        help="Optional text list of original or flattened target image names for diagnostic renders.",
    )
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        help="Render and score all real training cameras after the final iteration.",
    )
    parser.add_argument(
        "--evaluate-at-save",
        action="store_true",
        help=(
            "Also render requested diagnostic cameras at every saved checkpoint. "
            "This does not alter optimization and makes 1k/3k/7k comparisons explicit."
        ),
    )
    parser.add_argument(
        "--supervision-profile",
        choices=(
            "rgb",
            "ulfloc_masked",
            "g4_hard_masked",
            "g4_static_hybrid",
            "rigid_tree_excluded",
        ),
        default="rgb",
        help=(
            "rgb is a pure real-RGB control; ulfloc_masked exactly reproduces "
            "ULF-Loc's object/distortion/sky operation order after any needed "
            "mask resizing; g4_hard_masked "
            "reproduces G4's hard [0,1,2] masked RGB objective without Charts; "
            "g4_static_hybrid mixes the RGB and hard-static objectives; "
            "rigid_tree_excluded requires a four-channel tree pickle and "
            "trains only object/sky/distortion/tree-valid rigid pixels."
        ),
    )
    parser.add_argument(
        "--static-loss-mix",
        type=float,
        default=0.25,
        help=(
            "For g4_static_hybrid, interpolation weight from full RGB (0) to "
            "hard static [0,1,2] RGB loss (1)."
        ),
    )
    parser.add_argument(
        "--cambridge-mask-pickle",
        type=Path,
        default=None,
        help="Required by any non-rgb --supervision-profile.",
    )
    parser.add_argument(
        "--cambridge-mask-dataset-path",
        type=Path,
        default=None,
        help="Dataset carrying name_mapping.json for the external Cambridge masks.",
    )
    parser.add_argument(
        "--require-native-ulfloc-mask-resolution",
        action="store_true",
        help=(
            "Fail unless ULF-Loc mask channels already match every loaded RGB "
            "camera.  Use this for a literal upstream-pixel ULF-Loc control; "
            "leave it off for a resolution-normalized semantic ablation."
        ),
    )
    parser.add_argument(
        "--allow-nonreference-background",
        action="store_true",
        help=(
            "Permit a non-white background with ulfloc_masked supervision. "
            "This is an explicitly non-reference ablation: released ULF-Loc "
            "and STDLoc Cambridge defaults use a white background."
        ),
    )
    parser.add_argument(
        "--use-color-correction",
        action="store_true",
        help=(
            "Enable a regularized diagonal RGB affine correction for each real "
            "training camera. This is an explicit non-parity appearance "
            "ablation; unknown/test camera names receive identity."
        ),
    )
    parser.add_argument(
        "--color-correction-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for the optional per-image affine branch.",
    )
    parser.add_argument(
        "--color-correction-reg",
        type=float,
        default=1e-2,
        help="Identity regularization weight for the optional affine branch.",
    )
    args = parser.parse_args()
    dataset = model_params.extract(args)
    opt = optimization_params.extract(args)
    pipe = pipeline_params.extract(args)
    if not dataset.source_path or not dataset.model_path:
        parser.error("Both --source_path/-s and --model_path/-m are required")
    if opt.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.resume_training_state is not None and args.init_ply is not None:
        parser.error("--resume-training-state and --init-ply are mutually exclusive")
    if args.camera_only_scene and args.init_ply is None:
        parser.error("--camera-only-scene requires --init-ply")
    invalid_state_iterations = [
        iteration
        for iteration in args.state_checkpoint_iterations
        if iteration < 1 or iteration > opt.iterations
    ]
    if invalid_state_iterations:
        parser.error(
            "--state-checkpoint-iterations must lie in [1, --iterations], got "
            f"{invalid_state_iterations}"
        )
    if (
        args.stop_after_iteration is not None
        and not 1 <= args.stop_after_iteration <= opt.iterations
    ):
        parser.error("--stop-after-iteration must lie in [1, --iterations]")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.normal_weight < 0 or args.distortion_weight < 0:
        parser.error("native regularization weights must be non-negative")
    if args.supervision_profile != "rgb" and args.cambridge_mask_pickle is None:
        parser.error("A non-rgb --supervision-profile requires --cambridge-mask-pickle")
    if (
        args.supervision_profile == "ulfloc_masked"
        and not dataset.white_background
        and not args.allow_nonreference_background
    ):
        parser.error(
            "ulfloc_masked reference protocol requires --white_background; "
            "pass --allow-nonreference-background only for a declared "
            "non-reference ablation"
        )
    if (
        args.require_native_ulfloc_mask_resolution
        and args.supervision_profile != "ulfloc_masked"
    ):
        parser.error(
            "--require-native-ulfloc-mask-resolution applies only to "
            "--supervision-profile ulfloc_masked"
        )
    if not 0.0 <= args.static_loss_mix <= 1.0:
        parser.error("--static-loss-mix must lie in [0, 1]")
    if args.color_correction_lr <= 0.0:
        parser.error("--color-correction-lr must be positive")
    if args.color_correction_reg < 0.0:
        parser.error("--color-correction-reg must be non-negative")
    return args, dataset, opt, pipe


def main() -> None:
    args, dataset, opt, pipe = _parse_args()
    source_path = Path(dataset.source_path).resolve()
    model_path = Path(dataset.model_path).resolve()
    if model_path.exists() and any(model_path.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a new single-variable control with existing output: {model_path}"
        )
    model_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    if args.deterministic_cudnn:
        # clean ULF-Loc/STDLoc call ``seed_everything`` before ``safe_state``;
        # safe_state resets the random seeds but deliberately leaves these
        # CuDNN reproducibility flags in place.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    gaussians = GaussianModel(dataset.sh_degree)
    scene = (
        LazyScene(
            dataset,
            gaussians,
            image_cache_size=8,
            image_prefetch_workers=2,
        )
        if args.camera_only_scene
        else Scene(dataset, gaussians, shuffle=False)
    )
    views = scene.getTrainCameras()
    input_audit = _input_audit(source_path, views)
    train_image_names = [str(view.image_name) for view in views]
    if len(train_image_names) != len(set(train_image_names)):
        raise RuntimeError("Loaded training cameras do not have unique image names")
    sparse_ply_audit = (
        {
            "scene_sparse_ply": None,
            "scene_sparse_ply_sha256": None,
            "scene_sparse_ply_bytes": 0,
        }
        if args.camera_only_scene
        else _scene_input_ply_audit(model_path)
    )
    input_audit.update(sparse_ply_audit)
    fixed_camera_schedule = None
    if args.camera_schedule_protocol == "clean_ulfloc_main":
        fixed_camera_schedule, camera_schedule_audit = _ulfloc_clean_main_camera_schedule(
            views,
            iterations=int(opt.iterations),
            # ULF-Loc's clean main reaches safe_state(), which uses zero.
            seed=0 if args.camera_schedule_seed is None else int(args.camera_schedule_seed),
        )
    elif args.camera_schedule_seed is None:
        camera_schedule_audit = {
            "protocol": "legacy_global_rng_epoch_permutation",
            "seed": int(args.seed),
            "scheduled_iterations": int(opt.iterations),
            "training_camera_count": int(len(views)),
        }
    else:
        fixed_camera_schedule, camera_schedule_audit = _fixed_camera_schedule(
            views,
            iterations=int(opt.iterations),
            seed=int(args.camera_schedule_seed),
        )
    state_fork_requested = (
        args.resume_training_state is not None
        or bool(args.state_checkpoint_iterations)
        or args.stop_after_iteration is not None
    )
    if state_fork_requested and fixed_camera_schedule is None:
        raise RuntimeError(
            "Exact training-state forks require a recorded fixed camera schedule; "
            "pass --camera-schedule-seed or --camera-schedule-protocol clean_ulfloc_main"
        )
    input_audit["camera_schedule"] = camera_schedule_audit
    input_audit["opacity_reset_iterations"] = [
        iteration
        for iteration in range(1, int(opt.iterations) + 1)
        if opacity_reset_due(
            iteration,
            opacity_reset_interval=int(opt.opacity_reset_interval),
            densify_from_iter=int(opt.densify_from_iter),
            densify_until_iter=int(opt.densify_until_iter),
            white_background=bool(dataset.white_background),
            continue_after_densify=bool(args.continue_opacity_resets_after_densify),
        )
    ]
    resume_state_path = None
    resume_state_digest = None
    resume_state_payload = None
    if args.resume_training_state is not None:
        resume_state_path = args.resume_training_state.resolve()
        resume_state_digest = _file_digest(resume_state_path)
        resume_state_payload = _load_training_state(resume_state_path)

    init_ply = None
    init_ply_digest = None
    if args.init_ply is not None:
        init_ply = args.init_ply.resolve()
        if not init_ply.is_file():
            raise FileNotFoundError(f"Warm-start PLY does not exist: {init_ply}")
        init_ply_digest = _file_digest(init_ply)
        gaussians.load_ply(str(init_ply))
        # A standard control must not inherit optional MIP filtering merely
        # because a foreign warm-start PLY happened to serialize that field.
        # The raw PLY parameters remain intact; only the renderer path is
        # restored to the declared standard 2DGS protocol.
        gaussians.set_mip_filter(False)
        gaussians.max_radii2D = torch.zeros(
            (gaussians.get_xyz.shape[0],),
            device=gaussians.get_xyz.device,
        )
    effective_init_ply = (
        init_ply
        if init_ply is not None
        else Path(sparse_ply_audit["scene_sparse_ply"])
    )
    effective_init_ply_digest = (
        init_ply_digest or sparse_ply_audit["scene_sparse_ply_sha256"]
    )
    mask_lookup = None
    mask_dataset_path = None
    mask_resolution_audit = None
    if args.supervision_profile != "rgb":
        mask_dataset_path = (
            args.cambridge_mask_dataset_path.resolve()
            if args.cambridge_mask_dataset_path is not None
            else source_path
        )
        mask_indices = (
            [0, 1, 2, 3]
            if args.supervision_profile == "rigid_tree_excluded"
            else (
                [0, 1, 2]
                if args.supervision_profile
                in {"g4_hard_masked", "g4_static_hybrid"}
                else [0]
            )
        )
        mask_lookup = CambridgeMaskLookup(
            mask_dataset_path,
            args.cambridge_mask_pickle.resolve(),
            # The raw tuple is read below; this only establishes the stable
            # staged-name -> source-name mapping.
            mask_indices=mask_indices,
        )
        mask_resolution_audit = _mask_resolution_audit(
            mask_lookup,
            views,
            mask_indices=(0, 1, 2)
            if args.supervision_profile == "ulfloc_masked"
            else tuple(mask_indices),
        )
        if (
            args.require_native_ulfloc_mask_resolution
            and not mask_resolution_audit["native_pixel_geometry"]
        ):
            raise RuntimeError(
                "The requested literal ULF-Loc pixel protocol would resize "
                f"masks for {mask_resolution_audit['views_requiring_mask_resampling']}/"
                f"{len(views)} cameras.  Use masks prepared at the training "
                "resolution, or omit --require-native-ulfloc-mask-resolution "
                "and treat this as a resolution-normalized ablation."
            )
    color_correction = None
    color_correction_optimizer = None
    if args.use_color_correction:
        color_correction = PerImageAffineColorCorrection(train_image_names).cuda()
        color_correction_optimizer = torch.optim.Adam(
            color_correction.parameters(),
            lr=args.color_correction_lr,
        )
    implementation_audit = _implementation_audit()
    input_audit.update(
        {
            "experiment": "standard_full_view_2dgs",
            "objective": "all_real_RGB_from_iteration_1",
            "camera_container_only": bool(args.camera_only_scene),
            "colmap_points_or_tracks_used": not bool(
                args.camera_only_scene
            ),
            "initialization": (
                "training_state_resume"
                if resume_state_payload is not None
                else ("ply_warm_start" if init_ply is not None else "sparse_pcd")
            ),
            "init_ply": None if init_ply is None else str(init_ply),
            "init_ply_sha256": init_ply_digest,
            "effective_initialization_ply": (
                None if resume_state_payload is not None else str(effective_init_ply)
            ),
            "effective_initialization_ply_sha256": (
                None if resume_state_payload is not None else effective_init_ply_digest
            ),
            "init_gaussians": None,
            "optimizer_state": (
                "restored_exact_training_state"
                if resume_state_payload is not None
                else "fresh_for_this_control"
            ),
            "charts": "disabled",
            "dense_depth_prior": "disabled",
            "planes": "disabled",
            "see3d": "disabled",
            "semantic_masks": (
                "disabled"
                if mask_lookup is None
                else (
                    "ulfloc_object_distortion_sky_operation_order_channels_0_1_2"
                    if args.supervision_profile == "ulfloc_masked"
                    else (
                        "rigid_intersection_channels_0_1_2_3_tree_excluded"
                        if args.supervision_profile
                        == "rigid_tree_excluded"
                        else "g4_hard_and_channels_0_1_2"
                    )
                )
            ),
            "supervision_profile": args.supervision_profile,
            "surface_ownership_contract": (
                "rigid_pixels_tree_sky_transient_excluded"
                if args.supervision_profile == "rigid_tree_excluded"
                else "standard_profile_may_include_static_tree"
            ),
            "static_loss_mix": (
                args.static_loss_mix
                if args.supervision_profile == "g4_static_hybrid"
                else None
            ),
            "cambridge_mask_pickle": (
                None if args.cambridge_mask_pickle is None else str(args.cambridge_mask_pickle.resolve())
            ),
            "cambridge_mask_pickle_sha256": (
                None if args.cambridge_mask_pickle is None else _file_digest(args.cambridge_mask_pickle)
            ),
            "cambridge_mask_dataset_path": (
                None if mask_dataset_path is None else str(mask_dataset_path)
            ),
            "mask_resolution_audit": mask_resolution_audit,
            "ulfloc_native_pixel_protocol": (
                None
                if args.supervision_profile != "ulfloc_masked"
                else bool(mask_resolution_audit["native_pixel_geometry"])
            ),
            "white_background": bool(dataset.white_background),
            "allow_nonreference_background": bool(args.allow_nonreference_background),
            "per_image_affine_color": {
                "enabled": bool(args.use_color_correction),
                "type": "diagonal_rgb_scale_plus_bias",
                "scope": "real_training_cameras_only",
                "unknown_camera_policy": "identity",
                "train_camera_count": int(len(train_image_names)),
                "train_camera_name_set_sha256": _set_digest(set(train_image_names)),
                "optimizer": "Adam" if args.use_color_correction else None,
                "learning_rate": (
                    float(args.color_correction_lr) if args.use_color_correction else None
                ),
                "identity_regularization_weight": (
                    float(args.color_correction_reg) if args.use_color_correction else None
                ),
            },
            "normal_weight": args.normal_weight,
            "normal_from": args.normal_from,
            "distortion_weight": args.distortion_weight,
            "distortion_from": args.distortion_from,
            "opacity_cull": float(opt.opacity_cull),
            "densify_from_iter": int(opt.densify_from_iter),
            "densify_until_iter": int(opt.densify_until_iter),
            "densification_interval": int(opt.densification_interval),
            "opacity_reset_interval": int(opt.opacity_reset_interval),
            "optimization": _optimization_audit(opt),
            "continue_opacity_resets_after_densify": bool(
                args.continue_opacity_resets_after_densify
            ),
            "mip_filter": "disabled",
            "renderer": "diff_surfel_rasterization",
            "implementation": implementation_audit,
            "evaluate_at_save": args.evaluate_at_save,
            "iterations": opt.iterations,
            "resolution": dataset.resolution,
            "seed": args.seed,
            "cudnn": {
                "deterministic": bool(torch.backends.cudnn.deterministic),
                "benchmark": bool(torch.backends.cudnn.benchmark),
            },
            "training_state": {
                "protocol": "standard_2dgs_exact_optimizer_rng_fork_v1",
                "state_checkpoint_version": 2,
                "resume_training_state": (
                    None if resume_state_path is None else str(resume_state_path)
                ),
                "resume_training_state_sha256": resume_state_digest,
                "requested_state_checkpoint_iterations": sorted(
                    {int(iteration) for iteration in args.state_checkpoint_iterations}
                ),
                "stop_after_iteration": args.stop_after_iteration,
                "declared_total_iterations": int(opt.iterations),
            },
        }
    )

    state_contract = _training_state_contract(
        input_audit,
        camera_schedule_audit,
        declared_total_iterations=int(opt.iterations),
    )
    if resume_state_payload is not None:
        mismatches = _contract_mismatches(
            state_contract,
            resume_state_payload["contract"],
        )
        if mismatches:
            raise RuntimeError(
                "Training-state contract mismatch; refusing a non-causal resume:\n  - "
                + "\n  - ".join(mismatches[:12])
            )
        gaussians.restore(resume_state_payload["model_capture"], opt)
        gaussians.set_mip_filter(False)
        _restore_color_correction_state(
            resume_state_payload,
            color_correction=color_correction,
            color_correction_optimizer=color_correction_optimizer,
        )
        _restore_rng_state(resume_state_payload["rng_state"])
        resume_ema = float(resume_state_payload["training_loop_state"]["ema"])
        start_iteration = int(resume_state_payload["iteration"]) + 1
    else:
        gaussians.training_setup(opt)
        resume_ema = 0.0
        start_iteration = 1

    run_end_iteration = (
        int(args.stop_after_iteration)
        if args.stop_after_iteration is not None
        else int(opt.iterations)
    )
    if start_iteration > run_end_iteration:
        raise RuntimeError(
            "Training-state resume starts after this run's requested end: "
            f"start={start_iteration}, end={run_end_iteration}"
        )
    state_checkpoint_iterations = {
        int(iteration) for iteration in args.state_checkpoint_iterations
    }
    if args.stop_after_iteration is not None:
        state_checkpoint_iterations.add(run_end_iteration)
    invalid_for_this_run = sorted(
        iteration
        for iteration in state_checkpoint_iterations
        if iteration < start_iteration or iteration > run_end_iteration
    )
    if invalid_for_this_run:
        raise RuntimeError(
            "Requested training-state checkpoint iterations are outside this run's "
            f"range [{start_iteration}, {run_end_iteration}]: {invalid_for_this_run}"
        )
    input_audit["init_gaussians"] = int(gaussians.get_xyz.shape[0])
    input_audit["training_state"].update(
        {
            "state_checkpoint_iterations": sorted(state_checkpoint_iterations),
            "start_iteration": int(start_iteration),
            "run_end_iteration": int(run_end_iteration),
            "contract_sha256": _json_digest(state_contract),
        }
    )
    (model_path / "input_manifest.json").write_text(json.dumps(input_audit, indent=2) + "\n")
    # Keep the historical ``Namespace(...)`` cfg format renderer-compatible:
    # raw ``Path`` values stringify as ``PosixPath(...)`` and older renderers
    # cannot evaluate that constructor.
    cfg_values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (model_path / "cfg_args").write_text(str(argparse.Namespace(**cfg_values)) + "\n")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    saves = set(args.save_iterations)
    saves.add(opt.iterations)
    trace_path = model_path / "training_trace.jsonl"
    trace_path.touch()
    stack: list = []
    saved_training_states: list[Path] = []
    targets = _target_stems(args.target_images)
    ema = resume_ema
    progress = tqdm(
        range(start_iteration, run_end_iteration + 1),
        desc="standard full-view 2DGS",
    )
    start = time.time()

    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if fixed_camera_schedule is not None:
            view = views[fixed_camera_schedule[iteration - 1]]
        else:
            if not stack:
                stack = list(views)
                random.shuffle(stack)
            view = stack.pop()
        package = render(view, gaussians, pipe, background)
        image = package["render"]
        target = view.original_image.cuda()
        standard_mask = None
        image_for_loss = apply_per_image_affine_color_correction(
            color_correction,
            image,
            view.image_name,
        )
        target_for_loss = target
        if args.supervision_profile == "ulfloc_masked":
            object_mask, sky_mask, distortion_mask = _ulfloc_masks(
                mask_lookup,
                view.image_name,
                image.shape[-2:],
                image.device,
            )
            image_for_loss, target_for_loss, standard_mask = ulfloc_masked_supervision(
                image_for_loss,
                target,
                object_mask=object_mask,
                sky_mask=sky_mask,
                distortion_mask=distortion_mask,
            )
            rgb_l1 = l1_loss(image_for_loss, target_for_loss)
            rgb_loss = (1.0 - opt.lambda_dssim) * rgb_l1 + opt.lambda_dssim * (
                1.0 - ssim(image_for_loss, target_for_loss)
            )
        elif args.supervision_profile == "rigid_tree_excluded":
            standard_mask = _rigid_tree_excluded_mask(
                mask_lookup,
                view.image_name,
                image.shape[-2:],
                image.device,
            )
            rgb_l1, rgb_loss = compute_rgb_loss(
                image_for_loss,
                target,
                mask=standard_mask,
                lambda_dssim=opt.lambda_dssim,
                ssim_fn=ssim,
            )
        elif args.supervision_profile in {"g4_hard_masked", "g4_static_hybrid"}:
            standard_mask = mask_lookup.get_mask(
                view.image_name, image.shape[-2:], image.device
            )
            static_l1, static_loss = compute_rgb_loss(
                image_for_loss,
                target,
                mask=standard_mask,
                lambda_dssim=opt.lambda_dssim,
                ssim_fn=ssim,
            )
            if args.supervision_profile == "g4_static_hybrid":
                raw_l1 = l1_loss(image_for_loss, target)
                raw_loss = (1.0 - opt.lambda_dssim) * raw_l1 + opt.lambda_dssim * (
                    1.0 - ssim(image_for_loss, target)
                )
                rgb_l1 = (
                    (1.0 - args.static_loss_mix) * raw_l1
                    + args.static_loss_mix * static_l1
                )
                rgb_loss = (
                    (1.0 - args.static_loss_mix) * raw_loss
                    + args.static_loss_mix * static_loss
                )
            else:
                rgb_l1, rgb_loss = static_l1, static_loss
        else:
            rgb_l1 = l1_loss(image_for_loss, target_for_loss)
            rgb_loss = (1.0 - opt.lambda_dssim) * rgb_l1 + opt.lambda_dssim * (
                1.0 - ssim(image_for_loss, target_for_loss)
            )
        normal_loss = image.sum() * 0.0
        distortion_loss = image.sum() * 0.0
        color_correction_loss = image.sum() * 0.0
        if args.normal_weight > 0.0 and iteration >= args.normal_from:
            normal_error = 1.0 - (package["rend_normal"] * package["surf_normal"]).sum(dim=0)
            if standard_mask is not None:
                normal_error = normal_error * standard_mask.to(dtype=normal_error.dtype)
            normal_loss = args.normal_weight * normal_error.mean()
        if args.distortion_weight > 0.0 and iteration >= args.distortion_from:
            distortion_map = package["rend_dist"]
            if standard_mask is not None:
                distortion_map = distortion_map * standard_mask.to(dtype=distortion_map.dtype)
            distortion_loss = args.distortion_weight * distortion_map.mean()
        if color_correction is not None:
            color_correction_loss = (
                args.color_correction_reg * color_correction.identity_regularization()
            )
        loss = rgb_loss + normal_loss + distortion_loss + color_correction_loss
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if iteration % args.log_every == 0 or iteration == 1:
                row = {
                    "iteration": iteration,
                    "rgb_l1": rgb_l1.item(),
                    "rgb_loss": rgb_loss.item(),
                    "normal_loss": normal_loss.item(),
                    "distortion_loss": distortion_loss.item(),
                    "color_correction_loss": color_correction_loss.item(),
                    "total_loss": loss.item(),
                    "ema_total_loss": ema,
                    "gaussians": int(gaussians.get_xyz.shape[0]),
                    "camera": view.image_name,
                    "elapsed_sec": time.time() - start,
                }
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                progress.set_postfix(loss=f"{ema:.5f}", points=len(gaussians.get_xyz))

            # ULF-Loc/STDLoc save the visible checkpoint before their
            # post-backward topology and opacity branches.  This ordering is
            # observable at a requested save iteration (notably the common
            # 7k endpoint, which is also a densification iteration), so keep
            # the control checkpoint on the same side of those mutations.
            if iteration in saves:
                scene.save(iteration)
                if color_correction is not None:
                    save_per_image_affine_color_correction(
                        color_correction,
                        model_path
                        / "point_cloud"
                        / f"iteration_{iteration}"
                        / "color_correction.pth",
                    )
                if args.evaluate_at_save:
                    checkpoint_dir = model_path / "evaluation" / f"iteration_{iteration}"
                    raw_output_dir = (
                        checkpoint_dir / "raw_gaussian"
                        if color_correction is not None
                        else checkpoint_dir
                    )
                    checkpoint_metrics = _evaluate(
                        scene=scene,
                        gaussians=gaussians,
                        pipe=pipe,
                        background=background,
                        output_dir=raw_output_dir,
                        target_stems=targets,
                        evaluate_all=False,
                    )
                    if color_correction is not None:
                        checkpoint_metrics["train_camera_affine"] = _evaluate(
                            scene=scene,
                            gaussians=gaussians,
                            pipe=pipe,
                            background=background,
                            output_dir=checkpoint_dir / "train_camera_affine",
                            target_stems=targets,
                            evaluate_all=False,
                            color_correction=color_correction,
                        )
                        checkpoint_metrics["per_image_affine_color"] = _affine_color_audit(
                            color_correction
                        )
                    checkpoint_metrics.update(
                        {
                            "iteration": iteration,
                            "input": input_audit,
                            "gaussians": int(gaussians.get_xyz.shape[0]),
                            "elapsed_sec": time.time() - start,
                        }
                    )
                    (checkpoint_dir / "metrics.json").write_text(
                        json.dumps(checkpoint_metrics, indent=2) + "\n"
                    )

            # The released references still enter these branches at their
            # terminal iteration, but after writing the final checkpoint and
            # without taking another optimizer step.  The resulting mutation
            # is discarded on process exit.  Skipping it keeps this runner's
            # in-memory final evaluation equal to the persisted reference
            # checkpoint while leaving every optimization-bearing iteration
            # unchanged.
            if iteration < opt.iterations:
                if iteration < opt.densify_until_iter:
                    visibility = package["visibility_filter"]
                    radii = package["radii"]
                    gaussians.max_radii2D[visibility] = torch.maximum(
                        gaussians.max_radii2D[visibility], radii[visibility]
                    )
                    gaussians.add_densification_stats(package["viewspace_points"], visibility)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.opacity_cull,
                            scene.cameras_extent,
                            size_threshold,
                        )
                if opacity_reset_due(
                    iteration,
                    opacity_reset_interval=opt.opacity_reset_interval,
                    densify_from_iter=opt.densify_from_iter,
                    densify_until_iter=opt.densify_until_iter,
                    white_background=dataset.white_background,
                    continue_after_densify=args.continue_opacity_resets_after_densify,
                ):
                    gaussians.reset_opacity()
                gaussians.optimizer.step()
                if color_correction_optimizer is not None:
                    color_correction_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if color_correction_optimizer is not None:
                    color_correction_optimizer.zero_grad(set_to_none=True)

            # A causal fork must capture the state *after* the exact topology,
            # opacity-reset, optimizer-step, and zero-grad sequence that the
            # next iteration would inherit.  This is deliberately separate
            # from ``scene.save`` above, whose pre-topology ordering preserves
            # the external ULF-Loc/STDLoc PLY contract.
            if iteration in state_checkpoint_iterations:
                saved_training_states.append(
                    _save_training_state(
                        model_path=model_path,
                        iteration=iteration,
                        gaussians=gaussians,
                        training_loop_state={"ema": float(ema)},
                        contract=state_contract,
                        color_correction=color_correction,
                        color_correction_optimizer=color_correction_optimizer,
                    )
                )

    if run_end_iteration < opt.iterations:
        stage = {
            "protocol": "standard_2dgs_training_state_stage_v1",
            "declared_total_iterations": int(opt.iterations),
            "stopped_after_iteration": int(run_end_iteration),
            "next_iteration": int(run_end_iteration + 1),
            "final_gaussians": int(gaussians.get_xyz.shape[0]),
            "training_state_checkpoints": [str(path) for path in saved_training_states],
            "input_manifest": str((model_path / "input_manifest.json").resolve()),
            "input_contract_sha256": _json_digest(state_contract),
            "elapsed_sec": time.time() - start,
            "final_render": "intentionally_skipped_no_matching_ply_snapshot",
        }
        (model_path / "training_stage.json").write_text(json.dumps(stage, indent=2) + "\n")
        print(json.dumps(stage, indent=2))
        return

    evaluation_dir = model_path / "evaluation"
    raw_output_dir = (
        evaluation_dir / "raw_gaussian"
        if color_correction is not None
        else evaluation_dir
    )
    metrics = _evaluate(
        scene=scene,
        gaussians=gaussians,
        pipe=pipe,
        background=background,
        output_dir=raw_output_dir,
        target_stems=targets,
        evaluate_all=args.evaluate_all,
    )
    if color_correction is not None:
        metrics["train_camera_affine"] = _evaluate(
            scene=scene,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            output_dir=evaluation_dir / "train_camera_affine",
            target_stems=targets,
            evaluate_all=args.evaluate_all,
            color_correction=color_correction,
        )
        metrics["per_image_affine_color"] = _affine_color_audit(color_correction)
    metrics["input"] = input_audit
    metrics["final_gaussians"] = int(gaussians.get_xyz.shape[0])
    metrics["elapsed_sec"] = time.time() - start
    metrics["rigid_surface_handoff"] = str(
        _write_rigid_surface_handoff(
            model_path=model_path,
            iteration=opt.iterations,
            point_count=int(gaussians.get_xyz.shape[0]),
            input_audit=input_audit,
            init_ply=init_ply,
        )
    )
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
