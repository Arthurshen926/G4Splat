#!/usr/bin/env python3
"""Precompute immutable MoGe3 metric evidence for exact Cambridge cameras.

Run this script in a dedicated MoGe3 environment (Python >= 3.10).  The
resulting NPZ files and index have no runtime MoGe dependency and can be
consumed by G4Splat's training environment.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.moge3_evidence import (  # noqa: E402
    MOGE3_AUDITED_COMMIT,
    atomic_save_view,
    build_index,
    canonical_json_sha256,
    exact_pixel_intrinsics,
    horizontal_fov_degrees,
    load_view,
    make_view_payload,
    sha256_file,
)
from outdoor.scene_contract import build_scene_contract  # noqa: E402


DEFAULT_MODEL = "Ruicheng/moge-3-vitl"


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _resolve_checkpoint(model_id: str, revision: str) -> tuple[Path, str]:
    local = Path(model_id).expanduser()
    if local.is_file():
        digest = sha256_file(local.resolve())
        return local.resolve(), f"local-sha256:{digest}"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to resolve a content-addressed "
            "MoGe3 checkpoint"
        ) from error
    checkpoint = Path(
        hf_hub_download(
            repo_id=model_id,
            repo_type="model",
            filename="model.pt",
            revision=revision,
        )
    ).absolute()
    # Hugging Face resolves refs into .../snapshots/<immutable-commit>/model.pt.
    resolved_revision = checkpoint.parent.name
    if len(resolved_revision) < 12 or resolved_revision in {"main", "master"}:
        raise RuntimeError(
            "Could not recover an immutable Hugging Face snapshot revision "
            f"from {checkpoint}"
        )
    return checkpoint, resolved_revision


def _load_model(checkpoint: Path, device: str):
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "MoGe3 evidence generation requires an independent Python 3.10+ "
            f"environment; current interpreter is {platform.python_version()}"
        )
    try:
        import torch
        import moge
        from moge.model.v3 import MoGeModel
    except ImportError as error:
        raise RuntimeError(
            "MoGe3 is unavailable. Install the official microsoft/MoGe v3 "
            "package and its FlexGEMM/Triton dependencies in this generator "
            "environment; do not add them to the training environment."
        ) from error
    model = MoGeModel.from_pretrained(checkpoint).to(torch.device(device)).eval()
    source_root = Path(moge.__file__).resolve().parents[1]
    return model, torch, {
        "moge_source_root": str(source_root),
        "moge_git_revision": _git_revision(source_root),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    return np.asarray(value)


def _infer_one(
    *,
    model,
    torch,
    image: np.ndarray,
    fov_x: float,
    refine_steps: int,
    resolution_level: int,
    use_fp16: bool,
) -> dict[str, np.ndarray]:
    image_tensor = torch.from_numpy(
        np.array(image, dtype=np.float32, copy=True) / 255.0
    ).permute(2, 0, 1)
    with torch.inference_mode():
        output = model.infer(
            image_tensor,
            fov_x=float(fov_x),
            refine_steps=int(refine_steps),
            return_per_step=True,
            force_projection=True,
            apply_mask=False,
            resolution_level=int(resolution_level),
            use_fp16=bool(use_fp16),
        )
    required = {
        "depth",
        "normal",
        "mask",
        "intrinsics",
        "depth_per_step",
    }
    missing = required - set(output)
    if missing:
        raise RuntimeError(
            "MoGe3 v3 output lacks required evidence: "
            + ", ".join(sorted(missing))
        )
    per_step = output["depth_per_step"]
    if isinstance(per_step, (list, tuple)):
        per_step = np.stack([_to_numpy(value) for value in per_step], axis=0)
    else:
        per_step = _to_numpy(per_step)
    return {
        "depth": _to_numpy(output["depth"]),
        "normal": _to_numpy(output["normal"]),
        "mask": _to_numpy(output["mask"]),
        "intrinsics": _to_numpy(output["intrinsics"]),
        "depth_per_step": per_step,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--scene-contract",
        type=Path,
        help="Existing fixed-camera contract; otherwise one is built in output.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision",
        default="main",
        help=(
            "Hugging Face ref to resolve once; the index records the resolved "
            "immutable snapshot commit and model.pt SHA256."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--replace-views",
        action="store_true",
        help="Atomically replace already valid per-view archives.",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.refine_steps < 0:
        raise ValueError("--refine-steps must be non-negative")
    if not 0 <= args.resolution_level <= 9:
        raise ValueError("--resolution-level must be in [0, 9]")
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--start/--limit must select a positive view range")

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    views_root = output / "views"
    views_root.mkdir(parents=True, exist_ok=True)
    if args.scene_contract is None:
        scene_contract_path = output / "scene_contract.json"
        if not scene_contract_path.is_file():
            build_scene_contract(dataset, scene_contract_path)
    else:
        scene_contract_path = args.scene_contract.expanduser().resolve()
    scene_contract = json.loads(scene_contract_path.read_text(encoding="utf-8"))
    if Path(scene_contract["dataset"]).resolve() != dataset:
        raise RuntimeError("Scene contract belongs to a different dataset")

    checkpoint, resolved_revision = _resolve_checkpoint(
        str(args.model), str(args.model_revision)
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    model, torch, environment = _load_model(checkpoint, str(args.device))
    if environment.get("moge_git_revision") not in {
        MOGE3_AUDITED_COMMIT,
        None,
    }:
        raise RuntimeError(
            "Generator MoGe checkout differs from the audited v3 source: "
            f"{environment['moge_git_revision']} != {MOGE3_AUDITED_COMMIT}"
        )

    all_records = list(scene_contract.get("records", []))
    end = None if args.limit is None else args.start + args.limit
    selected = all_records[args.start:end]
    if not selected:
        raise RuntimeError("Selected MoGe3 view range is empty")
    image_root = dataset / "images"
    generated = 0
    reused = 0
    for position, record in enumerate(selected, start=args.start):
        image_path = image_root / str(record["image_name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_hash = sha256_file(image_path)
        if image_hash != record.get("image_sha256"):
            raise RuntimeError(f"Fixed image content changed: {image_path}")
        K = exact_pixel_intrinsics(record["camera"])
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        expected_shape = (
            int(record["camera"]["height"]),
            int(record["camera"]["width"]),
        )
        if image.shape[:2] != expected_shape:
            raise RuntimeError(
                f"Image/camera raster mismatch for {image_path.name}: "
                f"{image.shape[:2]} != {expected_shape}"
            )
        stem = Path(str(record["image_name"])).stem
        destination = views_root / f"{stem}.npz"
        if destination.is_file() and not args.replace_views:
            cached = load_view(destination)
            metadata = cached["metadata"]
            if (
                metadata.get("image_sha256") == image_hash
                and metadata.get("model_id") == str(args.model)
                and metadata.get("model_revision") == resolved_revision
                and metadata.get("model_checkpoint_sha256")
                == checkpoint_sha256
                and metadata.get("audited_moge_commit")
                == MOGE3_AUDITED_COMMIT
                and metadata.get("refine_steps") == int(args.refine_steps)
                and metadata.get("resolution_level")
                == int(args.resolution_level)
                and metadata.get("use_fp16") == bool(args.use_fp16)
                and metadata.get("camera_record_sha256")
                == canonical_json_sha256(record)
            ):
                reused += 1
                print(
                    json.dumps(
                        {"view": position, "image": image_path.name, "status": "reused"}
                    ),
                    flush=True,
                )
                continue
            raise RuntimeError(
                f"Existing MoGe3 view has different provenance: {destination}; "
                "use --replace-views only after reviewing the change"
            )
        fov_x = horizontal_fov_degrees(
            width=image.shape[1], fx=float(K[0, 0])
        )
        prediction = _infer_one(
            model=model,
            torch=torch,
            image=image,
            fov_x=fov_x,
            refine_steps=args.refine_steps,
            resolution_level=args.resolution_level,
            use_fp16=args.use_fp16,
        )
        payload = make_view_payload(
            depth_m=prediction["depth"],
            normal_direct_camera=prediction["normal"],
            model_valid_mask=prediction["mask"],
            depth_per_step_m=prediction["depth_per_step"],
            K_px=K,
            model_intrinsics_normalized=prediction["intrinsics"],
            camera_record=record,
            image_sha256=image_hash,
            model_id=str(args.model),
            model_revision=resolved_revision,
            model_checkpoint_sha256=checkpoint_sha256,
            refine_steps=args.refine_steps,
            resolution_level=args.resolution_level,
            use_fp16=args.use_fp16,
        )
        atomic_save_view(destination, payload)
        generated += 1
        print(
            json.dumps(
                {"view": position, "image": image_path.name, "status": "generated"}
            ),
            flush=True,
        )

    index_path = output / "moge3_index.json"
    if len(selected) == len(all_records):
        generator = {
            "schema_version": "g4splat-moge3-generator-v1",
            "script": str(Path(__file__).resolve()),
            "python": platform.python_version(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "refine_steps": int(args.refine_steps),
            "resolution_level": int(args.resolution_level),
            "use_fp16": bool(args.use_fp16),
            "force_projection": True,
            "apply_mask": False,
            "input_fov_policy": "exact_cambridge_fx_and_width",
            **environment,
        }
        index = build_index(
            view_root=views_root,
            scene_contract=scene_contract,
            output=index_path,
            model_id=str(args.model),
            model_revision=resolved_revision,
            generator=generator,
        )
        print(json.dumps(index, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {
                    "status": "partial",
                    "generated": generated,
                    "reused": reused,
                    "index_written": False,
                    "next_step": "run all remaining ranges, then rerun without --limit",
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
