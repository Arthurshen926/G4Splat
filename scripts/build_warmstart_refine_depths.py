#!/usr/bin/env python3
"""Build conservative depth priors for MAtCha warm-start refinement.

This intentionally bypasses G4Splat's global plane rewrite. Real chart views keep
only rendered depth with multi-view support, while pseudo views use the robustly
aligned See3D depth produced by ``see3d_dn_util.py``. Pseudo-view losses can apply
their inpaint masks later during training.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_PATTERN = re.compile(r"^depth_frame(\d{6})\.tiff$")


@dataclass(frozen=True)
class FrameDiagnostics:
    frame_id: int
    frame_type: str
    depth_valid_fraction: float
    support_fraction: float
    semantic_keep_fraction: float | None
    final_confident_fraction: float


def _read_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    return np.asarray(Image.open(path))


def _load_pseudo_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"Expected a JSON list of integer frame IDs: {path}")
    return set(values)


def _frame_ids(plane_root: Path) -> list[int]:
    frame_ids = []
    for path in plane_root.iterdir():
        match = FRAME_PATTERN.match(path.name)
        if match:
            frame_ids.append(int(match.group(1)))
    frame_ids.sort()
    if not frame_ids:
        raise FileNotFoundError(f"No depth_frameXXXXXX.tiff files in {plane_root}")
    expected = list(range(frame_ids[-1] + 1))
    if frame_ids != expected:
        missing = sorted(set(expected) - set(frame_ids))
        raise ValueError(f"Depth frame IDs must be contiguous from zero; missing {missing}")
    return frame_ids


def _require_same_shape(name: str, array: np.ndarray, depth: np.ndarray) -> np.ndarray:
    array = np.squeeze(array)
    if array.shape != depth.shape:
        raise ValueError(
            f"{name} shape {array.shape} does not match depth shape {depth.shape}"
        )
    return array


def build_warmstart_refine_depths(
    plane_root: Path,
    pseudo_ids_path: Path | None = None,
    support_threshold: float = 0.5,
    max_abs_depth: float = 1_000.0,
    require_real_semantic_keep: bool = True,
) -> list[FrameDiagnostics]:
    plane_root = plane_root.expanduser().resolve()
    if not plane_root.is_dir():
        raise NotADirectoryError(plane_root)
    pseudo_ids = _load_pseudo_ids(pseudo_ids_path)
    frame_ids = _frame_ids(plane_root)
    unknown_pseudo_ids = sorted(pseudo_ids - set(frame_ids))
    if unknown_pseudo_ids:
        raise ValueError(f"Pseudo frame IDs have no depth files: {unknown_pseudo_ids}")

    diagnostics: list[FrameDiagnostics] = []
    for frame_id in frame_ids:
        depth_path = plane_root / f"depth_frame{frame_id:06d}.tiff"
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth image at {depth_path}, got {depth.shape}")

        depth_valid = np.isfinite(depth) & (depth > 0) & (np.abs(depth) <= max_abs_depth)
        clean_depth = np.where(depth_valid, depth, 0).astype(np.float32, copy=False)
        frame_type = "pseudo" if frame_id in pseudo_ids else "real"
        semantic_fraction: float | None = None

        if frame_type == "real":
            visibility_path = plane_root / f"visibility_frame{frame_id:06d}.npy"
            if not visibility_path.is_file():
                raise FileNotFoundError(visibility_path)
            visibility = _require_same_shape(
                "visibility", _read_array(visibility_path), clean_depth
            )
            support = np.isfinite(visibility) & (visibility > support_threshold)

            semantic_path = plane_root / f"semantic_keep_frame{frame_id:06d}.npy"
            if semantic_path.is_file():
                semantic_keep = _require_same_shape(
                    "semantic keep", _read_array(semantic_path), clean_depth
                ) > 0
                semantic_fraction = float(semantic_keep.mean())
            elif require_real_semantic_keep:
                raise FileNotFoundError(semantic_path)
            else:
                semantic_keep = np.ones_like(depth_valid)
            confident = depth_valid & support & semantic_keep
        else:
            # See3D depth has already been support-aligned. The trainer applies the
            # synthesized-hole mask, so keep all finite aligned candidates here.
            support = depth_valid
            confident = depth_valid

        Image.fromarray(clean_depth, mode="F").save(
            plane_root / f"refine_depth_frame{frame_id:06d}.tiff"
        )
        Image.fromarray((confident.astype(np.uint8) * 255), mode="L").save(
            plane_root / f"confident_map_frame{frame_id:06d}.png"
        )
        diagnostics.append(
            FrameDiagnostics(
                frame_id=frame_id,
                frame_type=frame_type,
                depth_valid_fraction=float(depth_valid.mean()),
                support_fraction=float(support.mean()),
                semantic_keep_fraction=semantic_fraction,
                final_confident_fraction=float(confident.mean()),
            )
        )

    summary = {
        "mode": "conservative_warmstart_no_plane_rewrite",
        "support_threshold": support_threshold,
        "max_abs_depth": max_abs_depth,
        "pseudo_frame_ids": sorted(pseudo_ids),
        "frames": [asdict(item) for item in diagnostics],
    }
    with (plane_root / "warmstart_depth_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane_root", type=Path, required=True)
    parser.add_argument("--pseudo_ids", type=Path)
    parser.add_argument("--support_threshold", type=float, default=0.5)
    parser.add_argument("--max_abs_depth", type=float, default=1_000.0)
    parser.add_argument(
        "--allow_missing_semantic_keep",
        action="store_true",
        help="Allow real frames without semantic_keep_frame*.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = build_warmstart_refine_depths(
        plane_root=args.plane_root,
        pseudo_ids_path=args.pseudo_ids,
        support_threshold=args.support_threshold,
        max_abs_depth=args.max_abs_depth,
        require_real_semantic_keep=not args.allow_missing_semantic_keep,
    )
    real = [item for item in diagnostics if item.frame_type == "real"]
    pseudo = [item for item in diagnostics if item.frame_type == "pseudo"]
    real_mean = np.mean([item.final_confident_fraction for item in real]) if real else 0.0
    pseudo_mean = (
        np.mean([item.final_confident_fraction for item in pseudo]) if pseudo else 0.0
    )
    print(
        f"Built conservative priors for {len(real)} real and {len(pseudo)} pseudo views; "
        f"mean confidence real={real_mean:.3f}, pseudo={pseudo_mean:.3f}."
    )


if __name__ == "__main__":
    main()
