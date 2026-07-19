#!/usr/bin/env python3
"""Reject unsafe See3D pseudo-views before geometry aggregation or training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class See3DViewQuality:
    index: int
    accepted: bool
    reasons: list[str]
    synthesized_fraction: float
    visible_sharpness: float
    generated_hole_sharpness: float
    depth_alignment_accepted: bool
    depth_alignment_relative_rmse: float | None
    depth_alignment_support_pixels: int


def _laplacian_variance(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    values = laplacian[mask]
    return float(np.var(values)) if values.size else 0.0


def _load_alignment_records(stage_root: Path) -> dict[int, dict]:
    path = stage_root / "select-gs-planes" / "depth_alignment_diagnostics.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Invalid depth alignment diagnostics: {path}")
    return {int(record["frame"]): record for record in records}


def filter_see3d_views(
    stage_root: Path,
    max_synthesized_fraction: float = 0.35,
    min_visible_sharpness: float = 100.0,
    max_alignment_rmse: float = 0.12,
    min_alignment_support_pixels: int = 20_000,
) -> tuple[list[int], list[See3DViewQuality]]:
    stage_root = stage_root.expanduser().resolve()
    camera_path = next(stage_root.glob("stage*_see3d_cameras.npz"), None)
    if camera_path is None:
        raise FileNotFoundError(f"No stage*_see3d_cameras.npz in {stage_root}")
    with np.load(camera_path) as cameras:
        view_count = int(cameras["n_views"])
    alignments = _load_alignment_records(stage_root)

    records = []
    accepted_indices = []
    for index in range(view_count):
        visible_path = stage_root / "select-gs" / f"mask_frame{index:06d}.png"
        raw_path = stage_root / "select-gs" / f"ori_warp_frame{index:06d}.png"
        generated_path = (
            stage_root
            / "select-gs-inpainted-merged"
            / f"predict_warp_frame{index:06d}.png"
        )
        for path in (visible_path, raw_path, generated_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        visible = np.asarray(Image.open(visible_path).convert("L")) > 127
        synthesized = ~visible
        raw = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
        generated = np.asarray(Image.open(generated_path).convert("RGB"), dtype=np.uint8)
        synthesized_fraction = float(synthesized.mean())
        visible_sharpness = _laplacian_variance(raw, visible)
        generated_hole_sharpness = _laplacian_variance(generated, synthesized)

        alignment = alignments.get(index, {})
        alignment_accepted = bool(alignment.get("accepted", False))
        relative_rmse_value = alignment.get("relative_rmse")
        relative_rmse = (
            float(relative_rmse_value) if relative_rmse_value is not None else None
        )
        support_pixels = int(alignment.get("support_pixels", 0))

        reasons = []
        if synthesized_fraction > max_synthesized_fraction:
            reasons.append("too_much_synthesis")
        if visible_sharpness < min_visible_sharpness:
            reasons.append("blurry_or_smeared_baseline_render")
        if not alignment_accepted:
            reasons.append("depth_alignment_rejected")
        if relative_rmse is None or relative_rmse > max_alignment_rmse:
            reasons.append("depth_alignment_rmse")
        if support_pixels < min_alignment_support_pixels:
            reasons.append("insufficient_depth_support")

        accepted = not reasons
        if accepted:
            accepted_indices.append(index)
        records.append(
            See3DViewQuality(
                index=index,
                accepted=accepted,
                reasons=reasons,
                synthesized_fraction=synthesized_fraction,
                visible_sharpness=visible_sharpness,
                generated_hole_sharpness=generated_hole_sharpness,
                depth_alignment_accepted=alignment_accepted,
                depth_alignment_relative_rmse=relative_rmse,
                depth_alignment_support_pixels=support_pixels,
            )
        )

    report = {
        "thresholds": {
            "max_synthesized_fraction": max_synthesized_fraction,
            "min_visible_sharpness": min_visible_sharpness,
            "max_alignment_rmse": max_alignment_rmse,
            "min_alignment_support_pixels": min_alignment_support_pixels,
        },
        "accepted_indices": accepted_indices,
        "rejected_indices": [record.index for record in records if not record.accepted],
        "views": [asdict(record) for record in records],
    }
    with (stage_root / "see3d_quality_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with (stage_root / "accepted_view_indices.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(accepted_indices, handle)
        handle.write("\n")
    return accepted_indices, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage_root", type=Path, required=True)
    parser.add_argument("--max_synthesized_fraction", type=float, default=0.35)
    parser.add_argument("--min_visible_sharpness", type=float, default=100.0)
    parser.add_argument("--max_alignment_rmse", type=float, default=0.12)
    parser.add_argument("--min_alignment_support_pixels", type=int, default=20_000)
    args = parser.parse_args()
    accepted, records = filter_see3d_views(
        stage_root=args.stage_root,
        max_synthesized_fraction=args.max_synthesized_fraction,
        min_visible_sharpness=args.min_visible_sharpness,
        max_alignment_rmse=args.max_alignment_rmse,
        min_alignment_support_pixels=args.min_alignment_support_pixels,
    )
    print(f"Accepted See3D views: {accepted}")
    for record in records:
        print(
            f"view {record.index}: accepted={record.accepted}, "
            f"synth={record.synthesized_fraction:.3f}, "
            f"sharpness={record.visible_sharpness:.1f}, reasons={record.reasons}"
        )


if __name__ == "__main__":
    main()
