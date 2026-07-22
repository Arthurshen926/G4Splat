#!/usr/bin/env python3
"""Audit aligned Chart depth against the MASt3R pointmap observation target.

Legacy Chart archives were gated against their DepthAnything initialization.
That detects uncontrolled changes, but cannot say whether a large change moved
the Chart *towards* or *away from* the MASt3R observation which the alignment
loss actually optimizes.  Strict archives persist that target and the gate now
uses it; this tool independently reconstructs it from the pointmaps to audit
that serialization contract.

It recreates the dense, per-image MASt3R target depth directly from the saved
pointmap and calibrated camera, applies the same confidence/semantic support
used by ``align_charts.py``, and reports prior-to-target versus
aligned-to-target error for every Chart.  It is read-only with respect to the
MASt3R scene and Chart archive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def pointmap_target_depth(
    points: np.ndarray,
    camera_to_world: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Return the calibrated view-space depth of a MASt3R pointmap."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 pointmap, got {points.shape}")
    if camera_to_world.shape != (4, 4):
        raise ValueError(f"Expected 4x4 camera-to-world matrix, got {camera_to_world.shape}")
    world_to_camera = np.linalg.inv(camera_to_world)
    flat = points.reshape(-1, 3)
    depth = flat @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    return (depth[:, 2] * float(scale_factor)).reshape(points.shape[:2])


def relative_error_metrics(
    estimate: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int | None]:
    """Robust relative-depth metrics on a caller-defined common support."""
    if estimate.shape != target.shape or estimate.shape != valid.shape:
        raise ValueError(
            "estimate, target, and valid must share a shape; got "
            f"{estimate.shape}, {target.shape}, {valid.shape}"
        )
    valid = valid & np.isfinite(estimate) & np.isfinite(target)
    valid &= (estimate > 0.0) & (target > 0.0)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return {
            "valid_pixels": 0,
            "relative_median": None,
            "relative_p90": None,
            "relative_mean": None,
            "gt25_fraction": None,
        }
    relative = np.abs(estimate[valid] - target[valid]) / np.maximum(
        np.abs(target[valid]), 1e-6
    )
    return {
        "valid_pixels": count,
        "relative_median": float(np.quantile(relative, 0.50)),
        "relative_p90": float(np.quantile(relative, 0.90)),
        "relative_mean": float(np.mean(relative)),
        "gt25_fraction": float(np.mean(relative > 0.25)),
    }


def _scalar(payload: dict[str, np.ndarray], key: str, default: float) -> float:
    if key not in payload:
        return float(default)
    value = np.asarray(payload[key]).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError(f"{key} must contain one finite scalar, got {payload[key]!r}")
    return float(value[0])


def _read_chart_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        payload = {key: archive[key] for key in archive.files}
    required = {"prior_depths", "depths"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"{path} is missing required Chart tensors: {sorted(missing)}")
    return payload


def _gate_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    records = {}
    for record in payload.get("records", []):
        name = str(record["image_name"])
        if name in records:
            raise RuntimeError(f"Duplicate gate record for {name} in {path}")
        records[name] = record
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument(
        "--charts-data",
        type=Path,
        help=(
            "Chart NPZ to audit. Defaults to charts_data.pre_conflict_gate.npz "
            "when present, otherwise charts_data.npz under --mast3r-scene."
        ),
    )
    parser.add_argument("--mask-pickle", type=Path)
    parser.add_argument("--mask-dataset-path", type=Path)
    parser.add_argument("--mask-indices", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--sfm-confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--gate-report",
        type=Path,
        help="Optional aligned-chart gate report to join with target-agreement outcomes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mask_pickle is None and args.mask_dataset_path is not None:
        raise ValueError("--mask-dataset-path requires --mask-pickle")
    if args.mask_pickle is not None and args.mask_dataset_path is None:
        raise ValueError("--mask-pickle requires --mask-dataset-path")

    scene = args.mast3r_scene.resolve()
    default_pre_gate = scene / "charts_data.pre_conflict_gate.npz"
    charts_path = args.charts_data.resolve() if args.charts_data else (
        default_pre_gate if default_pre_gate.exists() else scene / "charts_data.npz"
    )
    charts = _read_chart_archive(charts_path)
    priors = np.asarray(charts["prior_depths"])
    aligned = np.asarray(charts["depths"])
    if priors.ndim != 3 or aligned.shape != priors.shape:
        raise RuntimeError(
            "Expected prior_depths and depths with common (charts,H,W) shape, got "
            f"{priors.shape} and {aligned.shape}"
        )
    scale_factor = _scalar(charts, "scale_factor", 1.0)
    persisted_reference = charts.get("reference_depths")
    if persisted_reference is not None:
        persisted_reference = np.asarray(persisted_reference)
        if persisted_reference.shape != priors.shape:
            raise RuntimeError(
                "Persisted reference_depths must match chart depth shape, got "
                f"{persisted_reference.shape} and {priors.shape}"
            )
    persisted_reference_mask = charts.get("alignment_reference_mask")
    if persisted_reference_mask is not None:
        persisted_reference_mask = np.asarray(persisted_reference_mask, dtype=bool)
        if persisted_reference_mask.shape != priors.shape:
            raise RuntimeError(
                "Persisted alignment_reference_mask must match chart depth shape, got "
                f"{persisted_reference_mask.shape} and {priors.shape}"
            )

    cameras = json.loads((scene / "cameras.json").read_text())
    paths = cameras.get("filepaths", [])
    poses = cameras.get("cams2world", [])
    if len(paths) != len(poses) or len(paths) != priors.shape[0]:
        raise RuntimeError(
            "MASt3R cameras and Chart tensors disagree: "
            f"filepaths={len(paths)}, poses={len(poses)}, charts={priors.shape[0]}"
        )
    names = [Path(path).name for path in paths]

    lookup = None
    if args.mask_pickle is not None:
        lookup = CambridgeMaskLookup(
            args.mask_dataset_path.resolve(),
            args.mask_pickle.resolve(),
            mask_indices=args.mask_indices,
        )
    gate_records = _gate_records(args.gate_report.resolve() if args.gate_report else None)

    reports: list[dict[str, Any]] = []
    for index, (name, c2w_payload) in enumerate(zip(names, poses)):
        pointmap_path = scene / "pointmaps" / f"{Path(name).stem}.json"
        if not pointmap_path.exists():
            raise FileNotFoundError(pointmap_path)
        pointmap = json.loads(pointmap_path.read_text())
        points = np.asarray(pointmap["points"], dtype=np.float32).reshape(
            *priors.shape[1:], 3
        )
        confidence = np.asarray(pointmap["confs"], dtype=np.float32).reshape(
            priors.shape[1:]
        )
        target = pointmap_target_depth(
            points,
            np.asarray(c2w_payload, dtype=np.float64),
            scale_factor,
        )
        valid = np.isfinite(target) & (target > 0.0)
        valid &= np.isfinite(confidence) & (confidence > args.sfm_confidence_threshold)
        if lookup is not None:
            semantic = lookup.get_mask(name, target.shape, "cpu").cpu().numpy()
            valid &= semantic
        prior_metrics = relative_error_metrics(priors[index], target, valid)
        aligned_metrics = relative_error_metrics(aligned[index], target, valid)
        persisted_reference_metrics = (
            None
            if persisted_reference is None
            else relative_error_metrics(persisted_reference[index], target, valid)
        )
        if prior_metrics["relative_p90"] is None or aligned_metrics["relative_p90"] is None:
            p90_delta = None
            mean_delta = None
        else:
            p90_delta = float(
                aligned_metrics["relative_p90"] - prior_metrics["relative_p90"]
            )
            mean_delta = float(
                aligned_metrics["relative_mean"] - prior_metrics["relative_mean"]
            )
        gate = gate_records.get(name)
        reports.append(
            {
                "index": index,
                "image_name": name,
                "effective_support_fraction": float(np.mean(valid)),
                "prior_to_mast3r": prior_metrics,
                "aligned_to_mast3r": aligned_metrics,
                "persisted_reference_to_mast3r": persisted_reference_metrics,
                "persisted_reference_mask_fraction": (
                    None
                    if persisted_reference_mask is None
                    else float(np.mean(persisted_reference_mask[index]))
                ),
                "aligned_minus_prior_target_relative_p90": p90_delta,
                "aligned_minus_prior_target_relative_mean": mean_delta,
                "target_p90_improved": None if p90_delta is None else bool(p90_delta < 0.0),
                "alignment_internal_valid": (
                    None
                    if "alignment_chart_valid" not in charts
                    else bool(np.asarray(charts["alignment_chart_valid"])[index])
                ),
                "hard_gate_rejected": None if gate is None else bool(gate.get("rejected")),
                "hard_gate_reasons": None if gate is None else gate.get("rejection_reasons", []),
            }
        )

    finite = [
        record
        for record in reports
        if record["aligned_minus_prior_target_relative_p90"] is not None
    ]
    p90_prior = np.asarray(
        [record["prior_to_mast3r"]["relative_p90"] for record in finite], dtype=np.float64
    )
    p90_aligned = np.asarray(
        [record["aligned_to_mast3r"]["relative_p90"] for record in finite], dtype=np.float64
    )
    deltas = p90_aligned - p90_prior
    persisted_reference_p90 = np.asarray(
        [
            record["persisted_reference_to_mast3r"]["relative_p90"]
            for record in finite
            if record["persisted_reference_to_mast3r"] is not None
            and record["persisted_reference_to_mast3r"]["relative_p90"] is not None
        ],
        dtype=np.float64,
    )
    rejected = [record for record in finite if record["hard_gate_rejected"]]
    accepted = [record for record in finite if record["hard_gate_rejected"] is False]
    result = {
        "method": "calibrated_mast3r_pointmap_target_depth",
        "mast3r_scene": str(scene),
        "charts_data": str(charts_path),
        "gate_report": None if args.gate_report is None else str(args.gate_report.resolve()),
        "scale_factor": scale_factor,
        "support": {
            "semantic_mask_indices": None if lookup is None else list(args.mask_indices),
            "sfm_confidence_threshold": float(args.sfm_confidence_threshold),
        },
        "summary": {
            "chart_count": len(reports),
            "finite_target_comparisons": len(finite),
            "target_p90_prior_median": float(np.median(p90_prior)),
            "target_p90_aligned_median": float(np.median(p90_aligned)),
            "target_p90_delta_median": float(np.median(deltas)),
            "target_p90_delta_mean": float(np.mean(deltas)),
            "target_p90_improved_count": int(np.count_nonzero(deltas < 0.0)),
            "target_p90_worsened_count": int(np.count_nonzero(deltas > 0.0)),
            "persisted_reference_available": persisted_reference is not None,
            "persisted_reference_target_p90_median": (
                None
                if persisted_reference_p90.size == 0
                else float(np.median(persisted_reference_p90))
            ),
            "persisted_reference_target_p90_max": (
                None
                if persisted_reference_p90.size == 0
                else float(np.max(persisted_reference_p90))
            ),
            "hard_gate_rejected_count": len(rejected),
            "hard_gate_rejected_target_improved_count": sum(
                bool(record["target_p90_improved"]) for record in rejected
            ),
            "hard_gate_accepted_target_worsened_count": sum(
                not bool(record["target_p90_improved"]) for record in accepted
            ),
        },
        "records": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
