#!/usr/bin/env python3
"""Hard-gate Chart geometry against the depth target used for alignment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path)
    parser.add_argument("--mask-dataset-path", type=Path)
    parser.add_argument("--mask-indices", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--max-relative-p90", type=float, default=0.50)
    parser.add_argument("--max-gt25-fraction", type=float, default=0.50)
    parser.add_argument("--min-valid-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-abs-depth",
        type=float,
        default=None,
        help="Optional absolute legacy depth cap; inferred from the run policy when omitted.",
    )
    parser.add_argument(
        "--max-abs-point",
        type=float,
        default=None,
        help="Optional absolute legacy point-norm cap; inferred from the run policy when omitted.",
    )
    parser.add_argument(
        "--require-reference-depths",
        action="store_true",
        help=(
            "Refuse legacy prior-only Chart archives.  New strict runs persist "
            "the MASt3R target depth used by alignment and must be gated against it."
        ),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    return cv2.resize(
        mask.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def resolve_absolute_caps(
    mast3r_scene: Path,
    *,
    max_abs_depth: float | None,
    max_abs_point: float | None,
) -> tuple[float | None, float | None, str]:
    """Keep legacy safety caps, but never impose them on outdoor inverse depth.

    The gate runs as a separate process spawned by both old and new runners.
    Reading the immutable mainline manifest avoids relying on an easy-to-forget
    flag in that hand-off while preserving the original 50-unit behavior for
    legacy experiments.
    """
    manifest_path = Path(mast3r_scene).parent / "outdoor_mainline_manifest.json"
    outdoor = False
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            outdoor = (
                manifest.get("policy_version") == "cambridge-outdoor-structural-mainline-v1"
                and manifest.get("depth_policy") == "inverse_depth_fusion_v1"
            )
        except (OSError, json.JSONDecodeError):
            outdoor = False
    if outdoor:
        return max_abs_depth, max_abs_point, "outdoor_manifest_no_implicit_cap"
    return (
        50.0 if max_abs_depth is None else max_abs_depth,
        50.0 if max_abs_point is None else max_abs_point,
        "legacy_default_absolute_cap",
    )


def gate_charts(
    payload: dict[str, np.ndarray],
    names: list[str],
    semantic_masks: list[np.ndarray] | None,
    *,
    max_relative_p90: float,
    max_gt25_fraction: float,
    min_valid_fraction: float,
    max_abs_depth: float | None = None,
    max_abs_point: float | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    depths = np.asarray(payload["depths"])
    priors = np.asarray(payload["prior_depths"])
    reference_depths = np.asarray(payload.get("reference_depths", priors))
    reference_source = (
        "mast3r_reference_depths"
        if "reference_depths" in payload
        else "depthanything_prior_legacy"
    )
    reference_masks = payload.get("alignment_reference_mask")
    if reference_masks is not None:
        reference_masks = np.asarray(reference_masks)
    confs = np.asarray(payload["confs"]).copy()
    points = payload.get("pts", payload.get("pts3d"))
    if not (
        len(names)
        == len(depths)
        == len(priors)
        == len(reference_depths)
        == len(confs)
    ):
        raise RuntimeError("Camera and charts_data counts do not match")
    if reference_depths.shape != depths.shape:
        raise RuntimeError(
            "reference_depths must match depths, got "
            f"{reference_depths.shape} and {depths.shape}"
        )
    if reference_masks is not None and reference_masks.shape != depths.shape:
        raise RuntimeError(
            "alignment_reference_mask must match depths, got "
            f"{reference_masks.shape} and {depths.shape}"
        )

    records: list[dict[str, object]] = []
    absolute_outlier_masks: list[np.ndarray] = []
    for index, name in enumerate(names):
        depth = np.squeeze(depths[index])
        prior = np.squeeze(priors[index])
        reference = np.squeeze(reference_depths[index])
        confidence = np.squeeze(confs[index])
        valid = (
            np.isfinite(depth)
            & np.isfinite(reference)
            & (depth > 0.0)
            & (reference > 0.0)
            & (confidence > 0.5)
        )
        absolute_outlier = np.zeros(depth.shape, dtype=bool)
        if max_abs_depth is not None and max_abs_depth > 0:
            absolute_outlier |= np.abs(depth) > max_abs_depth
            absolute_outlier |= np.abs(reference) > max_abs_depth
            absolute_outlier |= np.abs(prior) > max_abs_depth
        if points is not None and max_abs_point is not None and max_abs_point > 0:
            point = np.asarray(points[index])
            absolute_outlier |= (~np.isfinite(point).all(axis=-1)) | (
                np.linalg.norm(point, axis=-1) > max_abs_point
            )
        # A prior-agreement test cannot catch a depth and its prior failing
        # together. Remove physically implausible pixels before measuring chart
        # support so they cannot seed remote floaters downstream.
        valid &= ~absolute_outlier
        confs[index][absolute_outlier] = -2.0
        absolute_outlier_masks.append(absolute_outlier)
        eligible = np.ones(depth.shape, dtype=bool)
        if semantic_masks is not None:
            eligible = _resize_mask(semantic_masks[index], depth.shape)
        if reference_masks is not None:
            eligible &= _resize_mask(reference_masks[index], depth.shape)
        valid &= eligible
        valid_fraction = float(valid.mean())
        eligible_pixels = int(eligible.sum())
        support_ratio = float(valid.sum() / max(eligible_pixels, 1))
        relative = np.abs(depth - reference) / np.maximum(np.abs(reference), 1e-6)
        prior_relative = np.abs(depth - prior) / np.maximum(np.abs(prior), 1e-6)
        if np.any(valid):
            relative_p90 = float(np.quantile(relative[valid], 0.9))
            gt25_fraction = float(np.mean(relative[valid] > 0.25))
            prior_valid = valid & np.isfinite(prior) & (prior > 0.0)
            prior_relative_p90 = (
                float(np.quantile(prior_relative[prior_valid], 0.9))
                if np.any(prior_valid)
                else None
            )
            prior_gt25_fraction = (
                float(np.mean(prior_relative[prior_valid] > 0.25))
                if np.any(prior_valid)
                else None
            )
        else:
            relative_p90 = None
            gt25_fraction = None
            prior_relative_p90 = None
            prior_gt25_fraction = None
        insufficient_support = support_ratio < min_valid_fraction
        depth_conflict = (
            (relative_p90 is not None and relative_p90 > max_relative_p90)
            or (gt25_fraction is not None and gt25_fraction > max_gt25_fraction)
        )
        # A chart with too little surviving support is not a harmless no-op: it
        # creates a hole in the selected camera coverage and must be replaced.
        # The old gate only rejected depth conflicts *after* enough pixels had
        # survived, so almost-empty charts silently remained in the chart set.
        reject = insufficient_support or depth_conflict
        if reject:
            confs[index] = -2.0
        records.append(
            {
                "index": index,
                "image_name": name,
                "valid_fraction": valid_fraction,
                "eligible_fraction": float(eligible.mean()),
                "support_ratio": support_ratio,
                "absolute_outlier_fraction": float(absolute_outlier.mean()),
                "reference_source": reference_source,
                "relative_p90": relative_p90,
                "gt25_fraction": gt25_fraction,
                "prior_relative_p90": prior_relative_p90,
                "prior_gt25_fraction": prior_gt25_fraction,
                "insufficient_support": bool(insufficient_support),
                "depth_conflict": bool(depth_conflict),
                "rejection_reasons": [
                    reason
                    for reason, active in (
                        ("insufficient_support", insufficient_support),
                        ("depth_conflict", depth_conflict),
                    )
                    if active
                ],
                "rejected": bool(reject),
            }
        )

    gated = dict(payload)
    rejected_mask = np.asarray(
        [bool(record["rejected"]) for record in records], dtype=bool
    )
    # Make rejection a hard geometry operation, not merely advisory metadata.
    # Several downstream stages derive normals/planes from depth before looking
    # at confidence, so leaving rejected depth/points populated can still leak a
    # failed chart into the frontend.  Zero every chart-indexed geometry tensor
    # while preserving unrelated scalar/metadata arrays.
    geometry_keys = ("depths", "prior_depths", "pts", "pts3d", "confs")
    for key in geometry_keys:
        if key not in payload:
            continue
        value = np.asarray(payload[key]).copy()
        if value.ndim > 0 and value.shape[0] == len(names):
            value[rejected_mask] = 0
            if key != "confs":
                for index, outlier_mask in enumerate(absolute_outlier_masks):
                    if not rejected_mask[index]:
                        value[index][outlier_mask] = 0
        gated[key] = value
    gated["confs"] = confs
    gated["alignment_gate_valid"] = np.asarray(
        [not record["rejected"] for record in records], dtype=bool
    )
    return gated, records


def main() -> None:
    args = parse_args()
    max_abs_depth, max_abs_point, cap_policy = resolve_absolute_caps(
        args.mast3r_scene,
        max_abs_depth=args.max_abs_depth,
        max_abs_point=args.max_abs_point,
    )
    charts_path = args.mast3r_scene / "charts_data.npz"
    backup_path = args.mast3r_scene / "charts_data.pre_conflict_gate.npz"
    source_path = backup_path if backup_path.exists() else charts_path
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not backup_path.exists():
        shutil.copy2(charts_path, backup_path)

    cameras = json.loads((args.mast3r_scene / "cameras.json").read_text())
    names = [Path(path).name for path in cameras["filepaths"]]
    semantic_masks = None
    if args.mask_pickle is not None:
        if args.mask_dataset_path is None:
            raise ValueError("--mask-dataset-path is required with --mask-pickle")
        lookup = CambridgeMaskLookup(
            args.mask_dataset_path,
            args.mask_pickle,
            mask_indices=args.mask_indices,
        )
        semantic_masks = []
        for name in names:
            source_name = lookup.source_name_for(name)
            source_shape = tuple(lookup.masks[source_name][0].shape[-2:])
            semantic_masks.append(
                lookup.get_mask(name, source_shape, "cpu").cpu().numpy().astype(bool)
            )

    with np.load(source_path) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    if args.require_reference_depths and "reference_depths" not in payload:
        raise RuntimeError(
            "Strict Chart gating requires persisted MASt3R reference_depths; "
            f"{source_path} is a legacy prior-only archive. Re-run alignment."
        )
    gated, records = gate_charts(
        payload,
        names,
        semantic_masks,
        max_relative_p90=args.max_relative_p90,
        max_gt25_fraction=args.max_gt25_fraction,
        min_valid_fraction=args.min_valid_fraction,
        max_abs_depth=max_abs_depth,
        max_abs_point=max_abs_point,
    )
    temporary = charts_path.with_suffix(".gated.tmp.npz")
    np.savez_compressed(temporary, **gated)
    temporary.replace(charts_path)

    report_path = args.report or args.mast3r_scene / "aligned_chart_conflict_gate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(source_path),
        "output": str(charts_path),
        "thresholds": {
            "reference_source": (
                "mast3r_reference_depths"
                if "reference_depths" in payload
                else "depthanything_prior_legacy"
            ),
            "max_relative_p90": args.max_relative_p90,
            "max_gt25_fraction": args.max_gt25_fraction,
            "min_valid_fraction": args.min_valid_fraction,
            "max_abs_depth": max_abs_depth,
            "max_abs_point": max_abs_point,
            "absolute_cap_policy": cap_policy,
            "require_reference_depths": bool(args.require_reference_depths),
        },
        "rejected_count": sum(record["rejected"] for record in records),
        "records": records,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "rejected_count": report["rejected_count"],
        "rejected": [record["image_name"] for record in records if record["rejected"]],
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
