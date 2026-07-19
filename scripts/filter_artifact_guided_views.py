#!/usr/bin/env python3
"""Apply geometry-aware quality policy to artifact-directed See3D views.

Unlike the generic See3D gate, this distinguishes appearance repair on an
already supported surface from an actual geometry replacement.  It can replace
the pseudo depth with the frozen baseline depth for RGB-only static patches;
wrong-geometry views must instead prove that the generated depth is closer to
clean-view triangulation than the baseline depth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_guided_repair import CameraGeometry, project_points  # noqa: E402


STATIC_PATCH_CLASSES = {"supported_static", "supported_static_patch"}


@dataclass(frozen=True)
class ArtifactViewPolicy:
    index: int
    classification: str
    repair_source: str
    accepted: bool
    training_mode: str
    reasons: list[str]
    synthesized_fraction: float
    protected_pixels_exact: bool
    visible_sharpness: float
    generated_hole_sharpness: float
    generated_hole_mean_change: float
    point_count: int
    baseline_depth_median_relative_error: float | None
    generated_depth_median_relative_error: float | None
    depth_alignment_accepted: bool
    depth_alignment_relative_rmse: float | None
    projective_depth_supported_fraction: float | None
    projective_nearest_fill_fraction: float | None


def _read_depth(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32)


def _laplacian_variance(image: np.ndarray, mask: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    values = cv2.Laplacian(gray, cv2.CV_32F)[mask]
    return float(np.var(values)) if values.size else 0.0


def _point_depth_error(
    points: np.ndarray,
    camera: CameraGeometry,
    depth: np.ndarray,
) -> tuple[float | None, int]:
    pixels, expected_depth, inside = project_points(points, camera)
    ids = np.flatnonzero(inside)
    if not len(ids):
        return None, 0
    x = np.clip(np.rint(pixels[ids, 0]).astype(np.int64), 0, camera.width - 1)
    y = np.clip(np.rint(pixels[ids, 1]).astype(np.int64), 0, camera.height - 1)
    sampled = depth[y, x]
    valid = np.isfinite(sampled) & (sampled > 0) & np.isfinite(expected_depth[ids])
    if not np.any(valid):
        return None, 0
    denominator = np.maximum(
        np.maximum(np.abs(sampled[valid]), np.abs(expected_depth[ids][valid])),
        1e-6,
    )
    relative_error = np.abs(sampled[valid] - expected_depth[ids][valid]) / denominator
    return float(np.median(relative_error)), int(np.count_nonzero(valid))


def _alignment_records(stage_root: Path) -> dict[int, dict]:
    path = stage_root / "select-gs-planes" / "depth_alignment_diagnostics.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("frames", value) if isinstance(value, dict) else value
    return {int(record["frame"]): record for record in records}


def _projective_records(stage_root: Path) -> dict[int, dict]:
    path = stage_root / "projective_repair_report.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {int(record["index"]): record for record in value.get("views", [])}


def _apply_projective_policy(stage_root: Path, index: int) -> None:
    stem = f"{index:06d}"
    source_rgb = stage_root / "select-gs-projective-merged" / f"predict_warp_frame{stem}.png"
    destination_rgb = stage_root / "select-gs-inpainted-merged" / f"predict_warp_frame{stem}.png"
    destination_rgb.parent.mkdir(parents=True, exist_ok=True)
    if destination_rgb.is_file():
        backup = destination_rgb.with_suffix(".see3d-backup.png")
        if not backup.exists():
            shutil.copy2(destination_rgb, backup)
    shutil.copy2(source_rgb, destination_rgb)

    source_geometry = stage_root / "select-gs-projective-geometry"
    destination_geometry = stage_root / "select-gs-planes"
    destination_geometry.mkdir(parents=True, exist_ok=True)
    for source in source_geometry.glob(f"*_frame{stem}.*"):
        destination = destination_geometry / source.name
        if destination.exists():
            backup = destination.with_name(destination.name + ".generated-backup")
            if not backup.exists():
                shutil.copy2(destination, backup)
        shutil.copy2(source, destination)


def _save_baseline_depth_policy(
    stage_root: Path,
    index: int,
    baseline_depth_path: Path,
) -> None:
    plane_root = stage_root / "select-gs-planes"
    destination = plane_root / f"depth_frame{index:06d}.tiff"
    aligned_backup = plane_root / f"aligned_depth_frame{index:06d}.tiff"
    if not aligned_backup.exists():
        shutil.copy2(destination, aligned_backup)
    shutil.copy2(baseline_depth_path, destination)

    depth = _read_depth(destination)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    visualization = np.zeros_like(depth, dtype=np.uint8)
    if valid.size:
        lower, upper = np.quantile(valid, [0.01, 0.99])
        normalized = np.clip((depth - lower) / max(float(upper - lower), 1e-6), 0.0, 1.0)
        visualization = np.uint8(normalized * 255.0)
    Image.fromarray(visualization, mode="L").save(
        plane_root / f"depth_frame{index:06d}.png"
    )


def filter_artifact_guided_views(
    stage_root: Path,
    *,
    max_synthesized_fraction: float = 0.10,
    min_visible_sharpness: float = 100.0,
    min_generated_hole_sharpness: float = 50.0,
    min_generated_hole_change: float = 3.0,
    max_static_baseline_depth_error: float = 0.10,
    max_generated_depth_error: float = 0.15,
    min_wrong_geometry_improvement: float = 0.05,
    apply_depth_policy: bool = False,
    repair_source: str = "auto",
    min_projective_depth_supported_fraction: float = 0.85,
    max_projective_nearest_fill_fraction: float = 0.02,
    apply_repair_policy: bool = False,
) -> tuple[list[int], list[ArtifactViewPolicy]]:
    stage_root = stage_root.expanduser().resolve()
    manifest_path = stage_root / "artifact_guided_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alignments = _alignment_records(stage_root)
    projective_records = _projective_records(stage_root)
    policies = []
    accepted_indices = []

    for view in manifest["pseudo_views"]:
        index = int(view["pseudo_view_index"])
        classification = str(view["classification"])
        raw_path = stage_root / "select-gs" / f"ori_warp_frame{index:06d}.png"
        baseline_depth_path = stage_root / "select-gs" / f"depth_frame{index:06d}.tiff"
        known_path = stage_root / "select-gs" / f"mask_frame{index:06d}.png"
        projective = projective_records.get(index)
        use_projective = repair_source == "projective" or (
            repair_source == "auto" and bool(projective and projective.get("accepted"))
        )
        selected_source = "projective" if use_projective else "see3d"
        if use_projective:
            generated_path = (
                stage_root
                / "select-gs-projective-merged"
                / f"predict_warp_frame{index:06d}.png"
            )
            generated_depth_path = (
                stage_root
                / "select-gs-projective-geometry"
                / f"depth_frame{index:06d}.tiff"
            )
        else:
            generated_path = (
                stage_root
                / "select-gs-inpainted-merged"
                / f"predict_warp_frame{index:06d}.png"
            )
            generated_depth_path = (
                stage_root / "select-gs-planes" / f"depth_frame{index:06d}.tiff"
            )
        points_path = Path(view["repair_points_path"])
        for path in (
            raw_path,
            baseline_depth_path,
            known_path,
            generated_path,
            generated_depth_path,
            points_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        raw = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
        generated = np.asarray(Image.open(generated_path).convert("RGB"), dtype=np.uint8)
        known = np.asarray(Image.open(known_path).convert("L")) > 127
        synthesized = ~known
        points = np.load(points_path).astype(np.float64)
        camera = CameraGeometry.from_json(view["pseudo_camera"])
        baseline_depth = _read_depth(baseline_depth_path)
        generated_depth = _read_depth(generated_depth_path)
        baseline_error, baseline_point_count = _point_depth_error(
            points, camera, baseline_depth
        )
        generated_error, generated_point_count = _point_depth_error(
            points, camera, generated_depth
        )
        alignment = alignments.get(index, {})
        alignment_accepted = bool(alignment.get("accepted", False))
        alignment_rmse_value = alignment.get("relative_rmse")
        if use_projective:
            alignment_accepted = bool(projective and projective.get("accepted"))
            alignment_rmse_value = view.get("clean_support_plane", {}).get(
                "normalized_median_residual"
            )
        alignment_rmse = (
            float(alignment_rmse_value) if alignment_rmse_value is not None else None
        )

        synthesized_fraction = float(synthesized.mean())
        protected_exact = bool(np.array_equal(raw[known], generated[known]))
        visible_sharpness = _laplacian_variance(raw, known)
        generated_hole_sharpness = _laplacian_variance(generated, synthesized)
        generated_hole_change = float(
            np.mean(
                np.abs(generated[synthesized].astype(np.float32) - raw[synthesized])
            )
        ) if np.any(synthesized) else 0.0

        reasons = []
        if synthesized_fraction > max_synthesized_fraction:
            reasons.append("too_much_synthesis")
        if not protected_exact:
            reasons.append("protected_pixels_changed")
        # Absolute context sharpness is a property of the frozen baseline.
        # It should only veto a repair when those pixels were modified; an
        # exact protected region cannot have been blurred by this operation.
        if not protected_exact and visible_sharpness < min_visible_sharpness:
            reasons.append("blurry_visible_context")
        if generated_hole_sharpness < min_generated_hole_sharpness:
            reasons.append("blurry_generated_patch")
        if generated_hole_change < min_generated_hole_change:
            reasons.append("generated_patch_unchanged")

        training_mode = "reject"
        if classification in STATIC_PATCH_CLASSES:
            training_mode = "rgb_only_baseline_depth"
            if baseline_error is None or baseline_error > max_static_baseline_depth_error:
                reasons.append("baseline_depth_not_supported")
        elif classification in {"wrong_geometry", "coverage_hole"}:
            training_mode = (
                "rgb_and_clean_support_geometry" if use_projective else "rgb_and_geometry"
            )
            if use_projective:
                depth_supported_fraction = float(
                    (projective or {}).get("fusion", {}).get("depth_supported_fraction", 0.0)
                )
                nearest_fill_fraction = float(
                    (projective or {}).get("nearest_fill_fraction", 1.0)
                )
                if depth_supported_fraction < float(min_projective_depth_supported_fraction):
                    reasons.append("insufficient_projective_depth_support")
                if nearest_fill_fraction > float(max_projective_nearest_fill_fraction):
                    reasons.append("too_much_projective_fill")
            if not alignment_accepted:
                reasons.append("generated_depth_alignment_rejected")
            if generated_error is None or generated_error > max_generated_depth_error:
                reasons.append("generated_depth_not_supported")
            if (
                classification == "wrong_geometry"
                and baseline_error is not None
                and generated_error is not None
                and generated_error
                > baseline_error - min_wrong_geometry_improvement
            ):
                reasons.append("generated_depth_does_not_improve_baseline")
        else:
            reasons.append("unsupported_component_classification")

        accepted = not reasons
        if accepted:
            accepted_indices.append(index)
            if apply_depth_policy and training_mode == "rgb_only_baseline_depth":
                _save_baseline_depth_policy(stage_root, index, baseline_depth_path)
            if apply_repair_policy and use_projective:
                _apply_projective_policy(stage_root, index)
        policies.append(
            ArtifactViewPolicy(
                index=index,
                classification=classification,
                repair_source=selected_source,
                accepted=accepted,
                training_mode=training_mode,
                reasons=reasons,
                synthesized_fraction=synthesized_fraction,
                protected_pixels_exact=protected_exact,
                visible_sharpness=visible_sharpness,
                generated_hole_sharpness=generated_hole_sharpness,
                generated_hole_mean_change=generated_hole_change,
                point_count=min(baseline_point_count, generated_point_count),
                baseline_depth_median_relative_error=baseline_error,
                generated_depth_median_relative_error=generated_error,
                depth_alignment_accepted=alignment_accepted,
                depth_alignment_relative_rmse=alignment_rmse,
                projective_depth_supported_fraction=float(
                    (projective or {}).get("fusion", {}).get("depth_supported_fraction")
                )
                if projective and projective.get("fusion", {}).get("depth_supported_fraction") is not None
                else None,
                projective_nearest_fill_fraction=float(
                    (projective or {}).get("nearest_fill_fraction")
                )
                if projective and projective.get("nearest_fill_fraction") is not None
                else None,
            )
        )

    report = {
        "mode": "artifact_directed_geometry_aware_quality_gate",
        "depth_policy_applied": bool(apply_depth_policy),
        "repair_policy_applied": bool(apply_repair_policy),
        "requested_repair_source": repair_source,
        "accepted_indices": accepted_indices,
        "views": [asdict(policy) for policy in policies],
    }
    (stage_root / "artifact_training_policy.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (stage_root / "accepted_view_indices.json").write_text(
        json.dumps(accepted_indices) + "\n", encoding="utf-8"
    )
    return accepted_indices, policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage_root", type=Path, required=True)
    parser.add_argument("--max_synthesized_fraction", type=float, default=0.10)
    parser.add_argument("--min_visible_sharpness", type=float, default=100.0)
    parser.add_argument("--min_generated_hole_sharpness", type=float, default=50.0)
    parser.add_argument("--min_generated_hole_change", type=float, default=3.0)
    parser.add_argument("--max_static_baseline_depth_error", type=float, default=0.10)
    parser.add_argument("--max_generated_depth_error", type=float, default=0.15)
    parser.add_argument("--min_wrong_geometry_improvement", type=float, default=0.05)
    parser.add_argument("--apply_depth_policy", action="store_true")
    parser.add_argument("--repair_source", choices=["auto", "projective", "see3d"], default="auto")
    parser.add_argument("--min_projective_depth_supported_fraction", type=float, default=0.85)
    parser.add_argument("--max_projective_nearest_fill_fraction", type=float, default=0.02)
    parser.add_argument("--apply_repair_policy", action="store_true")
    args = parser.parse_args()
    accepted, policies = filter_artifact_guided_views(
        args.stage_root,
        max_synthesized_fraction=args.max_synthesized_fraction,
        min_visible_sharpness=args.min_visible_sharpness,
        min_generated_hole_sharpness=args.min_generated_hole_sharpness,
        min_generated_hole_change=args.min_generated_hole_change,
        max_static_baseline_depth_error=args.max_static_baseline_depth_error,
        max_generated_depth_error=args.max_generated_depth_error,
        min_wrong_geometry_improvement=args.min_wrong_geometry_improvement,
        apply_depth_policy=args.apply_depth_policy,
        repair_source=args.repair_source,
        min_projective_depth_supported_fraction=args.min_projective_depth_supported_fraction,
        max_projective_nearest_fill_fraction=args.max_projective_nearest_fill_fraction,
        apply_repair_policy=args.apply_repair_policy,
    )
    print(f"Accepted artifact-directed views: {accepted}")
    for policy in policies:
        print(
            f"view {policy.index}: accepted={policy.accepted}, "
            f"class={policy.classification}, source={policy.repair_source}, mode={policy.training_mode}, "
            f"baseline_depth_error={policy.baseline_depth_median_relative_error}, "
            f"generated_depth_error={policy.generated_depth_median_relative_error}, "
            f"reasons={policy.reasons}"
        )


if __name__ == "__main__":
    main()
