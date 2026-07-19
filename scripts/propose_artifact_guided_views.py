#!/usr/bin/env python3
"""Propose See3D views from train-view artifact components and clean support.

This is an experimental adapter around a frozen MAtCha/G4Splat checkpoint.  It
does not alter the generic G4Splat novel-view sampler.  Instead, it renders a
dense posed training trajectory, detects coherent rendering failures, looks for
clean multi-view support, and writes only geometrically supported repair views
in the file layout consumed by G4Splat's See3D stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "2d-gaussian-splatting"
for import_root in (REPO_ROOT, GS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from guidance.cam_utils import MiniCam  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

from artifact_guided_repair import (  # noqa: E402
    ArtifactROI3D,
    CameraGeometry,
    backproject_depth,
    build_artifact_components,
    depth_support_metrics,
    fit_support_plane,
    interpolate_c2w,
    match_component_geometry,
    multiview_clean_visible_points,
    multiview_depth_supported_points,
    rasterize_projected_component,
    render_support_plane_depth,
    select_wrong_geometry_interpolation,
)
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2) + "\n", encoding="utf-8")


def _save_rgb(path: Path, image: np.ndarray) -> None:
    value = np.asarray(image)
    if value.dtype != np.uint8:
        value = np.uint8(np.clip(value, 0.0, 1.0) * 255.0 + 0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.asarray(mask, dtype=bool)) * 255, mode="L").save(path)


def _read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def _image_index(image_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in image_root.iterdir():
        if not path.is_file() and not path.is_symlink():
            continue
        if path.stem in result:
            raise RuntimeError(f"Duplicate image stem {path.stem!r} in {image_root}")
        result[path.stem] = path
    return result


def _camera_geometry(camera: Any) -> CameraGeometry:
    return CameraGeometry(
        name=str(camera.image_name),
        width=int(camera.image_width),
        height=int(camera.image_height),
        w2c=camera.world_view_transform.T.detach().cpu().numpy(),
        fx=float(camera.focal_x),
        fy=float(camera.focal_y),
        cx=float(camera.image_width) / 2.0,
        cy=float(camera.image_height) / 2.0,
    )


def _load_no_reference_builder(detector_repo: Path | None):
    if detector_repo is None:
        return None
    detector_repo = detector_repo.expanduser().resolve()
    if not detector_repo.is_dir():
        raise FileNotFoundError(detector_repo)
    if str(detector_repo) not in sys.path:
        sys.path.append(str(detector_repo))
    from valid_support_mask import NoReferenceValidSupportMaskBuilder

    return NoReferenceValidSupportMaskBuilder()


def _no_reference_score(builder: Any, image: np.ndarray, scale: float) -> np.ndarray | None:
    if builder is None:
        return None
    height, width = image.shape[:2]
    scaled_width = max(32, int(round(width * scale)))
    scaled_height = max(32, int(round(height * scale)))
    scaled = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(scaled.copy()).permute(2, 0, 1)
    result = builder.build(tensor)
    score = result.invalid_score.detach().cpu().numpy().astype(np.float32)
    return cv2.resize(score, (width, height), interpolation=cv2.INTER_LINEAR)


def _diagnostic_paths(root: Path, index: int) -> dict[str, Path]:
    stem = f"{index:06d}"
    view_root = root / "views"
    return {
        "render": view_root / f"{stem}.render.png",
        "depth": view_root / f"{stem}.depth.npy",
        "alpha": view_root / f"{stem}.alpha.png",
        "labels": view_root / f"{stem}.labels.png",
        "score": view_root / f"{stem}.score.png",
        "semantic": view_root / f"{stem}.semantic.png",
    }


def _render_diagnostics(
    args: argparse.Namespace,
    dataset: Any,
    pipe: Any,
    mask_lookup: CambridgeMaskLookup,
) -> dict[str, Any]:
    diagnostics_root = Path(args.diagnostics_dir).expanduser().resolve()
    manifest_path = diagnostics_root / "diagnostic_manifest.json"
    if args.reuse_diagnostics and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing = []
        for view in manifest.get("views", []):
            for relative_path in view.get("files", {}).values():
                path = diagnostics_root / relative_path
                if not path.is_file():
                    missing.append(path)
        if not missing:
            print(f"[diagnostics] Reusing {len(manifest['views'])} cached views.")
            return manifest
        raise FileNotFoundError(f"Diagnostic cache is incomplete; first missing file: {missing[0]}")
    if diagnostics_root.exists() and any(diagnostics_root.iterdir()):
        raise FileExistsError(
            f"Diagnostic output is not empty: {diagnostics_root}. "
            "Use --reuse_diagnostics for a complete cache or choose another directory."
        )

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    if int(args.max_diagnostic_views) > 0:
        cameras = cameras[: int(args.max_diagnostic_views)]
    image_paths = _image_index(Path(dataset.source_path) / dataset.images)
    builder = _load_no_reference_builder(
        Path(args.detector_repo) if args.detector_repo else None
    )
    background_value = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_value, dtype=torch.float32, device="cuda")

    diagnostics_root.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for index, camera in enumerate(tqdm(cameras, desc="dense train diagnostics")):
            geometry = _camera_geometry(camera)
            if camera.image_name not in image_paths:
                raise FileNotFoundError(
                    f"No source image for camera {camera.image_name!r} in {dataset.source_path}"
                )
            package = render(camera, gaussians, pipe, background)
            render_rgb = package["render"].detach().cpu().permute(1, 2, 0).numpy()
            depth = package["surf_depth"][0].detach().cpu().numpy().astype(np.float32)
            alpha = package["rend_alpha"][0].detach().cpu().numpy().astype(np.float32)
            target_rgb = camera.original_image.detach().cpu().permute(1, 2, 0).numpy()
            semantic_keep = mask_lookup.get_mask(
                camera.image_name,
                (geometry.height, geometry.width),
                torch.device("cpu"),
            ).numpy()
            no_reference = _no_reference_score(
                builder, render_rgb, float(args.no_reference_scale)
            )
            result = build_artifact_components(
                render_rgb,
                target_rgb,
                alpha,
                depth,
                semantic_keep,
                no_reference_invalid=no_reference,
                residual_threshold=float(args.residual_threshold),
                min_area_fraction=float(args.min_component_fraction),
                max_area_fraction=float(args.max_component_fraction),
            )

            files = _diagnostic_paths(diagnostics_root, index)
            _save_rgb(files["render"], render_rgb)
            np.save(files["depth"], depth)
            _save_mask(files["alpha"], alpha >= 0.5)
            Image.fromarray(result["labels"].astype(np.uint16), mode="I;16").save(
                files["labels"]
            )
            Image.fromarray(np.uint8(np.clip(result["score"], 0.0, 1.0) * 255)).save(
                files["score"]
            )
            _save_mask(files["semantic"], semantic_keep)
            records.append(
                {
                    "index": index,
                    "image_name": camera.image_name,
                    "image_path": str(image_paths[camera.image_name].resolve()),
                    "camera": geometry.as_json(),
                    "summary": result["summary"],
                    "affine": result["affine"].tolist(),
                    "components": result["components"],
                    "semantic_keep_fraction": float(semantic_keep.mean()),
                    "files": {
                        key: str(path.relative_to(diagnostics_root))
                        for key, path in files.items()
                    },
                }
            )

    manifest = {
        "version": 1,
        "mode": "artifact_directed_dense_train_diagnostics",
        "source_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(args.iteration),
        "resolution": int(args.resolution),
        "semantic_mask_indices": list(args.semantic_mask_indices),
        "mask_pickle": str(Path(args.mask_pickle).resolve()),
        "detector_repo": str(Path(args.detector_repo).resolve()) if args.detector_repo else None,
        "views": records,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_view(root: Path, record: dict[str, Any]) -> dict[str, np.ndarray]:
    files = record["files"]
    return {
        "render": _read_rgb(root / files["render"]),
        "depth": np.load(root / files["depth"]).astype(np.float32),
        "alpha": np.asarray(Image.open(root / files["alpha"]).convert("L"), dtype=np.float32)
        / 255.0,
        "labels": np.asarray(Image.open(root / files["labels"]), dtype=np.uint16),
        "score": np.asarray(Image.open(root / files["score"]).convert("L"), dtype=np.float32)
        / 255.0,
        "semantic": np.asarray(
            Image.open(root / files["semantic"]).convert("L"), dtype=np.uint8
        )
        > 127,
    }


def _load_gt(record: dict[str, Any]) -> np.ndarray:
    camera = CameraGeometry.from_json(record["camera"])
    return _read_rgb(Path(record["image_path"]), (camera.width, camera.height))


def _forward_angle(first: CameraGeometry, second: CameraGeometry) -> float:
    first_forward = first.forward / max(float(np.linalg.norm(first.forward)), 1e-12)
    second_forward = second.forward / max(float(np.linalg.norm(second.forward)), 1e-12)
    return float(
        np.rad2deg(np.arccos(np.clip(np.dot(first_forward, second_forward), -1.0, 1.0)))
    )


def _scene_neighbor_scale(cameras: list[CameraGeometry]) -> float:
    centers = np.stack([camera.center for camera in cameras])
    nearest = []
    for index, center in enumerate(centers):
        distances = np.linalg.norm(centers - center[None], axis=1)
        distances[index] = np.inf
        finite = distances[np.isfinite(distances) & (distances > 1e-9)]
        if len(finite):
            nearest.append(float(finite.min()))
    if not nearest:
        return 1.0
    return max(float(np.median(nearest)), 1e-6)


def _select_diverse_supports(
    candidates: list[dict[str, Any]],
    cameras: list[CameraGeometry],
    neighbor_scale: float,
    count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["support_score"], reverse=True):
        camera = cameras[candidate["view_index"]]
        if not selected:
            selected.append(candidate)
            continue
        diverse = all(
            np.linalg.norm(camera.center - cameras[item["view_index"]].center)
            >= 0.35 * neighbor_scale
            or _forward_angle(camera, cameras[item["view_index"]]) >= 3.0
            for item in selected
        )
        if diverse:
            selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


def _canonical_image_name(value: str) -> str:
    """Normalize a dataset path or camera name to the COLMAP camera key."""
    text = str(value).strip().replace("\\", "/")
    suffix = Path(text).suffix
    if suffix:
        text = text[: -len(suffix)]
    return text.replace("/", "__")


def _target_image_names(path: Path | None) -> set[str] | None:
    """Load an optional explicit target list without changing normal global discovery."""
    if path is None:
        return None
    values = {
        _canonical_image_name(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"Target-image file contains no image names: {path}")
    return values


def _support_search(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics_root = Path(args.diagnostics_dir).expanduser().resolve()
    records = manifest["views"]
    cameras = [CameraGeometry.from_json(record["camera"]) for record in records]
    neighbor_scale = _scene_neighbor_scale(cameras)
    target_image_names = _target_image_names(args.target_image_names_file)

    components = []
    for record in records:
        if (
            target_image_names is not None
            and _canonical_image_name(record["image_name"]) not in target_image_names
        ):
            continue
        for component in record["components"]:
            if not bool(component.get("repair_eligible", True)):
                continue
            components.append(
                {
                    "view_index": record["index"],
                    "component_id": component["component_id"],
                    "priority": component["score_mean"]
                    * math.sqrt(component["area_fraction"]),
                    **component,
                }
            )
    components.sort(key=lambda item: item["priority"], reverse=True)

    reports: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    used_target_views: set[int] = set()
    for component in components:
        if len(reports) >= int(args.max_target_components):
            break
        target_index = int(component["view_index"])
        if target_index in used_target_views:
            continue
        used_target_views.add(target_index)
        target_record = records[target_index]
        target_camera = cameras[target_index]
        target_data = _load_view(diagnostics_root, target_record)
        target_gt = _load_gt(target_record)
        target_mask = target_data["labels"] == int(component["component_id"])
        target_mask = cv2.dilate(
            target_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
        ).astype(bool)
        depth_mask = target_mask & (target_data["alpha"] >= 0.5)
        target_points, _ = backproject_depth(
            target_data["depth"], depth_mask, target_camera, stride=2
        )

        pose_candidates = []
        for candidate_index, candidate_camera in enumerate(cameras):
            if candidate_index == target_index:
                continue
            baseline = float(np.linalg.norm(candidate_camera.center - target_camera.center))
            if baseline < 0.15 * neighbor_scale:
                continue
            pose_score = baseline / neighbor_scale + 0.35 * (
                _forward_angle(target_camera, candidate_camera) / 15.0
            )
            pose_candidates.append((pose_score, candidate_index, baseline))
        pose_candidates.sort(key=lambda item: item[0])

        candidate_reports = []
        triangulated_by_view: dict[int, np.ndarray] = {}
        for _, candidate_index, baseline in pose_candidates[: int(args.neighbor_count)]:
            candidate_record = records[candidate_index]
            candidate_camera = cameras[candidate_index]
            candidate_data = _load_view(diagnostics_root, candidate_record)
            candidate_gt = _load_gt(candidate_record)
            candidate_clean = (
                (candidate_data["labels"] == 0)
                & candidate_data["semantic"]
                & (candidate_data["score"] <= float(args.max_clean_artifact_score))
            )

            depth_metrics = depth_support_metrics(
                target_points,
                candidate_camera,
                candidate_data["depth"],
                candidate_data["alpha"],
                candidate_clean,
                relative_depth_threshold=float(args.relative_depth_threshold),
            )
            match_metrics, triangulated = match_component_geometry(
                target_gt,
                target_mask,
                target_camera,
                candidate_gt,
                candidate_clean,
                candidate_camera,
                max_reprojection_error=float(args.max_reprojection_error),
                min_parallax_degrees=float(args.min_parallax_degrees),
            )
            depth_ok = (
                depth_metrics["overlap_fraction"] >= 0.20
                and depth_metrics["depth_consistent_fraction"] >= 0.45
                and depth_metrics["clean_fraction"] >= 0.75
            )
            feature_ok = (
                match_metrics["triangulated_count"] >= int(args.min_pair_triangulated)
                and match_metrics["median_reprojection_error"]
                <= float(args.max_reprojection_error)
                and match_metrics["median_parallax_degrees"]
                >= float(args.min_parallax_degrees)
            )
            # Rendered-depth agreement is useful for visibility diagnostics, but
            # it is generated by the model being audited and cannot establish
            # that the geometry is correct. Clean support requires independent
            # calibrated feature triangulation.
            if not feature_ok:
                continue
            score = (
                0.10 * depth_metrics["depth_consistent_fraction"]
                + 2.5 * depth_metrics["clean_fraction"]
                + min(match_metrics["triangulated_count"] / 40.0, 1.0)
                + min(match_metrics["median_parallax_degrees"] / 5.0, 1.0)
                - 0.5 * float(candidate_record["summary"]["artifact_fraction"])
                - 0.02 * baseline / neighbor_scale
            )
            candidate_reports.append(
                {
                    "view_index": candidate_index,
                    "image_name": candidate_record["image_name"],
                    "baseline": baseline,
                    "forward_angle_degrees": _forward_angle(
                        target_camera, candidate_camera
                    ),
                    "model_depth_ok": depth_ok,
                    "feature_ok": feature_ok,
                    "depth": depth_metrics,
                    "matching": match_metrics,
                    "support_score": score,
                }
            )
            triangulated_by_view[candidate_index] = triangulated

        selected = _select_diverse_supports(
            candidate_reports,
            cameras,
            neighbor_scale,
            int(args.support_views),
        )
        report = {
            "target_view_index": target_index,
            "target_image_name": target_record["image_name"],
            "component": component,
            "candidate_count": len(candidate_reports),
            "selected_support": selected,
            "classification": "occlusion_or_ood",
            "dense_consensus_count": 0,
            "triangulated_consensus_count": 0,
            "generated": False,
        }
        if len(selected) < 2:
            report["reason"] = "fewer_than_two_diverse_clean_support_views"
            reports.append(report)
            continue

        support_view_data = []
        clean_visibility_views = []
        triangulated_sets = []
        for support in selected:
            support_index = int(support["view_index"])
            support_record = records[support_index]
            support_data = _load_view(diagnostics_root, support_record)
            support_view_data.append(
                (
                    cameras[support_index],
                    support_data["depth"],
                    support_data["alpha"],
                    (support_data["labels"] == 0)
                    & support_data["semantic"]
                    & (support_data["score"] <= float(args.max_clean_artifact_score)),
                )
            )
            clean_visibility_views.append(
                (
                    cameras[support_index],
                    (support_data["labels"] == 0)
                    & support_data["semantic"]
                    & (support_data["score"] <= float(args.max_clean_artifact_score)),
                )
            )
            points = triangulated_by_view.get(support_index)
            if points is not None and len(points):
                triangulated_sets.append(points)

        dense_supported, _ = multiview_depth_supported_points(
            target_points,
            support_view_data,
            relative_depth_threshold=float(args.relative_depth_threshold),
            min_views=2,
        )
        triangulated = (
            np.concatenate(triangulated_sets, axis=0)
            if triangulated_sets
            else np.empty((0, 3), dtype=np.float64)
        )
        triangulated_supported, triangulated_clean_counts = multiview_clean_visible_points(
            triangulated,
            clean_visibility_views,
            min_views=2,
        )
        report["dense_consensus_count"] = len(dense_supported)
        report["dense_consensus_evidence"] = "diagnostic_only_current_model_rendered_depth"
        report["target_point_count"] = len(target_points)
        report["dense_consensus_fraction"] = float(
            len(dense_supported) / max(len(target_points), 1)
        )
        report["triangulated_consensus_count"] = len(triangulated_supported)
        report["triangulated_consensus_evidence"] = (
            "independent_calibrated_feature_geometry_and_clean_image_visibility"
        )
        report["triangulated_clean_visibility_count_histogram"] = {
            str(count): int(np.count_nonzero(triangulated_clean_counts == count))
            for count in np.unique(triangulated_clean_counts)
        }
        triangulated_target_depth = depth_support_metrics(
            triangulated_supported,
            target_camera,
            target_data["depth"],
            target_data["alpha"],
            target_data["semantic"],
            relative_depth_threshold=float(args.relative_depth_threshold),
        )
        report["triangulated_target_depth"] = triangulated_target_depth
        if (
            component["alpha_mean"] < 0.5
            and len(triangulated_supported) >= int(args.min_triangulated_consensus)
        ):
            report["classification"] = "coverage_hole"
            report["classification_scope"] = "triangulated_patch"
            report["repair_points"] = triangulated_supported
        elif len(triangulated_supported) >= int(args.min_triangulated_consensus):
            target_depth_agreement = triangulated_target_depth[
                "depth_consistent_fraction"
            ]
            if target_depth_agreement >= float(args.static_patch_depth_agreement):
                report["classification"] = "supported_static_patch"
                report["classification_scope"] = "triangulated_patch"
                report["repair_points"] = triangulated_supported
            elif target_depth_agreement <= float(args.wrong_geometry_depth_agreement):
                report["classification"] = "wrong_geometry"
                report["classification_scope"] = "triangulated_patch"
                report["repair_points"] = triangulated_supported
            else:
                report["reason"] = "ambiguous_target_vs_clean_view_depth"
        else:
            report["reason"] = "clean_views_do_not_agree_on_component_geometry"
        reports.append(report)
        if "repair_points" in report:
            accepted.append(report)
        if len(accepted) >= int(args.max_pseudo_views):
            break

    return reports, accepted


def _infer_chart_count(source_path: Path) -> int:
    charts_path = source_path / "charts_data.npz"
    if not charts_path.is_file():
        raise FileNotFoundError(charts_path)
    with np.load(charts_path) as charts:
        for key in ("depths", "prior_depths", "pts", "confs"):
            if key in charts and charts[key].ndim > 0:
                return int(charts[key].shape[0])
    raise RuntimeError(f"Could not infer chart count from {charts_path}")


def _build_contact_sheet(
    diagnostics_root: Path,
    records: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = []
    for report in reports[:12]:
        record = records[report["target_view_index"]]
        data = _load_view(diagnostics_root, record)
        gt = _load_gt(record)
        labels = data["labels"] == int(report["component"]["component_id"])
        overlay = data["render"].copy()
        overlay[labels] = 0.55 * overlay[labels] + 0.45 * np.asarray([1.0, 0.1, 0.1])
        tiles = []
        for title, image in (("GT", gt), ("retained_v2", data["render"]), ("component", overlay)):
            tile = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255)).resize(
                (320, 180), Image.Resampling.LANCZOS
            )
            canvas = Image.new("RGB", (320, 208), "white")
            canvas.paste(tile, (0, 28))
            ImageDraw.Draw(canvas).text((6, 7), title, fill="black")
            tiles.append(canvas)
        row = Image.new("RGB", (960, 236), "white")
        for index, tile in enumerate(tiles):
            row.paste(tile, (320 * index, 28))
        label = (
            f"{record['image_name']} c{report['component']['component_id']} | "
            f"{report['classification']} | support={len(report['selected_support'])} | "
            f"dense={report['dense_consensus_count']} tri={report['triangulated_consensus_count']}"
        )
        ImageDraw.Draw(row).text((6, 7), label, fill="black")
        rows.append(row)
    if not rows:
        return
    sheet = Image.new("RGB", (960, 236 * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, 236 * index))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _make_pseudo_camera(
    target_camera: CameraGeometry,
    support_camera: CameraGeometry,
    fraction: float,
    name: str,
) -> tuple[np.ndarray, MiniCam, CameraGeometry]:
    c2w = interpolate_c2w(target_camera.c2w, support_camera.c2w, float(fraction))
    pseudo = MiniCam(
        c2w.astype(np.float32),
        target_camera.width,
        target_camera.height,
        2.0 * math.atan(target_camera.height / (2.0 * target_camera.fy)),
        2.0 * math.atan(target_camera.width / (2.0 * target_camera.fx)),
    )
    geometry = CameraGeometry(
        name=name,
        width=target_camera.width,
        height=target_camera.height,
        w2c=np.linalg.inv(c2w),
        fx=target_camera.fx,
        fy=target_camera.fy,
        cx=target_camera.cx,
        cy=target_camera.cy,
    )
    return c2w, pseudo, geometry


def _write_targeted_stage(
    args: argparse.Namespace,
    dataset: Any,
    pipe: Any,
    manifest: dict[str, Any],
    reports: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_source = Path(args.artifact_source_path).expanduser().resolve()
    stage_root = artifact_source / "see3d_render" / f"stage{args.stage}"
    ref_root = artifact_source / "see3d_render" / "artifact-ref-views"
    if args.replace_stage:
        adapter_manifest_path = artifact_source.parent / "warmstart_manifest.json"
        if not adapter_manifest_path.is_file():
            raise FileNotFoundError(
                f"Refusing Stage replacement without adapter manifest: {adapter_manifest_path}"
            )
        adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
        if Path(adapter_manifest.get("scene_dir", "")).resolve() != artifact_source:
            raise RuntimeError(
                f"Adapter manifest does not own artifact source {artifact_source}"
            )
        for generated_root in (stage_root, ref_root):
            if generated_root.exists():
                shutil.rmtree(generated_root)
    if stage_root.exists() and any(stage_root.iterdir()):
        raise FileExistsError(f"Targeted See3D stage is not empty: {stage_root}")
    if not accepted:
        raise RuntimeError("No component passed multi-view clean-support attribution")
    stage_root.mkdir(parents=True, exist_ok=True)
    select_root = stage_root / "select-gs"
    select_root.mkdir()
    if ref_root.exists() and any(ref_root.iterdir()):
        raise FileExistsError(f"Artifact reference directory is not empty: {ref_root}")
    ref_root.mkdir(parents=True, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    checkpoint = (
        Path(dataset.model_path)
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    gaussians.load_ply(str(checkpoint))
    background_value = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_value, dtype=torch.float32, device="cuda")
    records = manifest["views"]
    cameras = [CameraGeometry.from_json(record["camera"]) for record in records]
    camera_archive: dict[str, Any] = {
        "train_views": _infer_chart_count(artifact_source),
    }
    output_records = []
    all_repair_points = []
    reference_indices = []

    with torch.no_grad():
        for output_index, report in enumerate(accepted):
            target_index = int(report["target_view_index"])
            clean_support = sorted(
                (
                    support
                    for support in report["selected_support"]
                    if support["depth"]["clean_fraction"]
                    >= float(args.min_reference_clean_fraction)
                ),
                key=lambda support: (
                    bool(support["feature_ok"]),
                    support["depth"]["clean_fraction"],
                    support["support_score"],
                ),
                reverse=True,
            )
            if len(clean_support) < 2:
                report["reason"] = "fewer_than_two_high_clean_fraction_references"
                continue
            support_index = int(clean_support[0]["view_index"])
            target_camera = cameras[target_index]
            support_camera = cameras[support_index]
            repair_points = np.asarray(report["repair_points"], dtype=np.float64)
            interpolation_fraction = float(args.interpolation_fraction)
            if report["classification"] == "wrong_geometry":
                scan = []
                for fraction in args.wrong_geometry_interpolation_fractions:
                    _, scan_camera, scan_geometry = _make_pseudo_camera(
                        target_camera,
                        support_camera,
                        float(fraction),
                        f"artifact_scan_{output_index:06d}",
                    )
                    scan_package = render(scan_camera, gaussians, pipe, background)
                    scan_depth = scan_package["surf_depth"][0].detach().cpu().numpy()
                    scan_alpha = scan_package["rend_alpha"][0].detach().cpu().numpy()
                    scan_metrics = depth_support_metrics(
                        repair_points,
                        scan_geometry,
                        scan_depth,
                        scan_alpha,
                        np.ones_like(scan_depth, dtype=bool),
                        relative_depth_threshold=float(args.relative_depth_threshold),
                    )
                    scan.append(
                        {
                            "fraction": float(fraction),
                            "depth_metrics": scan_metrics,
                        }
                    )
                    del scan_package
                selected_scan = select_wrong_geometry_interpolation(
                    scan,
                    min_median_relative_depth_error=float(
                        args.wrong_geometry_min_pseudo_depth_error
                    ),
                    min_overlap_fraction=float(args.wrong_geometry_min_scan_overlap),
                    min_projected_points=int(args.wrong_geometry_min_scan_points),
                )
                report["interpolation_scan"] = scan
                if selected_scan is None:
                    report["reason"] = "wrong_geometry_disappears_before_supported_pseudo_pose"
                    continue
                interpolation_fraction = float(selected_scan["fraction"])

            c2w, pseudo, pseudo_geometry = _make_pseudo_camera(
                target_camera,
                support_camera,
                interpolation_fraction,
                f"artifact_{output_index:06d}",
            )
            repair_mask = rasterize_projected_component(
                repair_points,
                pseudo_geometry,
                point_radius=int(args.repair_point_radius),
                close_radius=int(args.repair_close_radius),
                max_area_fraction=float(args.max_repair_fraction),
            )
            repair_fraction = float(repair_mask.mean())
            if not float(args.min_repair_fraction) <= repair_fraction <= float(
                args.max_repair_fraction
            ):
                report["reason"] = f"projected_repair_fraction_{repair_fraction:.6f}"
                continue

            clean_depth = None
            clean_depth_valid = None
            support_plane = None
            if report["classification"] in {"wrong_geometry", "coverage_hole"}:
                try:
                    support_plane, plane_inliers, plane_metrics = fit_support_plane(
                        repair_points,
                        min_points=int(args.min_support_plane_points),
                    )
                except ValueError as error:
                    report["reason"] = f"clean_support_plane_failed: {error}"
                    continue
                if (
                    plane_metrics["inlier_fraction"] < float(args.min_support_plane_inlier_fraction)
                    or plane_metrics["normalized_median_residual"]
                    > float(args.max_support_plane_normalized_residual)
                    or plane_metrics["planarity_ratio"] > float(args.max_support_plane_planarity_ratio)
                ):
                    report["reason"] = "clean_support_points_are_not_locally_planar"
                    report["clean_support_plane"] = plane_metrics
                    continue
                clean_depth, clean_depth_valid = render_support_plane_depth(
                    pseudo_geometry,
                    support_plane,
                    repair_mask,
                )
                plane_coverage = float(clean_depth_valid[repair_mask].mean())
                if plane_coverage < float(args.min_support_plane_mask_coverage):
                    report["reason"] = f"clean_support_plane_coverage_{plane_coverage:.6f}"
                    continue
                report["clean_support_plane"] = {
                    **plane_metrics,
                    "mask_coverage": plane_coverage,
                    "plane_world": support_plane.tolist(),
                    "inlier_point_count": int(np.count_nonzero(plane_inliers)),
                }

            package = render(pseudo, gaussians, pipe, background)
            rgb = package["render"].detach().cpu().permute(1, 2, 0).numpy()
            depth = package["surf_depth"][0].detach().cpu().numpy().astype(np.float32)
            alpha = package["rend_alpha"][0].detach().cpu().numpy().astype(np.float32)
            known = ~repair_mask
            stem = f"{len(output_records):06d}"
            _save_rgb(select_root / f"ori_warp_frame{stem}.png", rgb)
            Image.fromarray(depth, mode="F").save(select_root / f"depth_frame{stem}.tiff")
            np.save(select_root / f"alpha_{stem}.npy", alpha)
            _save_mask(select_root / f"alpha_mask_frame{stem}.png", alpha > 0.99)
            _save_rgb(select_root / f"alpha_warp_frame{stem}.png", rgb * (alpha > 0.99)[..., None])
            _save_rgb(select_root / f"warp_frame{stem}.png", rgb * known[..., None])
            _save_mask(select_root / f"mask_frame{stem}.png", known)
            repair_points_path = stage_root / f"repair_points_frame{stem}.npy"
            np.save(repair_points_path, repair_points.astype(np.float32))
            roi = ArtifactROI3D(
                roi_id=f"artifact_{stem}",
                points=repair_points,
                source_view_ids=(target_index,),
                classification=str(report["classification"]),
                plane_world=support_plane,
                slab_thickness=float(
                    report.get("clean_support_plane", {}).get(
                        "normalized_median_residual", 0.0
                    )
                ),
                evidence_view_ids=tuple(
                    int(support["view_index"]) for support in clean_support
                ),
                evidence={
                    "triangulated_consensus_count": int(
                        report.get("triangulated_consensus_count", 0)
                    ),
                    "dense_consensus_count": int(report.get("dense_consensus_count", 0)),
                },
            )
            roi_path = roi.save(stage_root / f"roi3d_frame{stem}.npz")
            if clean_depth is not None:
                clean_depth_path = select_root / f"clean_support_depth_frame{stem}.tiff"
                clean_depth_mask_path = select_root / f"clean_support_depth_mask_frame{stem}.png"
                Image.fromarray(clean_depth, mode="F").save(clean_depth_path)
                _save_mask(clean_depth_mask_path, clean_depth_valid)
                report["clean_support_depth_path"] = str(clean_depth_path.resolve())
                report["clean_support_depth_mask_path"] = str(
                    clean_depth_mask_path.resolve()
                )

            archive_index = len(output_records)
            w2c = np.linalg.inv(c2w)
            camera_archive[f"R_{archive_index:06d}"] = c2w[:3, :3]
            camera_archive[f"T_{archive_index:06d}"] = w2c[:3, 3]
            camera_archive[f"FoVx_{archive_index:06d}"] = pseudo.FoVx
            camera_archive[f"FoVy_{archive_index:06d}"] = pseudo.FoVy
            camera_archive[f"image_width_{archive_index:06d}"] = pseudo.image_width
            camera_archive[f"image_height_{archive_index:06d}"] = pseudo.image_height
            report["generated"] = True
            report["pseudo_view_index"] = archive_index
            report["repair_mask_fraction"] = repair_fraction
            report["interpolation_fraction"] = interpolation_fraction
            report["pseudo_camera"] = pseudo_geometry.as_json()
            report["repair_points_path"] = str(repair_points_path.resolve())
            report["roi3d_path"] = str(roi_path.resolve())
            report["roi3d"] = roi.summary()
            report.pop("repair_points", None)
            output_records.append(report)
            all_repair_points.append(repair_points)
            report["clean_reference_support"] = clean_support[:2]
            for support in clean_support[:2]:
                support_index = int(support["view_index"])
                if support_index not in reference_indices:
                    reference_indices.append(support_index)

    if not output_records:
        raise RuntimeError("Attributed components produced no bounded pseudo-view repair masks")
    camera_archive["n_views"] = len(output_records)
    np.savez(
        stage_root / f"stage{args.stage}_see3d_cameras.npz",
        **camera_archive,
    )
    points = np.concatenate(all_repair_points, axis=0)
    trimesh.PointCloud(points).export(
        stage_root / f"stage{args.stage}_need_inpaint_views_points.ply"
    )

    reference_indices = reference_indices[: int(args.max_reference_views)]
    for reference_index in reference_indices:
        source = Path(records[reference_index]["image_path"])
        destination = ref_root / f"{records[reference_index]['image_name']}{source.suffix}"
        shutil.copy2(source, destination)

    stage_manifest = {
        "version": 1,
        "mode": "artifact_directed_clean_support_pseudo_views",
        "frozen_model": str(Path(dataset.model_path).resolve()),
        "iteration": int(args.iteration),
        "diagnostics_manifest": str(
            Path(args.diagnostics_dir).expanduser().resolve() / "diagnostic_manifest.json"
        ),
        "reference_views": [
            {
                "view_index": index,
                "image_name": records[index]["image_name"],
                "path": str(
                    (
                        ref_root
                        / f"{records[index]['image_name']}{Path(records[index]['image_path']).suffix}"
                    ).resolve()
                ),
            }
            for index in reference_indices
        ],
        "pseudo_views": output_records,
        "rejected_or_diagnostic_components": reports,
    }
    _write_json(stage_root / "artifact_guided_manifest.json", stage_manifest)
    _write_json(
        stage_root / "accepted_view_indices.json",
        list(range(len(output_records))),
    )
    return stage_manifest


def build_parser() -> tuple[argparse.ArgumentParser, ModelParams, PipelineParams]:
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--artifact_source_path", type=Path, required=True)
    parser.add_argument("--diagnostics_dir", type=Path, required=True)
    parser.add_argument("--mask_pickle", type=Path, required=True)
    parser.add_argument("--mask_dataset_path", type=Path, required=True)
    parser.add_argument(
        "--target_image_names_file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited target camera names or dataset paths. "
            "When set, support attribution only considers components from these views."
        ),
    )
    parser.add_argument("--semantic_mask_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--detector_repo", type=Path, default=Path("/root/STDLoc"))
    parser.add_argument("--no_reference_scale", type=float, default=0.5)
    parser.add_argument("--residual_threshold", type=float, default=0.16)
    parser.add_argument("--min_component_fraction", type=float, default=0.004)
    parser.add_argument("--max_component_fraction", type=float, default=0.40)
    parser.add_argument("--max_target_components", type=int, default=16)
    parser.add_argument("--max_pseudo_views", type=int, default=4)
    parser.add_argument("--neighbor_count", type=int, default=30)
    parser.add_argument("--support_views", type=int, default=3)
    parser.add_argument("--relative_depth_threshold", type=float, default=0.10)
    parser.add_argument("--max_clean_artifact_score", type=float, default=0.20)
    parser.add_argument("--max_reprojection_error", type=float, default=3.0)
    parser.add_argument("--min_parallax_degrees", type=float, default=0.5)
    parser.add_argument("--min_pair_triangulated", type=int, default=10)
    parser.add_argument("--min_dense_consensus", type=int, default=64)
    parser.add_argument("--min_dense_consensus_fraction", type=float, default=0.25)
    parser.add_argument("--min_triangulated_consensus", type=int, default=10)
    parser.add_argument("--static_patch_depth_agreement", type=float, default=0.50)
    parser.add_argument("--wrong_geometry_depth_agreement", type=float, default=0.25)
    parser.add_argument("--interpolation_fraction", type=float, default=0.60)
    parser.add_argument(
        "--wrong_geometry_interpolation_fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30],
    )
    parser.add_argument("--wrong_geometry_min_pseudo_depth_error", type=float, default=0.30)
    parser.add_argument("--wrong_geometry_min_scan_overlap", type=float, default=0.50)
    parser.add_argument("--wrong_geometry_min_scan_points", type=int, default=6)
    parser.add_argument("--repair_point_radius", type=int, default=7)
    parser.add_argument("--repair_close_radius", type=int, default=10)
    parser.add_argument("--min_repair_fraction", type=float, default=0.003)
    parser.add_argument("--max_repair_fraction", type=float, default=0.25)
    parser.add_argument("--max_reference_views", type=int, default=4)
    parser.add_argument("--min_reference_clean_fraction", type=float, default=0.75)
    parser.add_argument("--min_support_plane_points", type=int, default=6)
    parser.add_argument("--min_support_plane_inlier_fraction", type=float, default=0.65)
    parser.add_argument("--max_support_plane_normalized_residual", type=float, default=0.03)
    parser.add_argument("--max_support_plane_planarity_ratio", type=float, default=0.25)
    parser.add_argument("--min_support_plane_mask_coverage", type=float, default=0.95)
    parser.add_argument(
        "--max_diagnostic_views",
        type=int,
        default=0,
        help="Debug-only prefix limit; zero processes the complete dense trajectory.",
    )
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--reuse_diagnostics", action="store_true")
    parser.add_argument(
        "--replace_stage",
        action="store_true",
        help="Replace only this isolated artifact Stage and its dedicated references.",
    )
    parser.add_argument(
        "--diagnostics_only",
        action="store_true",
        help="Stop after writing train-view RGB/depth/alpha artifact diagnostics.",
    )
    parser.add_argument(
        "--support_diagnostics_only",
        action="store_true",
        help="Run clean-support attribution on cached diagnostics without writing a pseudo-view Stage.",
    )
    return parser, model, pipeline


def main() -> None:
    parser, model, pipeline = build_parser()
    args = get_combined_args(parser)
    safe_state(False)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    if Path(dataset.source_path).resolve() != Path(args.mask_dataset_path).resolve():
        raise ValueError(
            "Dense render source and --mask_dataset_path must be the same staged dataset"
        )
    mask_lookup = CambridgeMaskLookup(
        Path(args.mask_dataset_path),
        Path(args.mask_pickle),
        list(args.semantic_mask_indices),
    )
    manifest = _render_diagnostics(args, dataset, pipe, mask_lookup)
    if args.diagnostics_only:
        print(json.dumps({"diagnostic_views": len(manifest["views"])}, indent=2))
        return
    reports, accepted = _support_search(args, manifest)
    diagnostics_root = Path(args.diagnostics_dir).expanduser().resolve()
    serializable_reports = []
    for report in reports:
        serializable_reports.append(
            {key: value for key, value in report.items() if key != "repair_points"}
        )
    _write_json(
        diagnostics_root / "component_support_report.json",
        {
            "target_image_names_file": (
                None
                if args.target_image_names_file is None
                else str(Path(args.target_image_names_file).resolve())
            ),
            "target_image_names": (
                None
                if args.target_image_names_file is None
                else sorted(_target_image_names(args.target_image_names_file) or [])
            ),
            "component_count": len(reports),
            "accepted_count": len(accepted),
            "reports": serializable_reports,
        },
    )
    _build_contact_sheet(
        diagnostics_root,
        manifest["views"],
        reports,
        diagnostics_root / "component_support_contact_sheet.jpg",
    )
    if args.support_diagnostics_only:
        print(
            json.dumps(
                {
                    "diagnostic_views": len(manifest["views"]),
                    "inspected_components": len(reports),
                    "supported_components": len(accepted),
                    "support_report": str(
                        (diagnostics_root / "component_support_report.json").resolve()
                    ),
                },
                indent=2,
            )
        )
        return
    stage_manifest = _write_targeted_stage(
        args, dataset, pipe, manifest, reports, accepted
    )
    print(
        json.dumps(
            {
                "diagnostic_views": len(manifest["views"]),
                "inspected_components": len(reports),
                "supported_components": len(accepted),
                "generated_pseudo_views": len(stage_manifest["pseudo_views"]),
                "stage_root": str(
                    Path(args.artifact_source_path).resolve()
                    / "see3d_render"
                    / f"stage{args.stage}"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
