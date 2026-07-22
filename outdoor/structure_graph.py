"""Build a compact static-structure graph from hard-gated aligned Charts.

Cambridge's staged posed data does not always carry a COLMAP ``points3D.bin``
track model.  In that case this module constructs the same *selection unit*
from the fixed-camera MASt3R point maps after the alignment gate.  The output
records that provenance explicitly; it never pretends these are original SfM
tracks.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MAST3R_ROOT = REPO_ROOT / "mast3r"
if str(MAST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(MAST3R_ROOT))

from colmap.read_write_model import read_images_binary  # noqa: E402
from matcha.cambridge_masks import CambridgeMaskLookup  # noqa: E402
from view_quality_control.poses import qvec_to_rotmat  # noqa: E402


STRUCTURE_GRAPH_VERSION = "outdoor-static-structure-graph-v1"


def triangulation_angle_degrees(
    positions: np.ndarray,
    first_center: np.ndarray,
    second_center: np.ndarray,
) -> np.ndarray:
    """Return true per-structure-unit triangulation angles in degrees."""
    first = np.asarray(positions, dtype=np.float64) - np.asarray(first_center, dtype=np.float64)
    second = np.asarray(positions, dtype=np.float64) - np.asarray(second_center, dtype=np.float64)
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    denom = np.maximum(first_norm * second_norm, 1e-12)
    cosine = np.clip(np.sum(first * second, axis=-1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float32)


def _normalise_confidence(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    upper = max(float(np.quantile(finite, 0.95)), 1e-6)
    return np.clip(values / upper, 0.0, 1.0).astype(np.float32)


def _compact_codes(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(codes)
    return np.searchsorted(unique, codes).astype(np.int32), unique


def _quantise_points(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bins: int,
) -> np.ndarray:
    span = np.maximum(upper - lower, 1e-6)
    quantised = np.floor((points - lower) / span * bins).astype(np.int64)
    quantised = np.clip(quantised, 0, bins - 1)
    return quantised[:, 0] * bins * bins + quantised[:, 1] * bins + quantised[:, 2]


def build_structure_graph_from_samples(
    sample_points: list[np.ndarray],
    sample_confidences: list[np.ndarray],
    camera_centers: np.ndarray,
    *,
    bins: int = 40,
    max_units: int = 4096,
    min_unit_views: int = 2,
    block_bins: int = 4,
    valid_fractions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate static point-map samples into bounded multi-view units.

    A singleton spatial cell is deliberately discarded.  This avoids the old
    inverse-frequency voxel objective's failure mode of rewarding a lone
    erroneous point simply because it is rare.
    """
    if bins < 2 or block_bins < 1:
        raise ValueError("bins must be >= 2 and block_bins must be >= 1")
    if len(sample_points) != len(sample_confidences):
        raise ValueError("sample_points and sample_confidences must have equal length")
    if len(sample_points) != len(camera_centers):
        raise ValueError("one camera center is required for every sampled view")
    if min_unit_views < 1:
        raise ValueError("min_unit_views must be positive")
    nonempty = [points for points in sample_points if len(points)]
    if not nonempty:
        raise RuntimeError("No static aligned point-map samples are available")

    all_points = np.concatenate(nonempty, axis=0).astype(np.float32, copy=False)
    lower = np.quantile(all_points, 0.005, axis=0)
    upper = np.quantile(all_points, 0.995, axis=0)
    upper = np.maximum(upper, lower + 1e-6)

    per_view_codes: list[np.ndarray] = []
    per_view_confidences: list[np.ndarray] = []
    for points, confidences in zip(sample_points, sample_confidences):
        if len(points) == 0:
            per_view_codes.append(np.empty(0, dtype=np.int64))
            per_view_confidences.append(np.empty(0, dtype=np.float32))
            continue
        per_view_codes.append(_quantise_points(points, lower, upper, bins))
        per_view_confidences.append(_normalise_confidence(confidences))

    all_codes = np.concatenate([codes for codes in per_view_codes if len(codes)])
    compact_all, global_codes = _compact_codes(all_codes)
    global_count = np.bincount(compact_all, minlength=len(global_codes)).astype(np.int64)
    point_sum = np.zeros((len(global_codes), 3), dtype=np.float64)
    point_count = np.zeros(len(global_codes), dtype=np.int64)
    view_count = np.zeros(len(global_codes), dtype=np.int32)

    cursor = 0
    for points, codes in zip(sample_points, per_view_codes):
        if not len(codes):
            continue
        compact = np.searchsorted(global_codes, codes)
        np.add.at(point_sum, compact, points)
        np.add.at(point_count, compact, 1)
        view_count[np.unique(compact)] += 1
        cursor += len(codes)
    if cursor != len(all_codes):
        raise RuntimeError("Structure graph sample bookkeeping mismatch")

    # Require independent observations and retain the most reliable spatial
    # support if the scene is very large.  This makes selection scale with
    # structure coverage instead of an arbitrary fixed voxel population.
    raw_score = np.sqrt(np.maximum(view_count, 1)) * np.log1p(global_count)
    keep = np.flatnonzero(view_count >= min_unit_views)
    if not len(keep):
        raise RuntimeError(
            "No static structure unit has multi-view support; inspect fixed-camera alignment and masks."
        )
    if len(keep) > max_units:
        order = np.argsort(raw_score[keep], kind="stable")[-max_units:]
        keep = np.sort(keep[order])
    global_to_local = np.full(len(global_codes), -1, dtype=np.int32)
    global_to_local[keep] = np.arange(len(keep), dtype=np.int32)

    unit_centers = (point_sum[keep] / np.maximum(point_count[keep, None], 1)).astype(np.float32)
    unit_view_count = view_count[keep].astype(np.int16)
    unit_sample_count = global_count[keep].astype(np.int32)
    unit_weight = (
        np.sqrt(unit_view_count.astype(np.float64))
        * np.clip(np.log1p(unit_sample_count) / np.log(33.0), 0.0, 1.0)
    )
    unit_weight = unit_weight / max(float(unit_weight.sum()), 1e-12)

    support = np.zeros((len(sample_points), len(keep)), dtype=np.float32)
    for view_index, (codes, confidences) in enumerate(zip(per_view_codes, per_view_confidences)):
        if not len(codes):
            continue
        local = global_to_local[np.searchsorted(global_codes, codes)]
        local_keep = local >= 0
        if not np.any(local_keep):
            continue
        local = local[local_keep]
        confidences = confidences[local_keep]
        unique_local, inverse = np.unique(local, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique_local)).astype(np.float32)
        confidence_sums = np.bincount(
            inverse, weights=confidences, minlength=len(unique_local)
        ).astype(np.float32)
        # Pixel support is capped, so a nearby high-resolution facade cannot
        # consume the whole Chart budget merely through sample multiplicity.
        values = np.clip(np.log1p(counts) / np.log(9.0), 0.0, 1.0)
        values *= confidence_sums / np.maximum(counts, 1.0)
        support[view_index, unique_local] = values

    # Overlapping spatial blocks give local facades a voice even when global
    # extent is dominated by a long camera trajectory.
    block_lower = np.quantile(unit_centers, 0.02, axis=0)
    block_upper = np.quantile(unit_centers, 0.98, axis=0)
    block_upper = np.maximum(block_upper, block_lower + 1e-6)
    block_codes = _quantise_points(unit_centers, block_lower, block_upper, block_bins)
    block_ids, raw_block_codes = _compact_codes(block_codes)
    block_centers = np.zeros((len(raw_block_codes), 3), dtype=np.float32)
    block_weight = np.zeros(len(raw_block_codes), dtype=np.float64)
    for block in range(len(raw_block_codes)):
        member = block_ids == block
        local_weight = unit_weight[member]
        block_weight[block] = local_weight.sum()
        block_centers[block] = np.average(unit_centers[member], axis=0, weights=local_weight)
    block_support = np.zeros((len(sample_points), len(raw_block_codes)), dtype=np.float32)
    for block in range(len(raw_block_codes)):
        member = block_ids == block
        if np.any(member):
            block_support[:, block] = support[:, member].max(axis=1)

    return {
        "unit_centers": unit_centers,
        "unit_weight": unit_weight.astype(np.float32),
        "unit_support": support,
        "unit_view_count": unit_view_count,
        "unit_sample_count": unit_sample_count,
        "unit_block_ids": block_ids,
        "block_centers": block_centers,
        "block_weight": block_weight.astype(np.float32),
        "block_support": block_support,
        "camera_centers": np.asarray(camera_centers, dtype=np.float32),
        "bounds": np.stack([lower, upper]).astype(np.float32),
        "block_bounds": np.stack([block_lower, block_upper]).astype(np.float32),
        "valid_fractions": (
            np.asarray(valid_fractions, dtype=np.float32)
            if valid_fractions is not None
            else np.ones(len(sample_points), dtype=np.float32)
        ),
    }


def _active_gate_names(gate_report: Path) -> set[str]:
    payload = json.loads(Path(gate_report).read_text(encoding="utf-8"))
    names = {
        Path(str(record["image_name"])).name
        for record in payload.get("records", [])
        if record.get("rejected") is not True
    }
    if not names:
        raise RuntimeError(f"Alignment gate has no accepted Charts: {gate_report}")
    return names


def _graph_edges(graph: dict[str, np.ndarray], *, min_angle_degrees: float) -> list[dict[str, Any]]:
    support = graph["unit_support"]
    unit_centers = graph["unit_centers"]
    unit_weight = graph["unit_weight"]
    camera_centers = graph["camera_centers"]
    names = [str(name) for name in graph["image_names"]]
    edges: list[dict[str, Any]] = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            shared = (support[first] > 0.0) & (support[second] > 0.0)
            if not np.any(shared):
                continue
            angles = triangulation_angle_degrees(
                unit_centers[shared], camera_centers[first], camera_centers[second]
            )
            triangulation = np.clip((angles - min_angle_degrees) / 10.0, 0.0, 1.0)
            weighted = unit_weight[shared] * np.minimum(support[first, shared], support[second, shared])
            quality = float(np.sum(weighted * triangulation) / max(float(np.sum(unit_weight)), 1e-12))
            if quality <= 0.0:
                continue
            edges.append(
                {
                    "first": names[first],
                    "second": names[second],
                    "shared_unit_count": int(shared.sum()),
                    "triangulation_angle_median_degrees": float(np.median(angles)),
                    "structural_edge_weight": quality,
                }
            )
    return edges


def build_static_structure_graph(
    mast3r_scene: Path,
    gate_report: Path,
    mask_pickle: Path,
    mask_dataset_path: Path,
    output: Path,
    *,
    mask_indices: Iterable[int] = (0, 1, 2, 3),
    stride: int = 8,
    bins: int = 40,
    max_units: int = 4096,
    min_unit_views: int = 2,
    block_bins: int = 4,
    min_angle_degrees: float = 1.5,
) -> dict[str, Any]:
    """Build the hard-gated static structure/view graph used by final selection."""
    mast3r_scene = Path(mast3r_scene).resolve()
    gate_report = Path(gate_report).resolve()
    charts_path = mast3r_scene / "charts_data.npz"
    images_path = mast3r_scene / "sparse" / "0" / "images.bin"
    for path in (charts_path, images_path, gate_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    if stride < 1:
        raise ValueError("stride must be positive")

    accepted = _active_gate_names(gate_report)
    charts = np.load(charts_path)
    points_all = charts["pts"].astype(np.float32, copy=False)
    depths_all = charts["depths"].astype(np.float32, copy=False)
    confs_all = charts["confs"].astype(np.float32, copy=False)
    if points_all.ndim != 4 or depths_all.ndim != 3 or confs_all.ndim != 3:
        raise RuntimeError("charts_data must contain [view,height,width] depths/confs and world point maps")
    if points_all.shape[:3] != depths_all.shape or depths_all.shape != confs_all.shape:
        raise RuntimeError("charts_data point/depth/conf dimensions disagree")
    height, width = depths_all.shape[-2:]
    scale_factor = float(charts["scale_factor"])
    colmap_images = sorted(read_images_binary(str(images_path)).values(), key=lambda image: image.id)
    if len(colmap_images) != len(points_all):
        raise RuntimeError("MASt3R camera order and charts_data count disagree")
    names_all = [Path(str(image.name)).name for image in colmap_images]
    active_indices = [index for index, name in enumerate(names_all) if name in accepted]
    if len(active_indices) < min_unit_views:
        raise RuntimeError("Too few hard-gated Charts remain for a multi-view structure graph")

    lookup = CambridgeMaskLookup(
        Path(mask_dataset_path), Path(mask_pickle), mask_indices=list(mask_indices)
    )
    sample_points: list[np.ndarray] = []
    sample_confidences: list[np.ndarray] = []
    camera_centers: list[np.ndarray] = []
    valid_fractions: list[float] = []
    names: list[str] = []
    sample_grid = np.zeros((height, width), dtype=bool)
    sample_grid[::stride, ::stride] = True
    for chart_index in active_indices:
        image = colmap_images[chart_index]
        name = names_all[chart_index]
        static_mask = lookup.get_mask(name, (height, width), torch.device("cpu")).numpy()
        valid = (
            static_mask
            & sample_grid
            & np.isfinite(points_all[chart_index]).all(axis=-1)
            & np.isfinite(depths_all[chart_index])
            & np.isfinite(confs_all[chart_index])
            & (depths_all[chart_index] > 0.0)
            & (confs_all[chart_index] > 0.0)
        )
        sample_points.append(points_all[chart_index][valid])
        sample_confidences.append(confs_all[chart_index][valid])
        rotation = qvec_to_rotmat(image.qvec)
        camera_centers.append((-rotation.T @ (image.tvec * scale_factor)).astype(np.float32))
        valid_fractions.append(float(valid.mean()))
        names.append(name)

    graph = build_structure_graph_from_samples(
        sample_points,
        sample_confidences,
        np.stack(camera_centers),
        bins=bins,
        max_units=max_units,
        min_unit_views=min_unit_views,
        block_bins=block_bins,
        valid_fractions=np.asarray(valid_fractions),
    )
    graph["image_names"] = np.asarray(names)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **graph)
    edges = _graph_edges(graph, min_angle_degrees=min_angle_degrees)
    metadata = {
        "schema_version": STRUCTURE_GRAPH_VERSION,
        "source": "hard_gated_aligned_pointmaps_static_masked",
        "mast3r_scene": str(mast3r_scene),
        "charts_data": str(charts_path),
        "gate_report": str(gate_report),
        "mask_pickle": str(Path(mask_pickle).resolve()),
        "mask_dataset_path": str(Path(mask_dataset_path).resolve()),
        "mask_indices": [int(index) for index in mask_indices],
        "fallback_from_original_sfm_tracks": True,
        "fallback_reason": "staged Cambridge posed scene has no points3D.bin; units are fixed-camera aligned point-map clusters",
        "image_count": len(names),
        "structure_unit_count": int(graph["unit_centers"].shape[0]),
        "block_count": int(graph["block_centers"].shape[0]),
        "multi_view_unit_minimum": min_unit_views,
        "sample_stride": stride,
        "voxel_bins": bins,
        "minimum_triangulation_angle_degrees": min_angle_degrees,
        "valid_fraction_by_image": {
            name: float(value) for name, value in zip(names, graph["valid_fractions"])
        },
        "view_graph_edges": edges,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"graph_path": str(output), "metadata_path": str(metadata_path), **metadata}
