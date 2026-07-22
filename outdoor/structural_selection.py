"""Final Chart selection driven by multi-view static structure support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from outdoor.structure_graph import triangulation_angle_degrees
from scripts.select_chart_views import build_pose_geometry, load_scene_poses


STRUCTURAL_SELECTION_VERSION = "outdoor-structural-chart-selection-v1"


def _rank01(values: np.ndarray, *, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, 0.5, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return result
    subset = values[finite]
    if np.ptp(subset) <= 1e-12:
        return result
    order = np.argsort(subset, kind="stable")
    ranks = np.empty(len(subset), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(subset))
    if not higher_is_better:
        ranks = 1.0 - ranks
    result[finite] = ranks
    return result


def _gate_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = {
        Path(str(record["image_name"])).name: record
        for record in payload.get("records", [])
    }
    if not records:
        raise RuntimeError(f"Gate report has no chart records: {path}")
    return records


def _gate_reliability(records: dict[str, dict[str, Any]], names: list[str]) -> np.ndarray:
    valid = np.asarray(
        [
            float(records[name].get("valid_fraction", records[name].get("aligned_valid_fraction", 0.0)))
            for name in names
        ],
        dtype=np.float64,
    )
    relative_p90 = np.asarray(
        [
            float(records[name].get("relative_p90", records[name].get("relative_depth_p90", 0.0)))
            for name in names
        ],
        dtype=np.float64,
    )
    gt25 = np.asarray(
        [float(records[name].get("gt25_fraction", 0.0)) for name in names], dtype=np.float64
    )
    return 0.60 * _rank01(valid) + 0.25 * _rank01(relative_p90, higher_is_better=False) + 0.15 * _rank01(gt25, higher_is_better=False)


def _connected(selected: list[int], pair_score: np.ndarray, threshold: float) -> bool:
    if len(selected) <= 1:
        return True
    visited = {selected[0]}
    frontier = [selected[0]]
    selected_set = set(selected)
    while frontier:
        current = frontier.pop()
        neighbours = [
            candidate
            for candidate in selected_set - visited
            if pair_score[current, candidate] >= threshold
        ]
        visited.update(neighbours)
        frontier.extend(neighbours)
    return visited == selected_set


def _pair_unit_quality(
    centers: np.ndarray,
    unit_centers: np.ndarray,
    support: np.ndarray,
    *,
    min_triangulation_angle_degrees: float,
    target_triangulation_angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return structural pair support and weighted pair quality.

    Unlike legacy global-baseline checks, every pair is evaluated at the actual
    static structure units it jointly observes.
    """
    view_count, unit_count = support.shape
    quality = np.zeros((view_count, view_count, unit_count), dtype=np.float32)
    for first in range(view_count):
        for second in range(first + 1, view_count):
            shared = np.minimum(support[first], support[second])
            if not np.any(shared > 0.0):
                continue
            angle = triangulation_angle_degrees(unit_centers, centers[first], centers[second])
            angle_quality = np.clip(
                (angle - min_triangulation_angle_degrees)
                / max(target_triangulation_angle_degrees - min_triangulation_angle_degrees, 1e-6),
                0.0,
                1.0,
            )
            value = shared * angle_quality
            quality[first, second] = value
            quality[second, first] = value
    # Returning this independently keeps the caller's threshold/data model
    # visible in diagnostics and makes connectivity graph construction cheap.
    pair_score = quality
    return quality, pair_score


def _pair_summary(pair_unit: np.ndarray, unit_weight: np.ndarray) -> np.ndarray:
    return np.tensordot(pair_unit, unit_weight, axes=([2], [0])).astype(np.float32)


def _pose_features(target_scene_path: Path, names: list[str]) -> np.ndarray:
    all_names, poses = load_scene_poses(target_scene_path)
    features, directions, _ = build_pose_geometry(all_names, poses)
    combined = np.concatenate([features, 0.35 * directions], axis=1)
    by_name = {name: index for index, name in enumerate(all_names)}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise RuntimeError(f"Structural Chart is absent from the target camera contract: {missing[:5]}")
    return combined[np.asarray([by_name[name] for name in names], dtype=np.int64)]


def _diagnostics(
    selected: list[int],
    *,
    support: np.ndarray,
    unit_weight: np.ndarray,
    pair_unit: np.ndarray,
    block_ids: np.ndarray,
    block_weight: np.ndarray,
    min_block_views: int,
    pair_score: np.ndarray,
    edge_threshold: float,
    support_cap: int,
) -> dict[str, Any]:
    unit_count = np.zeros(len(unit_weight), dtype=np.int16)
    for index in selected:
        unit_count += (support[index] > 0.0).astype(np.int16)
    coverage = float(np.sum(unit_weight * np.minimum(unit_count / max(support_cap, 1), 1.0)))
    if len(selected) < 2:
        pair_best = np.zeros(len(unit_weight), dtype=np.float32)
    else:
        pair_best = np.max(pair_unit[np.ix_(selected, selected)], axis=(0, 1))
    pair_coverage = float(np.sum(unit_weight * (pair_best > 0.0)))
    blocks = []
    for block in range(len(block_weight)):
        unit_member = block_ids == block
        available = int(np.sum(np.any(support[:, unit_member] > 0.0, axis=1)))
        selected_support = int(
            np.sum(np.any(support[np.asarray(selected)][:, unit_member] > 0.0, axis=1))
        ) if selected else 0
        required = min(min_block_views, available)
        blocks.append(
            {
                "block": block,
                "weight": float(block_weight[block]),
                "available_views": available,
                "selected_views": selected_support,
                "required_views": required,
                "passed": selected_support >= required,
            }
        )
    return {
        "structural_coverage": coverage,
        "triangulated_multiview_coverage": pair_coverage,
        "connected": _connected(selected, pair_score, edge_threshold),
        "block_diagnostics": blocks,
        "all_blocks_supported": all(record["passed"] for record in blocks),
        "selected_count": len(selected),
        "support_cap": support_cap,
    }


def select_structural_charts(
    graph_path: Path,
    gate_report: Path,
    target_scene_path: Path,
    output: Path,
    *,
    base_selection: Path | None = None,
    min_charts: int = 24,
    max_charts: int | None = None,
    target_structural_coverage: float = 0.92,
    target_multiview_coverage: float = 0.70,
    min_block_views: int = 2,
    min_triangulation_angle_degrees: float = 1.5,
    target_triangulation_angle_degrees: float = 8.0,
    minimum_marginal_gain: float = 0.001,
) -> dict[str, Any]:
    """Select a connected Global+Local Chart set until structure is covered."""
    if not 0.0 < target_structural_coverage <= 1.0:
        raise ValueError("target_structural_coverage must be in (0, 1]")
    if not 0.0 < target_multiview_coverage <= 1.0:
        raise ValueError("target_multiview_coverage must be in (0, 1]")
    graph = np.load(Path(graph_path), allow_pickle=False)
    names = [str(name) for name in graph["image_names"]]
    support = graph["unit_support"].astype(np.float32, copy=False)
    unit_weight = graph["unit_weight"].astype(np.float32, copy=False)
    unit_centers = graph["unit_centers"].astype(np.float32, copy=False)
    camera_centers = graph["camera_centers"].astype(np.float32, copy=False)
    block_ids = graph["unit_block_ids"].astype(np.int32, copy=False)
    block_weight = graph["block_weight"].astype(np.float32, copy=False)
    valid_fraction = graph["valid_fractions"].astype(np.float32, copy=False)
    if support.shape != (len(names), len(unit_weight)):
        raise RuntimeError("Structure graph support matrix has incompatible dimensions")
    if min_charts < 1:
        raise ValueError("min_charts must be positive")
    max_charts = len(names) if max_charts is None else int(max_charts)
    if min_charts > max_charts or max_charts > len(names):
        raise ValueError("Chart count bounds are incompatible with the hard-gated structure graph")

    gate = _gate_records(gate_report)
    missing_gate = sorted(set(names) - set(gate))
    rejected = sorted(name for name in names if gate[name].get("rejected") is True)
    if missing_gate or rejected:
        raise RuntimeError(
            "Structure graph must contain only gate-accepted Charts: "
            f"missing_gate={missing_gate[:3]}, rejected={rejected[:3]}"
        )

    graph_reliability = np.sum(support * unit_weight[None], axis=1)
    reliabilities = (
        0.50 * _rank01(graph_reliability)
        + 0.25 * _rank01(valid_fraction)
        + 0.25 * _gate_reliability(gate, names)
    )
    pair_unit, _ = _pair_unit_quality(
        camera_centers,
        unit_centers,
        support,
        min_triangulation_angle_degrees=min_triangulation_angle_degrees,
        target_triangulation_angle_degrees=target_triangulation_angle_degrees,
    )
    pair_score = _pair_summary(pair_unit, unit_weight)
    nonzero_edges = pair_score[np.triu_indices(len(names), k=1)]
    edge_threshold = max(float(np.quantile(nonzero_edges[nonzero_edges > 0.0], 0.10)) * 0.2, 1e-6) if np.any(nonzero_edges > 0.0) else 1e-6
    pose = _pose_features(Path(target_scene_path), names)
    pose_normalizer = max(float(np.quantile(np.linalg.norm(pose[:, None] - pose[None, :], axis=2), 0.90)), 1e-6)

    selected: list[int] = []
    unit_count = np.zeros(len(unit_weight), dtype=np.int16)
    pair_best = np.zeros(len(unit_weight), dtype=np.float32)
    block_count = np.zeros(len(block_weight), dtype=np.int16)
    support_cap = 3
    trace: list[dict[str, Any]] = []

    def candidate_score(candidate: int) -> tuple[float, dict[str, float]]:
        candidate_units = support[candidate] > 0.0
        coverage_gain = float(
            np.sum(
                unit_weight
                * (
                    np.minimum((unit_count + candidate_units) / support_cap, 1.0)
                    - np.minimum(unit_count / support_cap, 1.0)
                )
            )
        )
        if selected:
            candidate_pair = np.max(pair_unit[candidate, np.asarray(selected)], axis=0)
            pair_gain = float(np.sum(unit_weight * np.maximum(candidate_pair - pair_best, 0.0)))
            nearest = np.min(np.linalg.norm(pose[candidate] - pose[np.asarray(selected)], axis=1))
            pose_bonus = min(nearest / pose_normalizer, 1.0)
        else:
            pair_gain = 0.0
            pose_bonus = 1.0
        block_gain = 0.0
        for block in range(len(block_weight)):
            if block_count[block] >= min_block_views or not np.any(candidate_units & (block_ids == block)):
                continue
            block_gain += float(block_weight[block])
        score = (
            0.52 * coverage_gain
            + 0.30 * pair_gain
            + 0.10 * block_gain
            + 0.06 * float(reliabilities[candidate])
            + 0.02 * pose_bonus
        )
        return score, {
            "coverage_gain": coverage_gain,
            "pair_gain": pair_gain,
            "block_gain": block_gain,
            "reliability": float(reliabilities[candidate]),
            "pose_bonus": float(pose_bonus),
        }

    while len(selected) < max_charts:
        candidates = [index for index in range(len(names)) if index not in selected]
        if selected:
            connected_candidates = [
                candidate
                for candidate in candidates
                if float(np.max(pair_score[candidate, np.asarray(selected)])) >= edge_threshold
            ]
            # A hard graph disconnect is evidence of a broken alignment or
            # missing structural bridge, not a reason to silently select an
            # isolated camera merely to satisfy a fixed count.
            if not connected_candidates:
                raise RuntimeError(
                    "Structural view graph disconnected before coverage targets were met; "
                    "replace/re-align the broad Chart pool."
                )
            candidates = connected_candidates
        scores = {candidate: candidate_score(candidate) for candidate in candidates}
        chosen = max(candidates, key=lambda item: (scores[item][0], -item))
        score, parts = scores[chosen]
        selected.append(chosen)
        candidate_units = support[chosen] > 0.0
        unit_count += candidate_units.astype(np.int16)
        if len(selected) > 1:
            pair_best = np.maximum(pair_best, np.max(pair_unit[chosen, np.asarray(selected[:-1])], axis=0))
        for block in range(len(block_weight)):
            if np.any(candidate_units & (block_ids == block)):
                block_count[block] += 1
        current = _diagnostics(
            selected,
            support=support,
            unit_weight=unit_weight,
            pair_unit=pair_unit,
            block_ids=block_ids,
            block_weight=block_weight,
            min_block_views=min_block_views,
            pair_score=pair_score,
            edge_threshold=edge_threshold,
            support_cap=support_cap,
        )
        trace.append({"image_name": names[chosen], "score": score, **parts, **current})
        enough = (
            len(selected) >= min_charts
            and current["structural_coverage"] >= target_structural_coverage
            and current["triangulated_multiview_coverage"] >= target_multiview_coverage
            and current["all_blocks_supported"]
            and current["connected"]
        )
        if enough:
            # Count is a scene output.  Once coverage is met, only keep
            # adding cameras while they make material structural progress.
            if len(selected) >= max_charts or parts["coverage_gain"] + parts["pair_gain"] < minimum_marginal_gain:
                break
            # The first time targets are met is the default stop.  Continuing
            # would recreate a hidden fixed-N policy.
            break

    diagnostics = _diagnostics(
        selected,
        support=support,
        unit_weight=unit_weight,
        pair_unit=pair_unit,
        block_ids=block_ids,
        block_weight=block_weight,
        min_block_views=min_block_views,
        pair_score=pair_score,
        edge_threshold=edge_threshold,
        support_cap=support_cap,
    )
    if not (
        len(selected) >= min_charts
        and diagnostics["structural_coverage"] >= target_structural_coverage
        and diagnostics["triangulated_multiview_coverage"] >= target_multiview_coverage
        and diagnostics["all_blocks_supported"]
        and diagnostics["connected"]
    ):
        raise RuntimeError(
            "Could not satisfy structural Chart selection targets within the hard-gated pool: "
            + json.dumps(diagnostics, sort_keys=True)
        )

    all_target_names, _ = load_scene_poses(Path(target_scene_path))
    target_index = {name: index for index, name in enumerate(all_target_names)}
    selected_names = [names[index] for index in selected]
    selected_indices = [target_index[name] for name in selected_names]
    base_payload: dict[str, Any] = {}
    if base_selection is not None:
        base_payload = json.loads(Path(base_selection).read_text(encoding="utf-8"))
    payload = {
        **base_payload,
        "schema_version": STRUCTURAL_SELECTION_VERSION,
        "scene_path": str(Path(target_scene_path).resolve()),
        "structural_graph": str(Path(graph_path).resolve()),
        "gate_report": str(Path(gate_report).resolve()),
        "n_images": len(selected_names),
        "requested_n_images": None,
        "actual_n_images": len(selected_names),
        "image_names": selected_names,
        "selected_image_names": selected_names,
        "image_idx": selected_indices,
        "image_idx_coordinate_system": "full_target_scene_order",
        "audit_active_from_selection": True,
        "structural_selection": {
            "version": STRUCTURAL_SELECTION_VERSION,
            "selection_unit": "static_structure_unit",
            "support_source": "hard_gated_aligned_pointmaps_static_masked",
            "minimum_triangulation_angle_degrees": min_triangulation_angle_degrees,
            "target_triangulation_angle_degrees": target_triangulation_angle_degrees,
            "target_structural_coverage": target_structural_coverage,
            "target_multiview_coverage": target_multiview_coverage,
            "min_block_views": min_block_views,
            "edge_threshold": edge_threshold,
            "reliability": {name: float(reliabilities[index]) for index, name in enumerate(names)},
            "selection_trace": trace,
            "diagnostics": diagnostics,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def validate_structural_selection(selection_path: Path, gate_report: Path) -> dict[str, Any]:
    """Check the active selection independently before it can mutate charts_data."""
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if selection.get("schema_version") != STRUCTURAL_SELECTION_VERSION:
        raise RuntimeError("Selection is not an outdoor structural selection")
    names = [str(name) for name in selection.get("image_names", [])]
    indices = selection.get("image_idx", [])
    if not names or len(names) != len(indices) or len(names) != len(set(names)):
        raise RuntimeError("Structural selection has invalid name/index entries")
    gate = _gate_records(gate_report)
    rejected = [name for name in names if name not in gate or gate[name].get("rejected") is True]
    if rejected:
        raise RuntimeError(f"Structural selection reactivates invalid Chart(s): {rejected[:5]}")
    diagnostics = selection.get("structural_selection", {}).get("diagnostics", {})
    expected = ["structural_coverage", "triangulated_multiview_coverage", "connected", "all_blocks_supported"]
    if any(key not in diagnostics for key in expected):
        raise RuntimeError("Structural selection lacks final structural diagnostics")
    if diagnostics["connected"] is not True or diagnostics["all_blocks_supported"] is not True:
        raise RuntimeError("Structural selection failed connectivity/block constraints")
    return {
        "selected_count": len(names),
        "structural_coverage": diagnostics["structural_coverage"],
        "triangulated_multiview_coverage": diagnostics["triangulated_multiview_coverage"],
        "passed": True,
    }
