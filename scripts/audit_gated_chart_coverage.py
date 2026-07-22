#!/usr/bin/env python3
"""Audit coverage after aligned-chart rejection, including retained anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.select_chart_views import (  # noqa: E402
    _farthest_point_indices,
    _target_coverage_diagnostics,
    build_pose_geometry,
    camera_center_and_direction,
    load_scene_poses,
    normalize_required_name,
)


def archived_reference_features(
    cameras_json: Path,
    scene_names: list[str],
    scene_poses: dict,
) -> dict[str, np.ndarray]:
    payload = json.loads(Path(cameras_json).read_text())
    centers = np.stack(
        [camera_center_and_direction(*scene_poses[name])[0] for name in scene_names]
    )
    center = centers.mean(axis=0)
    extent = max(float(np.linalg.norm(centers.max(0) - centers.min(0))), 1e-12)
    output = {}
    for path, c2w in zip(payload["filepaths"], payload["cams2world"]):
        matrix = np.asarray(c2w, dtype=np.float64)
        direction = matrix[:3, 2]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        output[Path(path).name] = np.concatenate(
            [(matrix[:3, 3] - center) / extent, 0.35 * direction]
        )
    return output


def audit_coverage(
    selection: dict,
    gate_report: dict,
    scene_path: Path,
    *,
    minimum_active: int = 24,
    max_target_pose_distance: float = 0.35,
    max_sequence_p95: float = 0.30,
    reference_features: dict[str, np.ndarray] | None = None,
    max_reference_pose_distance: float = 0.30,
    require_exact_references: bool = False,
    min_reference_support: int | None = None,
) -> dict:
    image_names, poses = load_scene_poses(scene_path)
    normalized_centers, directions, _ = build_pose_geometry(image_names, poses)
    features = np.concatenate([normalized_centers, 0.35 * directions], axis=1)
    name_to_index = {name: index for index, name in enumerate(image_names)}
    gated_active_names = [
        normalize_required_name(Path(str(record["image_name"])).name)
        for record in gate_report.get("records", [])
        if not record.get("rejected")
    ]
    # A quality-aware selection is deliberately a *subset* of charts that
    # passed the hard geometry gate.  Historically this audit always treated
    # every non-rejected chart as active, which made an N=24/N=28 ablation
    # appear to pass even when its selected charts no longer covered a pose
    # cluster.  Keep the legacy behavior unless the selection opts in so old
    # audit artifacts remain comparable.
    if selection.get("audit_active_from_selection", False):
        selected_names = {
            normalize_required_name(Path(str(name)).name)
            for name in selection.get(
                "selected_image_names", selection.get("image_names", [])
            )
        }
        missing_from_gate = sorted(selected_names - set(gated_active_names))
        if missing_from_gate:
            preview = ", ".join(missing_from_gate[:5])
            raise ValueError(
                "Quality-aware selection contains chart(s) absent from the "
                f"accepted hard-gate set: {preview}"
            )
        active_names = [name for name in gated_active_names if name in selected_names]
    else:
        active_names = gated_active_names
    active_indices = [name_to_index[name] for name in active_names]
    coverage_config = selection.get("coverage", {})
    n_clusters = int(coverage_config.get("view_clusters", 8))
    min_support = int(coverage_config.get("min_views_per_cluster", 2))
    min_baseline = float(coverage_config.get("min_baseline_ratio", 0.02))
    max_borrowed = float(coverage_config.get("max_borrowed_support_distance", 0.32))
    anchor_indices = _farthest_point_indices(features, n_clusters)
    anchor_features = features[np.asarray(anchor_indices)]
    labels = np.argmin(
        np.linalg.norm(features[:, None] - anchor_features[None], axis=2), axis=1
    )
    clusters = []
    for cluster in range(n_clusters):
        support = [
            index for index in active_indices
            if labels[index] == cluster
            or float(np.linalg.norm(features[index] - anchor_features[cluster]))
            <= max_borrowed
        ]
        baseline_ok = any(
            float(np.linalg.norm(normalized_centers[first] - normalized_centers[second]))
            >= min_baseline
            for offset, first in enumerate(support)
            for second in support[offset + 1 :]
        )
        clusters.append(
            {
                "cluster": cluster,
                "support_count": len(support),
                "support_names": [image_names[index] for index in support],
                "baseline_satisfied": baseline_ok,
                "passed": len(support) >= min_support and baseline_ok,
            }
        )
    target = _target_coverage_diagnostics(image_names, features, active_indices)
    sequence_pass = all(
        item["p95"] <= max_sequence_p95 for item in target["per_sequence"].values()
    )
    # Global target-pose coverage may borrow a Chart from another traversal.
    # That is useful for appearance but does not prove that a temporal corridor
    # has an independently-baselined geometry pair.  Mirror the selector's
    # explicit same-sequence contract after the hard depth gate.
    sequence_mode = str(coverage_config.get("sequence_coverage_mode", "off"))
    pre_gate_min_sequence_support = int(
        coverage_config.get("min_views_per_sequence", 0)
    )
    min_sequence_support = int(
        coverage_config.get(
            "post_gate_min_views_per_sequence",
            pre_gate_min_sequence_support,
        )
    )
    sequence_gate_failure_budget = int(
        coverage_config.get("sequence_gate_failure_budget", 0)
    )
    sequence_support = []
    if sequence_mode not in {"off", "soft", "strict"}:
        raise ValueError(f"Unknown sequence coverage mode in selection: {sequence_mode}")
    if pre_gate_min_sequence_support < 0 or min_sequence_support < 0:
        raise ValueError("Selection has a negative same-sequence support count")
    if sequence_gate_failure_budget < 0:
        raise ValueError("Selection has a negative same-sequence gate-failure budget")
    if sequence_mode != "off" and pre_gate_min_sequence_support > 0:
        target_by_sequence: dict[str, list[int]] = {}
        for index, name in enumerate(image_names):
            target_by_sequence.setdefault(name.split("__", 1)[0], []).append(index)
        for sequence, targets in sorted(target_by_sequence.items()):
            # A shorter capture could not satisfy the pre-gate reserve
            # contract in the first place, so keep the selector's explicit
            # exception rather than silently turning it into a post-gate
            # failure.  For every applicable temporal corridor, however, the
            # surviving requirement is the post-gate count, not the broader
            # primary-reserve count.
            applicable = len(targets) >= pre_gate_min_sequence_support
            supports = [
                index
                for index in active_indices
                if image_names[index].split("__", 1)[0] == sequence
            ]
            baseline_ok = any(
                float(np.linalg.norm(normalized_centers[first] - normalized_centers[second]))
                >= min_baseline
                for offset, first in enumerate(supports)
                for second in supports[offset + 1 :]
            )
            sequence_support.append(
                {
                    "sequence": sequence,
                    "target_count": len(targets),
                    "applicable": applicable,
                    "required_count": min_sequence_support if applicable else 0,
                    "active_count": len(supports),
                    "active_names": [image_names[index] for index in supports],
                    "baseline_satisfied": baseline_ok if applicable and min_sequence_support >= 2 else True,
                    "passed": (
                        not applicable
                        or (
                            len(supports) >= min_sequence_support
                            and (min_sequence_support < 2 or baseline_ok)
                        )
                    ),
                }
            )
    strict_sequence_support_pass = all(
        record["passed"] for record in sequence_support if record["applicable"]
    )
    quality_constraints = selection.get("quality_aware_selection", {}).get("constraints", {})
    configured_reference_support = int(
        min_reference_support
        if min_reference_support is not None
        else quality_constraints.get("min_reference_support_requested", 1)
    )
    if configured_reference_support < 1:
        raise ValueError("min_reference_support must be positive")
    reference_records = []
    if reference_features:
        active_features = features[np.asarray(active_indices)]
        gated_indices = [name_to_index[name] for name in gated_active_names]
        gated_features = features[np.asarray(gated_indices)]
        for name, feature in reference_features.items():
            reference_sequence = name.split("__", 1)[0]
            same_sequence_offsets = [
                offset for offset, active_name in enumerate(active_names)
                if active_name.split("__", 1)[0] == reference_sequence
            ]
            search_offsets = same_sequence_offsets or list(range(len(active_names)))
            distances = np.linalg.norm(active_features[search_offsets] - feature[None], axis=1)
            local_offset = int(np.argmin(distances))
            offset = search_offsets[local_offset]
            support_offsets = [
                candidate_offset
                for candidate_offset in same_sequence_offsets
                if float(np.linalg.norm(active_features[candidate_offset] - feature))
                <= max_reference_pose_distance
            ]
            hard_gated_offsets = [
                candidate_offset
                for candidate_offset, candidate_name in enumerate(gated_active_names)
                if candidate_name.split("__", 1)[0] == reference_sequence
            ]
            hard_gated_distances = np.asarray(
                [
                    float(np.linalg.norm(gated_features[candidate_offset] - feature))
                    for candidate_offset in hard_gated_offsets
                ],
                dtype=np.float64,
            )
            eligible_hard_gated_offsets = [
                candidate_offset
                for candidate_offset, distance in zip(hard_gated_offsets, hard_gated_distances)
                if distance <= max_reference_pose_distance
            ]
            available_support_count = len(eligible_hard_gated_offsets)
            required_support_count = min(
                configured_reference_support,
                available_support_count,
            )
            exact_retained = name in active_names
            reference_records.append(
                {
                    "reference_name": name,
                    "nearest_active_name": active_names[offset],
                    "pose_distance": float(distances[local_offset]),
                    "same_sequence": bool(same_sequence_offsets),
                    "available_hard_gated_support_count": available_support_count,
                    "required_support_count": required_support_count,
                    "selected_support_count": len(support_offsets),
                    "supporting_active_names": [active_names[item] for item in support_offsets],
                    "supporting_pose_distances": [
                        float(np.linalg.norm(active_features[item] - feature))
                        for item in support_offsets
                    ],
                    "exact_retained": exact_retained,
                    "passed": (
                        bool(same_sequence_offsets)
                        and available_support_count > 0
                        and len(support_offsets) >= required_support_count
                        and (exact_retained or not require_exact_references)
                    ),
                }
            )
    checks = {
        "minimum_active": len(active_names) >= minimum_active,
        "no_active_depth_conflict": not any(
            record.get("depth_conflict") and not record.get("rejected")
            for record in gate_report.get("records", [])
        ),
        "all_clusters_supported": all(item["passed"] for item in clusters),
        "target_max_pose_distance": target["all_targets"]["max"] <= max_target_pose_distance,
        "sequence_p95": sequence_pass,
        "same_sequence_static_support": (
            strict_sequence_support_pass if sequence_mode == "strict" else True
        ),
        "retained_reference_coverage": all(item["passed"] for item in reference_records),
        "retained_reference_same_sequence": all(
            item["same_sequence"] for item in reference_records
        ),
        "retained_reference_min_support": all(
            item["selected_support_count"] >= item["required_support_count"]
            and item["available_hard_gated_support_count"] > 0
            for item in reference_records
        ),
        "exact_reference_retention": (
            all(item["exact_retained"] for item in reference_records)
            if require_exact_references else True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_count": len(active_names),
        "active_names": active_names,
        "thresholds": {
            "minimum_active": minimum_active,
            "max_target_pose_distance": max_target_pose_distance,
            "max_sequence_p95": max_sequence_p95,
            "max_reference_pose_distance": max_reference_pose_distance,
            "require_exact_references": require_exact_references,
            "min_reference_support_requested": configured_reference_support,
            "sequence_coverage_mode": sequence_mode,
            "pre_gate_min_views_per_sequence": pre_gate_min_sequence_support,
            "min_views_per_sequence": min_sequence_support,
            "sequence_gate_failure_budget": sequence_gate_failure_budget,
        },
        "clusters": clusters,
        "target_coverage": target,
        "same_sequence_support": sequence_support,
        "reference_coverage": reference_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path)
    parser.add_argument("--reference-cameras-json", type=Path)
    parser.add_argument("--minimum-active", type=int, default=24)
    parser.add_argument("--max-target-pose-distance", type=float, default=0.35)
    parser.add_argument("--max-sequence-p95", type=float, default=0.30)
    parser.add_argument("--max-reference-pose-distance", type=float, default=0.30)
    parser.add_argument("--require-exact-references", action="store_true")
    parser.add_argument(
        "--min-reference-support",
        type=int,
        help=(
            "Override the selection's retained-anchor support requirement. "
            "The effective requirement is capped by available hard-gated views."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    gate = json.loads(args.gate_report.read_text())
    scene_path = args.scene_path or Path(selection["scene_path"])
    names, poses = load_scene_poses(scene_path)
    references = (
        archived_reference_features(args.reference_cameras_json, names, poses)
        if args.reference_cameras_json else None
    )
    result = audit_coverage(
        selection,
        gate,
        scene_path,
        minimum_active=args.minimum_active,
        max_target_pose_distance=args.max_target_pose_distance,
        max_sequence_p95=args.max_sequence_p95,
        reference_features=references,
        max_reference_pose_distance=args.max_reference_pose_distance,
        require_exact_references=args.require_exact_references,
        min_reference_support=args.min_reference_support,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
