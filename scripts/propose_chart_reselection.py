#!/usr/bin/env python3
"""Replace alignment-rejected charts while preserving accepted chart coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.select_chart_views import (
    load_scene_poses,
    normalize_required_name,
    select_clustered_coverage_indices,
    select_gate_replacement_reserves,
)


def build_reselection(
    selection: dict,
    gate_report: dict,
    scene_path: Path,
    rejection_neighbor_radius: int = 1,
    max_borrowed_support_distance: float | None = None,
    gate_failure_budget: int = 0,
    reserve_max_pose_distance: float | None = None,
    reserves_per_vulnerable_support: int = 1,
    joint_reserve_candidate_limit: int = 48,
) -> dict:
    if gate_failure_budget < 0:
        raise ValueError("gate_failure_budget must be non-negative")
    if reserves_per_vulnerable_support < 0:
        raise ValueError("reserves_per_vulnerable_support must be non-negative")
    if gate_failure_budget > 0 and reserve_max_pose_distance is None:
        raise ValueError(
            "reserve_max_pose_distance is required when gate_failure_budget is enabled"
        )
    image_names, poses = load_scene_poses(scene_path)
    selected_names = [normalize_required_name(name) for name in selection["image_names"]]
    rejected_names = {
        normalize_required_name(Path(str(record["image_name"])).name)
        for record in gate_report.get("records", [])
        if record.get("rejected")
    }
    historical_rejected_names = {
        normalize_required_name(name)
        for round_record in selection.get("alignment_feedback_history", [])
        for name in round_record.get("rejected_names", [])
    }
    blocked_names = rejected_names | historical_rejected_names
    accepted_names = [name for name in selected_names if name not in rejected_names]
    frame_pattern = re.compile(r"^(?P<sequence>.+)__frame(?P<frame>\d+)\.png$")
    rejected_frames: dict[str, list[int]] = {}
    for name in blocked_names:
        match = frame_pattern.match(name)
        if match:
            rejected_frames.setdefault(match.group("sequence"), []).append(
                int(match.group("frame"))
            )

    def is_rejected_neighbor(name: str) -> bool:
        if name in accepted_names:
            return False
        match = frame_pattern.match(name)
        if not match:
            return False
        frame = int(match.group("frame"))
        return any(
            abs(frame - rejected) <= rejection_neighbor_radius
            for rejected in rejected_frames.get(match.group("sequence"), [])
        )

    rejected_neighbor_names = {
        normalize_required_name(name)
        for name in selection.get("candidate_pool_names", image_names)
        if is_rejected_neighbor(normalize_required_name(name))
    }
    candidate_pool = list(dict.fromkeys([
        normalize_required_name(name)
        for name in selection.get("candidate_pool_names", image_names)
        if normalize_required_name(name) not in blocked_names
        and normalize_required_name(name) not in rejected_neighbor_names
    ]))
    if len(candidate_pool) < int(selection["n_images"]):
        raise RuntimeError(
            f"Only {len(candidate_pool)} non-rejected reserve charts remain for "
            f"n_images={selection['n_images']}"
        )

    coverage = selection.get("coverage", {})
    selected_indices, diagnostics = select_clustered_coverage_indices(
        image_names=image_names,
        candidate_image_names=candidate_pool,
        poses=poses,
        n_images=int(selection["n_images"]),
        n_view_clusters=int(coverage.get("view_clusters", 8)),
        min_views_per_cluster=int(coverage.get("min_views_per_cluster", 2)),
        min_baseline_ratio=float(coverage.get("min_baseline_ratio", 0.02)),
        min_global_pose_distance=float(coverage.get("min_global_pose_distance", 0.05)),
        max_borrowed_support_distance=(
            float(max_borrowed_support_distance)
            if max_borrowed_support_distance is not None
            else float(coverage.get("max_borrowed_support_distance", 0.25))
        ),
        required_names=accepted_names,
        coverage_objective=str(coverage.get("coverage_objective", "target_kcenter")),
    )
    next_names = [image_names[index] for index in selected_indices]
    replacements = [name for name in next_names if name not in accepted_names]
    history = list(selection.get("alignment_feedback_history", []))
    history.append(
        {
            "rejected_names": sorted(rejected_names),
            "cumulative_rejected_names": sorted(blocked_names),
            "preserved_names": accepted_names,
            "replacement_names": replacements,
            "rejected_neighbor_names": sorted(rejected_neighbor_names),
            "rejection_neighbor_radius": rejection_neighbor_radius,
            "max_borrowed_support_distance_override": max_borrowed_support_distance,
            "candidate_pool_size_after_feedback": len(candidate_pool),
        }
    )
    result = {
        **selection,
        "image_idx": selected_indices,
        "image_names": next_names,
        "run_sfm_arg": "--image_idx " + " ".join(str(index) for index in selected_indices),
        # This must be the feedback-clean pool, not the historical pool.  A
        # later reserve computation must never quietly reintroduce a chart
        # that this target-depth gate (or its local temporal neighbour) ruled
        # out.
        "candidate_pool_names": candidate_pool,
        "candidate_pool_size": len(candidate_pool),
        "coverage": diagnostics,
        "alignment_feedback_history": history,
        "alignment_feedback_converged": not rejected_names,
    }
    if gate_failure_budget > 0:
        reserves = select_gate_replacement_reserves(
            image_names=image_names,
            candidate_image_names=candidate_pool,
            poses=poses,
            selected_indices=selected_indices,
            n_view_clusters=int(diagnostics.get("view_clusters", 8)),
            post_gate_min_views_per_cluster=int(
                diagnostics.get("min_views_per_cluster", 2)
            ),
            min_baseline_ratio=float(diagnostics.get("min_baseline_ratio", 0.02)),
            max_borrowed_support_distance=(
                float(max_borrowed_support_distance)
                if max_borrowed_support_distance is not None
                else float(diagnostics.get("max_borrowed_support_distance", 0.25))
            ),
            reserve_max_pose_distance=float(reserve_max_pose_distance),
            reserves_per_vulnerable_support=reserves_per_vulnerable_support,
            gate_failure_budget=gate_failure_budget,
            observed_rejected_image_names=sorted(rejected_names),
            joint_reserve_candidate_limit=joint_reserve_candidate_limit,
        )
        result["gate_replacement_reserves"] = reserves
        result["alignment_run_sfm_arg"] = "--image_idx " + " ".join(
            str(index) for index in reserves["alignment_image_idx"]
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--scene-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejection-neighbor-radius", type=int, default=1)
    parser.add_argument("--max-borrowed-support-distance", type=float)
    parser.add_argument(
        "--gate-failure-budget",
        type=int,
        default=0,
        help=(
            "If positive, append a jointly certified reserve set that survives this "
            "many correlated target-depth gate failures per pose cluster."
        ),
    )
    parser.add_argument("--reserve-max-pose-distance", type=float)
    parser.add_argument("--reserves-per-vulnerable-support", type=int, default=1)
    parser.add_argument("--joint-reserve-candidate-limit", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    gate_report = json.loads(args.gate_report.read_text())
    scene_path = args.scene_path or Path(selection["scene_path"])
    result = build_reselection(
        selection,
        gate_report,
        scene_path,
        rejection_neighbor_radius=args.rejection_neighbor_radius,
        max_borrowed_support_distance=args.max_borrowed_support_distance,
        gate_failure_budget=args.gate_failure_budget,
        reserve_max_pose_distance=args.reserve_max_pose_distance,
        reserves_per_vulnerable_support=args.reserves_per_vulnerable_support,
        joint_reserve_candidate_limit=args.joint_reserve_candidate_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    latest = result["alignment_feedback_history"][-1]
    print(json.dumps({
        "rejected": latest["rejected_names"],
        "replacements": latest["replacement_names"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
