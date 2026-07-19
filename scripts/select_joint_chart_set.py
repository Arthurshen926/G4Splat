#!/usr/bin/env python3
"""Jointly reselect a redundant chart set from cumulative alignment feedback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.select_chart_views import (  # noqa: E402
    build_pose_geometry,
    camera_center_and_direction,
    load_scene_poses,
    normalize_required_name,
    select_clustered_coverage_indices,
)
import numpy as np


FRAME_PATTERN = re.compile(r"^(?P<sequence>.+)__frame(?P<frame>\d+)\.[^.]+$")


def rejected_names_from_reports(reports: list[dict]) -> set[str]:
    return {
        normalize_required_name(Path(str(record["image_name"])).name)
        for report in reports
        for record in report.get("records", [])
        if record.get("rejected")
    }


def temporal_neighbors(
    names: list[str], rejected_names: set[str], radius: int
) -> set[str]:
    rejected_frames: dict[str, list[int]] = {}
    for name in rejected_names:
        match = FRAME_PATTERN.match(name)
        if match:
            rejected_frames.setdefault(match.group("sequence"), []).append(
                int(match.group("frame"))
            )
    blocked = set(rejected_names)
    for raw_name in names:
        name = normalize_required_name(raw_name)
        match = FRAME_PATTERN.match(name)
        if not match:
            continue
        frame = int(match.group("frame"))
        if any(
            abs(frame - rejected) <= radius
            for rejected in rejected_frames.get(match.group("sequence"), [])
        ):
            blocked.add(name)
    return blocked


def correlated_failure_corridors(
    names: list[str],
    rejected_names: set[str],
    *,
    min_run_length: int = 3,
    padding: int = 2,
    max_bridge_gap: int = 2,
) -> list[dict]:
    """Return locally correlated failure corridors supported by repeated gates.

    A single failed Chart is direct evidence only against that image.  Once a
    *contiguous* run of independently gated failures is observed, however,
    treating the intervening trajectory as independent is no longer justified:
    it is a local geometry/occlusion failure mode.  A newly failed frame can
    also bridge a short pre-existing quarantine gap; such gaps are precisely
    the frames that had been held out for re-audit.  We then temporarily keep
    a small padded corridor out of the next joint candidate set.  The caller
    records the exact evidence and can still disable this escalation.
    """
    if min_run_length <= 0:
        raise ValueError("min_run_length must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if max_bridge_gap < 0:
        raise ValueError("max_bridge_gap must be non-negative")

    normalized_names = [normalize_required_name(name) for name in names]
    rejected_by_sequence: dict[str, list[int]] = {}
    for name in rejected_names:
        match = FRAME_PATTERN.match(normalize_required_name(name))
        if match is not None:
            rejected_by_sequence.setdefault(match.group("sequence"), []).append(
                int(match.group("frame"))
            )

    parsed_names: dict[str, list[tuple[int, str]]] = {}
    for name in normalized_names:
        match = FRAME_PATTERN.match(name)
        if match is not None:
            parsed_names.setdefault(match.group("sequence"), []).append(
                (int(match.group("frame")), name)
            )

    corridors: list[dict] = []
    for sequence, frames in sorted(rejected_by_sequence.items()):
        ordered = sorted(set(frames))
        if not ordered:
            continue
        groups: list[list[int]] = [[ordered[0]]]
        for frame in ordered[1:]:
            # A gap of two untested frames joins the failure components only
            # after both endpoints have independently failed.  This is the
            # same neighbourhood that was quarantined for re-audit, rather
            # than a blanket temporal-neighbour rejection.
            if frame - groups[-1][-1] <= int(max_bridge_gap) + 1:
                groups[-1].append(frame)
            else:
                groups.append([frame])
        for rejected_run in groups:
            if len(rejected_run) < min_run_length:
                continue
            lower = max(0, int(rejected_run[0]) - int(padding))
            upper = int(rejected_run[-1]) + int(padding)
            blocked_names = sorted(
                name
                for candidate_frame, name in parsed_names.get(sequence, [])
                if lower <= candidate_frame <= upper
            )
            corridors.append(
                {
                    "sequence": sequence,
                    "rejected_frames": rejected_run,
                    "frame_start": int(rejected_run[0]),
                    "frame_end": int(rejected_run[-1]),
                    "padding": int(padding),
                    "max_bridge_gap": int(max_bridge_gap),
                    "blocked_names": blocked_names,
                }
            )
    return corridors


def build_joint_selection(
    selection: dict,
    gate_reports: list[dict],
    scene_path: Path,
    *,
    n_images: int,
    rejection_neighbor_radius: int = 2,
    max_borrowed_support_distance: float | None = None,
    explicit_blocked_names: list[str] | None = None,
    reference_names: list[str] | None = None,
    reference_features: dict[str, np.ndarray] | None = None,
    reference_max_pose_distance: float = 0.25,
    min_global_pose_distance_override: float | None = None,
    preserve_reference_names: bool = False,
    hard_block_rejection_neighbors: bool = False,
    hard_block_correlated_failure_corridors: bool = True,
    correlated_failure_min_run_length: int = 3,
    correlated_failure_padding: int = 2,
    correlated_failure_max_bridge_gap: int = 2,
) -> dict:
    """Re-optimize the complete set; no previous accepted chart is pinned.

    A failed Chart is direct evidence against that Chart, not automatically
    against every temporally adjacent image.  Adjacent frames are recorded as
    a quarantine/risk set and re-audited by the next joint alignment.  They
    are only hard-blocked when the caller explicitly asks for the legacy
    conservative policy.  This avoids turning a one-frame depth conflict into
    an artificial coverage hole when the neighbouring frame has independent
    QC and semantic evidence.
    """
    image_names, poses = load_scene_poses(scene_path)
    cumulative_rejected = rejected_names_from_reports(gate_reports)
    cumulative_rejected.update(
        normalize_required_name(name)
        for item in selection.get("alignment_feedback_history", [])
        for name in item.get("rejected_names", [])
    )
    explicit_blocked = {
        normalize_required_name(name) for name in (explicit_blocked_names or [])
    }
    base_pool = [
        normalize_required_name(name)
        for name in selection.get("candidate_pool_names", image_names)
    ]
    direct_blocked = cumulative_rejected | explicit_blocked
    # Build the evidence record against the full audited train trajectory, not
    # merely the already-shrunk previous candidate pool.  Otherwise a later
    # re-selection would hide the early members of a corridor that were
    # excluded in an earlier round, weakening auditability.
    corridors = correlated_failure_corridors(
        image_names,
        cumulative_rejected,
        min_run_length=int(correlated_failure_min_run_length),
        padding=int(correlated_failure_padding),
        max_bridge_gap=int(correlated_failure_max_bridge_gap),
    )
    corridor_blocked = {
        name for corridor in corridors for name in corridor["blocked_names"]
    }
    neighbour_quarantine = temporal_neighbors(
        base_pool, direct_blocked, int(rejection_neighbor_radius)
    ) - direct_blocked
    blocked = direct_blocked | (
        corridor_blocked if hard_block_correlated_failure_corridors else set()
    ) | (neighbour_quarantine if hard_block_rejection_neighbors else set())
    candidate_pool = [name for name in base_pool if name not in blocked]
    if len(candidate_pool) < int(n_images):
        raise RuntimeError(
            f"Only {len(candidate_pool)} candidates remain after blocking "
            f"{len(blocked)} rejected/neighbor views; requested {n_images}."
        )

    previous_coverage = selection.get("coverage", {})
    min_global_pose_distance = (
        float(min_global_pose_distance_override)
        if min_global_pose_distance_override is not None
        else float(previous_coverage.get("min_global_pose_distance", 0.05))
    )
    required_names: list[str] = []
    reference_assignments: list[dict] = []
    if reference_names:
        normalized_centers, directions, _ = build_pose_geometry(image_names, poses)
        features = np.concatenate([normalized_centers, 0.35 * directions], axis=1)
        name_to_index = {name: index for index, name in enumerate(image_names)}
        candidate_indices = [name_to_index[name] for name in candidate_pool]
        selected_required_indices: list[int] = []
        for raw_reference in reference_names:
            reference = normalize_required_name(raw_reference)
            if reference in name_to_index:
                reference_feature = features[name_to_index[reference]]
            elif reference_features is not None and reference in reference_features:
                reference_feature = np.asarray(reference_features[reference], dtype=np.float64)
            else:
                raise ValueError(
                    f"Reference chart has no scene pose or archived geometry: {reference}"
                )
            reference_sequence = reference.split("__", 1)[0]
            ranked = sorted(
                candidate_indices,
                key=lambda index: (
                    # Cambridge sequences encode observation continuity and
                    # occlusion history. A nearby camera from another sequence
                    # is not an equivalent geometric anchor.
                    image_names[index].split("__", 1)[0] != reference_sequence,
                    float(np.linalg.norm(features[index] - reference_feature)),
                    index,
                ),
            )
            exact_index = name_to_index.get(reference)
            chosen = (
                exact_index
                if preserve_reference_names
                and exact_index in candidate_indices
                and exact_index not in selected_required_indices
                else None
            )
            if chosen is None:
                chosen = next(
                (
                    index
                    for index in ranked
                    if index not in selected_required_indices
                    and (
                        not selected_required_indices
                        or min(
                            float(np.linalg.norm(features[index] - features[other]))
                            for other in selected_required_indices
                        ) >= min_global_pose_distance
                    )
                ),
                None,
            )
            if chosen is None:
                raise RuntimeError(f"No distinct clean candidate covers reference {reference}")
            distance = float(np.linalg.norm(features[chosen] - reference_feature))
            if distance > float(reference_max_pose_distance):
                raise RuntimeError(
                    f"Nearest distinct clean candidate for {reference} is too far: "
                    f"{distance:.4f} > {reference_max_pose_distance:.4f}"
                )
            selected_required_indices.append(chosen)
            required_names.append(image_names[chosen])
            reference_assignments.append(
                {
                    "reference_name": reference,
                    "selected_name": image_names[chosen],
                    "pose_distance": distance,
                }
            )
    selected_indices, coverage = select_clustered_coverage_indices(
        image_names=image_names,
        candidate_image_names=candidate_pool,
        poses=poses,
        n_images=int(n_images),
        n_view_clusters=int(previous_coverage.get("view_clusters", 8)),
        min_views_per_cluster=int(previous_coverage.get("min_views_per_cluster", 2)),
        min_baseline_ratio=float(previous_coverage.get("min_baseline_ratio", 0.02)),
        min_global_pose_distance=min_global_pose_distance,
        max_borrowed_support_distance=(
            float(max_borrowed_support_distance)
            if max_borrowed_support_distance is not None
            else float(previous_coverage.get("max_borrowed_support_distance", 0.25))
        ),
        coverage_objective="target_kcenter",
        required_names=required_names,
    )
    selected_names = [image_names[index] for index in selected_indices]
    history = list(selection.get("alignment_feedback_history", []))
    history.append(
        {
            "mode": "joint_full_set_reselection",
            "rejected_names": sorted(cumulative_rejected),
            "blocked_names": sorted(blocked),
            "quarantined_neighbor_names": sorted(neighbour_quarantine),
            "rejection_neighbor_radius": int(rejection_neighbor_radius),
            "hard_block_rejection_neighbors": bool(hard_block_rejection_neighbors),
            "correlated_failure_corridors": corridors,
            "correlated_corridor_blocked_names": sorted(corridor_blocked),
            "hard_block_correlated_failure_corridors": bool(
                hard_block_correlated_failure_corridors
            ),
            "correlated_failure_min_run_length": int(correlated_failure_min_run_length),
            "correlated_failure_padding": int(correlated_failure_padding),
            "correlated_failure_max_bridge_gap": int(correlated_failure_max_bridge_gap),
            "previous_names": list(selection.get("image_names", [])),
            "joint_selected_names": selected_names,
        }
    )
    return {
        **selection,
        "n_images": int(n_images),
        "requested_n_images": int(n_images),
        "actual_n_images": int(n_images),
        "image_idx": selected_indices,
        "image_names": selected_names,
        "candidate_pool_names": candidate_pool,
        "candidate_pool_size": len(candidate_pool),
        "run_sfm_arg": "--image_idx " + " ".join(map(str, selected_indices)),
        "coverage": coverage,
        "alignment_feedback_history": history,
        "alignment_feedback_converged": False,
        "joint_selection": {
            "version": 3,
            "mode": "redundant_full_set_reoptimization",
            "cumulative_rejected_names": sorted(cumulative_rejected),
            "blocked_names": sorted(blocked),
            "quarantined_neighbor_names": sorted(neighbour_quarantine),
            "hard_block_rejection_neighbors": bool(hard_block_rejection_neighbors),
            "correlated_failure_corridors": corridors,
            "correlated_corridor_blocked_names": sorted(corridor_blocked),
            "hard_block_correlated_failure_corridors": bool(
                hard_block_correlated_failure_corridors
            ),
            "correlated_failure_min_run_length": int(correlated_failure_min_run_length),
            "correlated_failure_padding": int(correlated_failure_padding),
            "correlated_failure_max_bridge_gap": int(correlated_failure_max_bridge_gap),
            "target_active_charts_after_gate": 24,
            "preserve_reference_names": bool(preserve_reference_names),
            "reference_anchor_assignments": reference_assignments,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, action="append", required=True)
    parser.add_argument("--scene-path", type=Path)
    parser.add_argument("--n-images", type=int, default=32)
    parser.add_argument("--rejection-neighbor-radius", type=int, default=2)
    parser.add_argument(
        "--hard-block-rejection-neighbors",
        action="store_true",
        help=(
            "Use the legacy policy that excludes every temporal neighbour of "
            "a rejected Chart. Default: retain independently clean neighbours "
            "as re-audited joint candidates."
        ),
    )
    parser.add_argument(
        "--disable-correlated-failure-corridor-block",
        action="store_true",
        help=(
            "Do not escalate a contiguous run of independently failed Charts "
            "to a local padded hard-exclusion corridor."
        ),
    )
    parser.add_argument(
        "--correlated-failure-min-run-length",
        type=int,
        default=3,
        help="Contiguous directly failed frames required before local escalation.",
    )
    parser.add_argument(
        "--correlated-failure-padding",
        type=int,
        default=2,
        help="Frames to exclude on each side of an escalated failure corridor.",
    )
    parser.add_argument(
        "--correlated-failure-max-bridge-gap",
        type=int,
        default=2,
        help=(
            "Maximum untested temporal gap that can join two independently "
            "failed ends of one local corridor."
        ),
    )
    parser.add_argument("--max-borrowed-support-distance", type=float)
    parser.add_argument(
        "--blocked-name",
        action="append",
        default=[],
        help="Additional chart rejected by an independent geometry/plane audit.",
    )
    parser.add_argument(
        "--reference-cameras-json",
        type=Path,
        help="MAtCha cameras.json whose proven chart poses must remain covered.",
    )
    parser.add_argument("--reference-max-pose-distance", type=float, default=0.25)
    parser.add_argument(
        "--preserve-reference-names",
        action="store_true",
        help="Keep every available proven reference chart exactly; otherwise prefer same-sequence replacements.",
    )
    parser.add_argument("--min-global-pose-distance", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    reports = [json.loads(path.read_text()) for path in args.gate_report]
    reference_names = None
    reference_features = None
    if args.reference_cameras_json is not None:
        reference_payload = json.loads(args.reference_cameras_json.read_text())
        reference_names = [Path(path).name for path in reference_payload["filepaths"]]
        scene_names, scene_poses = load_scene_poses(
            args.scene_path or Path(selection["scene_path"])
        )
        scene_centers = np.stack(
            [camera_center_and_direction(*scene_poses[name])[0] for name in scene_names]
        )
        scene_center = scene_centers.mean(axis=0)
        bbox_diag = max(
            float(np.linalg.norm(scene_centers.max(axis=0) - scene_centers.min(axis=0))),
            1e-12,
        )
        reference_features = {}
        for path, c2w in zip(
            reference_payload["filepaths"], reference_payload["cams2world"]
        ):
            matrix = np.asarray(c2w, dtype=np.float64)
            direction = matrix[:3, 2]
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            reference_features[Path(path).name] = np.concatenate(
                [(matrix[:3, 3] - scene_center) / bbox_diag, 0.35 * direction]
            )
    result = build_joint_selection(
        selection,
        reports,
        args.scene_path or Path(selection["scene_path"]),
        n_images=args.n_images,
        rejection_neighbor_radius=args.rejection_neighbor_radius,
        max_borrowed_support_distance=args.max_borrowed_support_distance,
        explicit_blocked_names=args.blocked_name,
        reference_names=reference_names,
        reference_features=reference_features,
        reference_max_pose_distance=args.reference_max_pose_distance,
        min_global_pose_distance_override=args.min_global_pose_distance,
        preserve_reference_names=args.preserve_reference_names,
        hard_block_rejection_neighbors=args.hard_block_rejection_neighbors,
        hard_block_correlated_failure_corridors=not args.disable_correlated_failure_corridor_block,
        correlated_failure_min_run_length=args.correlated_failure_min_run_length,
        correlated_failure_padding=args.correlated_failure_padding,
        correlated_failure_max_bridge_gap=args.correlated_failure_max_bridge_gap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "selected_count": len(result["image_names"]),
                "blocked_count": len(result["joint_selection"]["blocked_names"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
