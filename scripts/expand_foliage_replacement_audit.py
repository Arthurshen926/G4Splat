#!/usr/bin/env python
"""Safely expand legacy tree candidates from the retained full audit statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def expanded_candidate_mask(
    statistics,
    *,
    minimum_support_views,
    minimum_support_sequences,
    minimum_canopy_responsibility,
    maximum_rigid_responsibility,
    maximum_view_rigid_responsibility,
    minimum_canopy_residual,
    minimum_camera_depth,
    allow_giant_candidates,
    uv_local_ownership,
    minimum_risky_radius_views,
):
    if uv_local_ownership:
        # Primitive-level rigid rejection is invalid once ownership is decided
        # per intrinsic texel.  Admit every repeatedly observed canopy-touching
        # risky surfel; the online UV proof still pins rigid cells to one.
        risk = (
            (statistics["giant_radius_view_count"] > 0)
            | (
                statistics["risky_radius_view_count"]
                >= int(minimum_risky_radius_views)
            )
            | (
                statistics["minimum_camera_depth"]
                < float(minimum_camera_depth)
            )
        )
        return (
            (statistics["total_contribution"] > 0)
            & (statistics["support_views"] >= minimum_support_views)
            & (
                statistics["support_sequences"]
                >= minimum_support_sequences
            )
            & (
                statistics["canopy_responsibility"]
                >= minimum_canopy_responsibility
            )
            & risk
        )
    safe = (
        (statistics["support_views"] >= minimum_support_views)
        & (
            statistics["support_sequences"]
            >= minimum_support_sequences
        )
        & (
            statistics["canopy_responsibility"]
            >= minimum_canopy_responsibility
        )
        & (
            statistics["rigid_responsibility"]
            <= maximum_rigid_responsibility
        )
        & (
            statistics["maximum_view_rigid_responsibility"]
            <= maximum_view_rigid_responsibility
        )
        & (statistics["canopy_residual"] >= minimum_canopy_residual)
        & (statistics["minimum_camera_depth"] >= minimum_camera_depth)
    )
    if not allow_giant_candidates:
        safe &= statistics["giant_radius_view_count"] == 0
    return safe


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-support-views", type=int, default=3)
    parser.add_argument("--minimum-support-sequences", type=int, default=2)
    parser.add_argument("--minimum-canopy-responsibility", type=float, default=0.45)
    parser.add_argument("--maximum-rigid-responsibility", type=float, default=0.20)
    parser.add_argument(
        "--maximum-view-rigid-responsibility", type=float, default=0.25
    )
    parser.add_argument("--minimum-canopy-residual", type=float, default=0.0)
    parser.add_argument("--minimum-camera-depth", type=float, default=1.5)
    parser.add_argument(
        "--allow-giant-candidates",
        action="store_true",
        help=(
            "Admit giant legacy surfels when their canopy/rigid evidence is "
            "safe. They remain gated by online multi-view local replacement "
            "proof before retirement."
        ),
    )
    parser.add_argument(
        "--uv-local-ownership",
        action="store_true",
        help=(
            "Expand to all repeatedly observed canopy-touching risky/giant "
            "surfels, including mixed-rigid parents. Rigid safety is then "
            "enforced per UV texel rather than per primitive."
        ),
    )
    parser.add_argument(
        "--minimum-risky-radius-views", type=int, default=3
    )
    args = parser.parse_args()

    payload = _load(args.input.resolve())
    statistics = payload["statistics"]
    safe = expanded_candidate_mask(
        statistics,
        minimum_support_views=args.minimum_support_views,
        minimum_support_sequences=args.minimum_support_sequences,
        minimum_canopy_responsibility=args.minimum_canopy_responsibility,
        maximum_rigid_responsibility=args.maximum_rigid_responsibility,
        maximum_view_rigid_responsibility=(
            args.maximum_view_rigid_responsibility
        ),
        minimum_canopy_residual=args.minimum_canopy_residual,
        minimum_camera_depth=args.minimum_camera_depth,
        allow_giant_candidates=args.allow_giant_candidates,
        uv_local_ownership=args.uv_local_ownership,
        minimum_risky_radius_views=args.minimum_risky_radius_views,
    )
    # Previously accepted candidates already passed the stricter v5 protocol
    # and retain their learned gates when warm-starting the second stage.
    expanded = safe | statistics["candidate"].bool()
    candidate_indices = torch.nonzero(expanded, as_tuple=False).flatten()
    result = dict(payload)
    result["candidate_indices"] = candidate_indices
    result["statistics"] = dict(statistics)
    result["statistics"]["candidate"] = expanded
    result["expansion"] = {
        "protocol": (
            "intrinsic_uv_local_ownership_candidate_expansion_v2"
            if args.uv_local_ownership
            else "retained_full_statistics_safe_candidate_expansion_v1"
        ),
        "original_candidate_count": int(
            payload["candidate_indices"].numel()
        ),
        "safe_new_candidate_count": int(
            (safe & ~statistics["candidate"].bool()).sum()
        ),
        "expanded_candidate_count": int(candidate_indices.numel()),
        "preserve_original_candidates": True,
        "criteria": {
            "minimum_support_views": args.minimum_support_views,
            "minimum_support_sequences": args.minimum_support_sequences,
            "minimum_canopy_responsibility": (
                args.minimum_canopy_responsibility
            ),
            "maximum_rigid_responsibility": (
                args.maximum_rigid_responsibility
            ),
            "maximum_view_rigid_responsibility": (
                args.maximum_view_rigid_responsibility
            ),
            "minimum_canopy_residual": args.minimum_canopy_residual,
            "minimum_camera_depth": args.minimum_camera_depth,
            "require_zero_giant_radius_views": (
                not args.allow_giant_candidates
            ),
            "giant_candidates_require_online_local_proof": (
                args.allow_giant_candidates
            ),
            "uv_local_ownership": args.uv_local_ownership,
            "minimum_risky_radius_views": (
                args.minimum_risky_radius_views
            ),
            "primitive_level_rigid_rejection": (
                not args.uv_local_ownership
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(result["expansion"])


if __name__ == "__main__":
    main()
