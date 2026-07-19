"""Eligibility audit for globally fused planar depth constraints.

A plane observed in only one image has no cross-view geometric evidence.  It
may still be useful as a *local* segmentation, but it must not rewrite an
aligned Chart depth as though it were a global surface.  This module is kept
dependency-free so the policy can be unit-tested independently of CUDA.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def filter_global_planes_by_view_support(
    global_plane_members: Mapping[int, Sequence[Sequence[int]]],
    *,
    minimum_distinct_views: int,
    real_view_upper_bound: int | None = None,
) -> tuple[dict[int, Sequence[Sequence[int]]], dict[str, Any]]:
    """Return only global planes supported by enough distinct real views.

    ``global_plane_members`` maps a global plane ID to ``(view_id,
    local_plane_id)`` memberships.  Repeated local memberships in the same
    view do not increase geometric support.  When pseudo views are appended to
    the camera list, ``real_view_upper_bound`` limits support to the original
    real Chart IDs; generated views must never make a plane geometrically
    eligible.
    """
    if minimum_distinct_views < 1:
        raise ValueError("minimum_distinct_views must be at least one")
    if real_view_upper_bound is not None and real_view_upper_bound < 0:
        raise ValueError("real_view_upper_bound must be non-negative")

    active: dict[int, Sequence[Sequence[int]]] = {}
    records: list[dict[str, Any]] = []
    for raw_plane_id, members in global_plane_members.items():
        plane_id = int(raw_plane_id)
        view_ids = sorted({int(member[0]) for member in members})
        real_view_ids = (
            view_ids
            if real_view_upper_bound is None
            else [view_id for view_id in view_ids if view_id < real_view_upper_bound]
        )
        accepted = len(real_view_ids) >= minimum_distinct_views
        if accepted:
            active[plane_id] = members
        records.append(
            {
                "global_plane_id": plane_id,
                "distinct_view_count": len(view_ids),
                "distinct_real_view_count": len(real_view_ids),
                "local_membership_count": len(members),
                "view_ids": view_ids,
                "real_view_ids": real_view_ids,
                "accepted": accepted,
                "reason": "cross_view_support_passed"
                if accepted
                else "insufficient_cross_view_support",
            }
        )

    records.sort(key=lambda item: item["global_plane_id"])
    audit = {
        "version": 1,
        "minimum_distinct_views": int(minimum_distinct_views),
        "real_view_upper_bound": (
            int(real_view_upper_bound) if real_view_upper_bound is not None else None
        ),
        "total_global_plane_count": len(records),
        "accepted_global_plane_count": len(active),
        "rejected_global_plane_count": len(records) - len(active),
        "accepted_local_membership_count": int(
            sum(item["local_membership_count"] for item in records if item["accepted"])
        ),
        "rejected_local_membership_count": int(
            sum(item["local_membership_count"] for item in records if not item["accepted"])
        ),
        "records": records,
    }
    return active, audit


def append_plane_coverage_audit(
    audit: dict[str, Any],
    *,
    active_global_plane_members: Mapping[int, Sequence[Sequence[int]]],
    plane_masks: Sequence[np.ndarray],
    visibility_masks: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Attach per-Chart coverage of *eligible* global planes to an audit.

    This is intentionally an audit rather than a quota: a Chart without a
    multi-view planar patch retains its aligned depth and remains a valid RGB
    supervision view.  The important invariant is that no unsupported plane
    gets to rewrite it.
    """
    eligible_by_view: dict[int, set[int]] = {}
    for members in active_global_plane_members.values():
        for raw_view_id, raw_plane_id in members:
            eligible_by_view.setdefault(int(raw_view_id), set()).add(int(raw_plane_id))

    rows: list[dict[str, Any]] = []
    for view_id, raw_plane_mask in enumerate(plane_masks):
        plane_mask = np.asarray(raw_plane_mask)
        if plane_mask.ndim != 2:
            raise ValueError("plane_masks must contain 2-D label arrays")
        plane_region = plane_mask > 0
        eligible_ids = sorted(eligible_by_view.get(view_id, set()))
        eligible_region = np.isin(plane_mask, eligible_ids) if eligible_ids else np.zeros_like(plane_region)
        visible_eligible_region = None
        if visibility_masks is not None:
            visible = np.asarray(visibility_masks[view_id]) > 0.5
            if visible.shape != plane_mask.shape:
                raise ValueError("visibility mask shape must match plane mask shape")
            visible_eligible_region = eligible_region & visible
        rows.append(
            {
                "view_id": int(view_id),
                "local_plane_pixel_fraction": float(plane_region.mean()),
                "eligible_cross_view_plane_pixel_fraction": float(eligible_region.mean()),
                "eligible_fraction_of_local_plane_pixels": (
                    float(eligible_region[plane_region].mean()) if np.any(plane_region) else 0.0
                ),
                "eligible_visible_pixel_fraction": (
                    float(visible_eligible_region.mean())
                    if visible_eligible_region is not None
                    else None
                ),
                "eligible_local_plane_count": len(eligible_ids),
            }
        )

    fraction = np.asarray(
        [row["eligible_cross_view_plane_pixel_fraction"] for row in rows], dtype=np.float64
    )
    audit["eligible_cross_view_coverage"] = {
        "coverage_is_diagnostic_not_a_required_per_view_quota": True,
        "chart_count": len(rows),
        "charts_without_eligible_cross_view_plane": int(np.sum(fraction == 0.0)),
        "mean_eligible_cross_view_plane_pixel_fraction": float(fraction.mean()) if len(fraction) else 0.0,
        "median_eligible_cross_view_plane_pixel_fraction": float(np.median(fraction)) if len(fraction) else 0.0,
        "minimum_eligible_cross_view_plane_pixel_fraction": float(fraction.min()) if len(fraction) else 0.0,
        "all_active_global_planes_have_min_real_support": bool(
            all(record.get("accepted", False) for record in audit.get("records", []) if record.get("accepted"))
        ),
        "per_chart": rows,
    }
    return audit
