#!/usr/bin/env python3
"""Audit that a causal 2DGS repair report satisfies its own safety contracts.

This is deliberately read-only.  It turns the experiment report into explicit
pass/fail checks so a later local PLY is never promoted just because a visual
looks better in one target image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audit(report: dict[str, Any], *, expected_full_train_views: int | None) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    full_count = int(report.get("full_train_view_count", 0))
    checks["full_train_frontend"] = {
        "observed_view_count": full_count,
        "expected_view_count": expected_full_train_views,
        "passed": expected_full_train_views is None or full_count == int(expected_full_train_views),
    }
    if not checks["full_train_frontend"]["passed"]:
        failures.append("full_train_view_count does not match the required all-train reconstruction set")

    # Version 5 introduced a mandatory all-real-view anomaly prescan before a
    # small subset may enter expensive primitive attribution.  Older reports
    # remain readable for historical comparison, but a new report cannot claim
    # a causal local repair after looking only at a hand-picked target list.
    if int(report.get("version", 0)) >= 5:
        prescan = report.get("full_train_anomaly_prescan", {})
        expected = full_count
        processed = int(prescan.get("processed_view_count", 0))
        checks["full_train_anomaly_prescan"] = {
            "status": prescan.get("status"),
            "expected_view_count": expected,
            "processed_view_count": processed,
            "report_path": prescan.get("report_path"),
            "passed": bool(
                prescan.get("status") == "completed"
                and processed == expected
                and bool(prescan.get("report_path"))
            ),
        }
        if not checks["full_train_anomaly_prescan"]["passed"]:
            failures.append(
                "version-5 causal repair lacks a complete all-real-training-view anomaly prescan"
            )

    removal_checks = []
    for cluster in report.get("counterfactual_clusters", []):
        if not bool(cluster.get("accepted")):
            continue
        threshold = cluster.get("acceptance_thresholds", {})
        row = {
            "cluster_id": cluster.get("cluster_id"),
            "visible_clean_controls": int(cluster.get("visible_clean_control_count", 0)),
            "bad_relative_improvement": _as_float(cluster.get("median_bad_relative_improvement"), -float("inf")),
            "max_clean_mae_increase": _as_float(cluster.get("worst_clean_mae_increase")),
            "thresholds": threshold,
        }
        envelope = cluster.get("control_envelope", {})
        if int(report.get("version", 0)) >= 6:
            row["real_camera_envelope"] = envelope
            row["real_camera_envelope_passed"] = bool(
                envelope.get("policy") == "real_camera_envelope_v1"
                and bool(envelope.get("quality_is_not_a_filter"))
                and int(envelope.get("temporal_radius_frames", 0)) > 0
                and int(envelope.get("minimum_visible_control_count", 0))
                == int(threshold.get("min_visible_clean_controls", 0))
            )
        else:
            row["real_camera_envelope_passed"] = True
        # ``causal_2dgs_repair.py`` deliberately reports a *relative* bad-ray
        # improvement.  Keep the old spelling as a read-only compatibility
        # fallback for early experiment reports, but never turn a missing
        # threshold into a permissive pass.
        min_bad_improvement = _as_float(
            threshold.get(
                "min_bad_relative_improvement",
                threshold.get("min_bad_improvement"),
            ),
            float("inf"),
        )
        row["passed"] = bool(
            row["visible_clean_controls"] >= int(threshold.get("min_visible_clean_controls", 1))
            and row["bad_relative_improvement"] >= min_bad_improvement
            and row["max_clean_mae_increase"] <= _as_float(threshold.get("max_clean_mae_increase"))
            and row["real_camera_envelope_passed"]
        )
        removal_checks.append(row)
        if not row["passed"]:
            failures.append(f"accepted G_remove cluster {row['cluster_id']} fails its opacity-zero contract")
    checks["accepted_removals"] = removal_checks

    projection_reports = report.get("geometry_projection_counterfactuals", {})
    projection_checks = []
    for cluster_id, trial in projection_reports.items():
        if not bool(trial.get("accepted")):
            continue
        threshold = trial.get("acceptance_thresholds", {})
        row = {
            "cluster_id": cluster_id,
            "candidate_count": int(trial.get("candidate_count", 0)),
            "visible_clean_controls": len(trial.get("clean_controls", [])),
            "target_relative_improvement": _as_float(trial.get("median_target_relative_improvement"), -float("inf")),
            "max_clean_mae_increase": _as_float(trial.get("worst_clean_mae_increase")),
            "thresholds": threshold,
        }
        row["passed"] = bool(
            row["candidate_count"] > 0
            and row["visible_clean_controls"] >= int(threshold.get("min_visible_clean_controls", 1))
            and row["target_relative_improvement"]
            >= _as_float(threshold.get("min_target_relative_improvement"), float("inf"))
            and row["max_clean_mae_increase"] <= _as_float(threshold.get("max_clean_mae_increase"))
        )
        projection_checks.append(row)
        if not row["passed"]:
            failures.append(f"accepted G_update projection cluster {cluster_id} fails its geometry contract")
    checks["accepted_geometry_projections"] = projection_checks

    audit_groups = report.get("edit_plan", {}).get("audit_groups", [])
    if not audit_groups:
        # Current reports write the same content under ``sets`` only through a
        # sidecar; absence is safe for a no-op but not for a reported G_update.
        update_count = int(report.get("selected_update_primitive_count", 0))
        if update_count:
            failures.append("G_update is non-empty but no edit-plan audit groups were recorded")
    else:
        for group in audit_groups:
            if int(group.get("G_update_count", 0)) <= 0:
                continue
            trial = group.get("plane_projection_counterfactual", {})
            if not bool(trial.get("accepted")):
                failures.append(
                    f"G_update group {group.get('cluster_id')} lacks an accepted plane-projection counterfactual"
                )

    applied = report.get("applied_edit")
    if applied and applied.get("mode") != "no_op":
        gate = applied.get("validation_gate", {})
        checks["written_edit_validation"] = gate
        if "output_model" in applied and not bool(gate.get("passed")):
            failures.append("an edited model was written without a passing final validation gate")
        if int(report.get("version", 0)) >= 7 and "output_model" in applied:
            target_scope = gate.get("target_scope", {})
            scoped_targets = list(target_scope.get("names", []))
            checks["post_edit_causal_target_scope"] = {
                "policy": target_scope.get("policy"),
                "count": len(scoped_targets),
                "names": scoped_targets,
                "passed": bool(
                    target_scope.get("policy") == "accepted_counterfactual_causal_targets_only"
                    and scoped_targets
                ),
            }
            if not checks["post_edit_causal_target_scope"]["passed"]:
                failures.append("version-7 written edit lacks a non-empty accepted-causal-target validation scope")
    elif int(report.get("selected_update_primitive_count", 0)) and not projection_reports:
        warnings.append("legacy report predates direct plane-projection counterfactuals")

    return {
        "mode": "causal_repair_contract_audit",
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-full-train-views",
        type=int,
        default=1487,
        help="Set to 0 to skip the count check for a non-Cambridge fixture.",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = audit(
        report,
        expected_full_train_views=None if int(args.expected_full_train_views) == 0 else int(args.expected_full_train_views),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
