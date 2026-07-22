#!/usr/bin/env python3
"""Audit the live camera-stream contract of two 2DGS optimization prefixes.

Wall-clock timing is intentionally excluded.  CUDA rasterization and topology
statistics can diverge slightly across otherwise identical launches, so RGB
losses and Gaussian counts are reported diagnostically rather than treated as
a false bitwise-determinism requirement.  The logged camera identity at every
iteration remains the strict execution-contract check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_IDENTITY_FIELDS = (
    "iteration",
    "camera",
)
_DIAGNOSTIC_FIELDS = (
    "rgb_l1",
    "rgb_loss",
    "normal_loss",
    "distortion_loss",
    "total_loss",
    "ema_total_loss",
    "gaussians",
)
_TRACE_FIELDS = _IDENTITY_FIELDS + _DIAGNOSTIC_FIELDS


def _read_trace(path: Path) -> dict[int, dict[str, Any]]:
    path = path.resolve()
    records: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise RuntimeError(f"Could not read training trace {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict) or "iteration" not in row:
            raise RuntimeError(f"Trace row at {path}:{line_number} lacks iteration")
        iteration = int(row["iteration"])
        if iteration in records:
            raise RuntimeError(f"Trace {path} contains duplicate iteration {iteration}")
        missing = [field for field in _TRACE_FIELDS if field not in row]
        if missing:
            raise RuntimeError(f"Trace row at {path}:{line_number} lacks fields {missing}")
        records[iteration] = row
    if not records:
        raise RuntimeError(f"Training trace is empty: {path}")
    return records


def build_training_trace_pair_audit(baseline: Path, candidate: Path) -> dict[str, Any]:
    """Compare exact camera identity and summarize numerical drift."""
    baseline_rows = _read_trace(baseline)
    candidate_rows = _read_trace(candidate)
    baseline_iterations = set(baseline_rows)
    candidate_iterations = set(candidate_rows)
    shared = sorted(baseline_iterations & candidate_iterations)
    identity_mismatches: list[dict[str, Any]] = []
    gaussian_count_mismatches: list[dict[str, Any]] = []
    numeric_max_abs_delta = {
        field: 0.0
        for field in _DIAGNOSTIC_FIELDS
        if field != "gaussians"
    }
    for iteration in shared:
        before = baseline_rows[iteration]
        after = candidate_rows[iteration]
        for field in _IDENTITY_FIELDS:
            if before[field] != after[field]:
                identity_mismatches.append(
                    {
                        "iteration": iteration,
                        "field": field,
                        "baseline": before[field],
                        "candidate": after[field],
                    }
                )
        if before["gaussians"] != after["gaussians"]:
            gaussian_count_mismatches.append(
                {
                    "iteration": iteration,
                    "baseline": before["gaussians"],
                    "candidate": after["gaussians"],
                }
            )
        for field in numeric_max_abs_delta:
            numeric_max_abs_delta[field] = max(
                numeric_max_abs_delta[field],
                abs(float(before[field]) - float(after[field])),
            )
    return {
        "protocol": "standard_2dgs_training_trace_pair_audit_v1",
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "strict_identity_fields": list(_IDENTITY_FIELDS),
        "diagnostic_fields": list(_DIAGNOSTIC_FIELDS),
        "ignored_fields": ["elapsed_sec"],
        "baseline_logged_iterations": sorted(baseline_iterations),
        "candidate_logged_iterations": sorted(candidate_iterations),
        "shared_logged_iterations": shared,
        "baseline_only_iterations": sorted(baseline_iterations - candidate_iterations),
        "candidate_only_iterations": sorted(candidate_iterations - baseline_iterations),
        "identity_mismatch_count": len(identity_mismatches),
        "identity_mismatches": identity_mismatches,
        "gaussian_count_mismatch_count": len(gaussian_count_mismatches),
        "gaussian_count_mismatches": gaussian_count_mismatches,
        "numeric_max_abs_delta": numeric_max_abs_delta,
        "passed": (
            baseline_iterations == candidate_iterations
            and not identity_mismatches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite trace audit: {output}")
    report = build_training_trace_pair_audit(args.baseline, args.candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Training trace camera-stream audit failed; see report for details")


if __name__ == "__main__":
    main()
