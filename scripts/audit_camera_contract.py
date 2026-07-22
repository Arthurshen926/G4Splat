#!/usr/bin/env python3
"""Compare exported camera contracts from two independent implementations.

The G4Splat and clean STDLoc/ULF-Loc scene loaders export ``cameras.json``
when they construct a scene.  This tool checks the camera *set* and every
geometric field by image identity, rather than assuming matching output order
proves that two training runs saw the same cameras.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


_FIELDS = ("id", "width", "height", "fx", "fy", "position", "rotation")


def canonical_name(value: str) -> str:
    """Normalize extension and staged ``/`` -> ``__`` path flattening."""
    return str(value).replace("\\", "/").replace("/", "__").rsplit(".", 1)[0]


def load_camera_index(path: Path) -> dict[str, dict[str, Any]]:
    """Read one exported camera list into a unique, normalized-name mapping."""
    path = path.resolve()
    try:
        values = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read camera JSON {path}: {error}") from error
    if not isinstance(values, list):
        raise RuntimeError(f"Camera JSON must contain a list: {path}")

    indexed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        if not isinstance(value, dict) or "img_name" not in value:
            raise RuntimeError(f"Camera entry {index} in {path} lacks img_name")
        name = canonical_name(str(value["img_name"]))
        if name in indexed:
            raise RuntimeError(f"Camera JSON has duplicate normalized name {name!r}: {path}")
        missing = [field for field in _FIELDS if field not in value]
        if missing:
            raise RuntimeError(f"Camera {name!r} in {path} lacks fields {missing}")
        indexed[name] = value
    return indexed


def _max_abs_difference(left: Any, right: Any) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        return float("inf")
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        return float("inf")
    if left_values.size == 0:
        return 0.0
    return float(np.max(np.abs(left_values - right_values)))


def build_camera_contract_audit(
    baseline: Path,
    candidate: Path,
    *,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Return an explicit, image-aligned geometry comparison report."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    baseline_index = load_camera_index(baseline)
    candidate_index = load_camera_index(candidate)
    baseline_names = set(baseline_index)
    candidate_names = set(candidate_index)
    shared_names = sorted(baseline_names & candidate_names)

    field_max_abs = {field: 0.0 for field in _FIELDS}
    mismatches: list[dict[str, Any]] = []
    for name in shared_names:
        baseline_camera = baseline_index[name]
        candidate_camera = candidate_index[name]
        for field in _FIELDS:
            difference = _max_abs_difference(
                baseline_camera[field], candidate_camera[field]
            )
            field_max_abs[field] = max(field_max_abs[field], difference)
            if difference > tolerance:
                mismatches.append(
                    {"image": name, "field": field, "max_abs_difference": difference}
                )

    return {
        "protocol": "cross_implementation_camera_contract_audit_v1",
        "baseline": str(baseline.resolve()),
        "candidate": str(candidate.resolve()),
        "tolerance": float(tolerance),
        "baseline_camera_count": len(baseline_index),
        "candidate_camera_count": len(candidate_index),
        "shared_camera_count": len(shared_names),
        "baseline_only_images": sorted(baseline_names - candidate_names),
        "candidate_only_images": sorted(candidate_names - baseline_names),
        "field_max_abs_difference": field_max_abs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": (
            baseline_names == candidate_names
            and not mismatches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite camera audit: {output}")
    report = build_camera_contract_audit(
        args.baseline,
        args.candidate,
        tolerance=float(args.tolerance),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Camera contract audit failed; see report for details")


if __name__ == "__main__":
    main()
