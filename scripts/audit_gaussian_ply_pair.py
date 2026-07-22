#!/usr/bin/env python3
"""Numerically audit two ordered Gaussian PLY checkpoints.

An exact optimizer/RNG state fork should preserve point count, field order,
and the parent topology.  Separate CUDA processes can still introduce a few
last-bit differences through atomic rasterization reductions, so byte hashes
are intentionally not the pass criterion.  This tool records a per-field
numeric bound and fails closed if it exceeds the declared tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


def _vertex_table(path: Path) -> np.ndarray:
    path = path.resolve()
    try:
        ply = PlyData.read(str(path))
    except Exception as error:  # pragma: no cover - depends on plyfile errors
        raise RuntimeError(f"Could not read Gaussian PLY {path}: {error}") from error
    if "vertex" not in ply:
        raise RuntimeError(f"Gaussian PLY {path} does not contain a vertex element")
    table = ply["vertex"].data
    if table.dtype.names is None:
        raise RuntimeError(f"Gaussian PLY {path} has no named vertex properties")
    return table


def _field_difference(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if baseline.dtype != candidate.dtype:
        return {
            "comparable": False,
            "reason": f"dtype mismatch: {baseline.dtype} vs {candidate.dtype}",
        }
    if np.issubdtype(baseline.dtype, np.number):
        before = baseline.astype(np.float64, copy=False)
        after = candidate.astype(np.float64, copy=False)
        equal = np.array_equal(before, after, equal_nan=True)
        difference = np.abs(after - before)
        finite = np.isfinite(difference)
        nonfinite_mismatch = int(np.count_nonzero(~finite & (before != after)))
        finite_difference = difference[finite]
        return {
            "comparable": True,
            "exact": bool(equal),
            "nonzero_count": int(np.count_nonzero(difference[finite] != 0.0))
            + nonfinite_mismatch,
            "max_abs_difference": (
                float(finite_difference.max()) if finite_difference.size else 0.0
            ),
            "mean_abs_difference": (
                float(finite_difference.mean()) if finite_difference.size else 0.0
            ),
            "nonfinite_mismatch_count": nonfinite_mismatch,
        }
    return {
        "comparable": True,
        "exact": bool(np.array_equal(baseline, candidate)),
        "nonzero_count": int(np.count_nonzero(baseline != candidate)),
        "max_abs_difference": None,
        "mean_abs_difference": None,
        "nonfinite_mismatch_count": 0,
    }


def build_ply_pair_audit(
    baseline_path: Path,
    candidate_path: Path,
    *,
    atol: float,
) -> dict[str, Any]:
    """Compare two Gaussian PLYs by row and property, without reordering."""
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    baseline_path = baseline_path.resolve()
    candidate_path = candidate_path.resolve()
    baseline = _vertex_table(baseline_path)
    candidate = _vertex_table(candidate_path)
    baseline_fields = list(baseline.dtype.names or ())
    candidate_fields = list(candidate.dtype.names or ())

    fields: dict[str, Any] = {}
    schema_match = baseline.shape == candidate.shape and baseline_fields == candidate_fields
    if schema_match:
        for name in baseline_fields:
            fields[name] = _field_difference(baseline[name], candidate[name])

    numeric_maxima = [
        field["max_abs_difference"]
        for field in fields.values()
        if field.get("comparable") and field.get("max_abs_difference") is not None
    ]
    noncomparable = [name for name, field in fields.items() if not field.get("comparable")]
    nonfinite = sum(
        int(field.get("nonfinite_mismatch_count", 0)) for field in fields.values()
    )
    max_abs = max(numeric_maxima, default=0.0)
    passed = bool(
        schema_match
        and not noncomparable
        and nonfinite == 0
        and max_abs <= float(atol)
    )
    return {
        "protocol": "ordered_gaussian_ply_pair_audit_v1",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "row_order_contract": "same serialized row order; no nearest-neighbor matching",
        "atol": float(atol),
        "baseline_vertex_count": int(baseline.shape[0]),
        "candidate_vertex_count": int(candidate.shape[0]),
        "baseline_fields": baseline_fields,
        "candidate_fields": candidate_fields,
        "schema_match": bool(schema_match),
        "global_max_abs_difference": float(max_abs),
        "noncomparable_fields": noncomparable,
        "nonfinite_mismatch_count": int(nonfinite),
        "fields": fields,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite PLY-pair audit: {output}")
    audit = build_ply_pair_audit(args.baseline, args.candidate, atol=args.atol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "baseline_vertex_count",
                    "candidate_vertex_count",
                    "schema_match",
                    "global_max_abs_difference",
                    "nonfinite_mismatch_count",
                    "passed",
                )
            },
            indent=2,
        )
    )
    if not audit["passed"]:
        raise SystemExit("Gaussian PLY pair audit failed; see report for details")


if __name__ == "__main__":
    main()
