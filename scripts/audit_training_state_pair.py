#!/usr/bin/env python3
"""Strictly compare two saved standard-2DGS training-state checkpoints.

Unlike a PLY comparison, this audit includes the optimizer moments,
densification buffers, loop EMA and RNG payload.  It is intentionally run on
CPU so comparing a checkpoint never alters GPU allocator or RNG state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_state(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Training state is not a mapping: {path}")
    return state


def _tensor_summary(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "torch_tensor",
        "dtype": str(left.dtype),
        "shape": list(left.shape),
        "exact": bool(torch.equal(left, right)),
    }
    if left.dtype.is_floating_point or left.dtype.is_complex:
        comparison_dtype = torch.complex128 if left.dtype.is_complex else torch.float64
        left64 = left.detach().to(dtype=comparison_dtype)
        right64 = right.detach().to(dtype=comparison_dtype)
        finite = torch.isfinite(left64) & torch.isfinite(right64)
        nonfinite_mismatch = int((~finite & ~(left64 == right64)).sum().item())
        if bool(finite.any()):
            differences = (left64[finite] - right64[finite]).abs()
            result.update(
                {
                    "max_abs_difference": float(differences.max().item()),
                    "mean_abs_difference": float(differences.mean().item()),
                    "nonzero_count": int((differences != 0).sum().item())
                    + nonfinite_mismatch,
                }
            )
        else:
            result.update(
                {
                    "max_abs_difference": 0.0,
                    "mean_abs_difference": 0.0,
                    "nonzero_count": nonfinite_mismatch,
                }
            )
        result["nonfinite_mismatch_count"] = nonfinite_mismatch
    else:
        result.update(
            {
                "max_abs_difference": None,
                "mean_abs_difference": None,
                "nonzero_count": int((left != right).sum().item()),
                "nonfinite_mismatch_count": 0,
            }
        )
    return result


def _ndarray_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "numpy_array",
        "dtype": str(left.dtype),
        "shape": list(left.shape),
        "exact": bool(np.array_equal(left, right, equal_nan=True)),
    }
    if np.issubdtype(left.dtype, np.inexact):
        left64 = left.astype(np.complex128 if np.iscomplexobj(left) else np.float64)
        right64 = right.astype(np.complex128 if np.iscomplexobj(right) else np.float64)
        finite = np.isfinite(left64) & np.isfinite(right64)
        nonfinite_mismatch = int(np.count_nonzero(~finite & ~(left64 == right64)))
        differences = np.abs(left64[finite] - right64[finite])
        result.update(
            {
                "max_abs_difference": float(differences.max()) if differences.size else 0.0,
                "mean_abs_difference": float(differences.mean()) if differences.size else 0.0,
                "nonzero_count": int(np.count_nonzero(differences != 0.0))
                + nonfinite_mismatch,
                "nonfinite_mismatch_count": nonfinite_mismatch,
            }
        )
    else:
        result.update(
            {
                "max_abs_difference": None,
                "mean_abs_difference": None,
                "nonzero_count": int(np.count_nonzero(left != right)),
                "nonfinite_mismatch_count": 0,
            }
        )
    return result


def _scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return bool(left == right)


def _compare(
    left: Any,
    right: Any,
    *,
    path: str,
    leaves: list[dict[str, Any]],
    structural_mismatches: list[dict[str, str]],
) -> None:
    """Recursively record exact comparisons without depending on dict order."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            structural_mismatches.append(
                {"path": path, "reason": f"type mismatch: {type(left)} vs {type(right)}"}
            )
            return
        if left.dtype != right.dtype or left.shape != right.shape:
            structural_mismatches.append(
                {
                    "path": path,
                    "reason": (
                        f"tensor dtype/shape mismatch: {left.dtype} {tuple(left.shape)} vs "
                        f"{right.dtype} {tuple(right.shape)}"
                    ),
                }
            )
            return
        summary = _tensor_summary(left, right)
        summary["path"] = path
        leaves.append(summary)
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            structural_mismatches.append(
                {"path": path, "reason": f"type mismatch: {type(left)} vs {type(right)}"}
            )
            return
        if left.dtype != right.dtype or left.shape != right.shape:
            structural_mismatches.append(
                {
                    "path": path,
                    "reason": (
                        f"array dtype/shape mismatch: {left.dtype} {left.shape} vs "
                        f"{right.dtype} {right.shape}"
                    ),
                }
            )
            return
        summary = _ndarray_summary(left, right)
        summary["path"] = path
        leaves.append(summary)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            structural_mismatches.append(
                {"path": path, "reason": f"type mismatch: {type(left)} vs {type(right)}"}
            )
            return
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            structural_mismatches.append(
                {
                    "path": path,
                    "reason": f"mapping keys differ: left_only={sorted(left_keys - right_keys)!r}, "
                    f"right_only={sorted(right_keys - left_keys)!r}",
                }
            )
        for key in sorted(left_keys & right_keys, key=repr):
            _compare(
                left[key],
                right[key],
                path=f"{path}[{key!r}]",
                leaves=leaves,
                structural_mismatches=structural_mismatches,
            )
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or not isinstance(left, (list, tuple)):
            structural_mismatches.append(
                {"path": path, "reason": f"sequence type mismatch: {type(left)} vs {type(right)}"}
            )
            return
        if len(left) != len(right):
            structural_mismatches.append(
                {"path": path, "reason": f"sequence length mismatch: {len(left)} vs {len(right)}"}
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _compare(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                leaves=leaves,
                structural_mismatches=structural_mismatches,
            )
        return
    if not _scalar_equal(left, right):
        leaves.append(
            {
                "path": path,
                "kind": "scalar",
                "dtype": type(left).__name__,
                "shape": [],
                "exact": False,
                "baseline": repr(left),
                "candidate": repr(right),
                "max_abs_difference": None,
                "mean_abs_difference": None,
                "nonzero_count": 1,
                "nonfinite_mismatch_count": 0,
            }
        )


def build_training_state_pair_audit(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    """Compare complete checkpoints, including tensors not serialised in a PLY."""
    baseline_path = baseline_path.resolve()
    candidate_path = candidate_path.resolve()
    leaves: list[dict[str, Any]] = []
    structural_mismatches: list[dict[str, str]] = []
    _compare(
        _load_state(baseline_path),
        _load_state(candidate_path),
        path="$",
        leaves=leaves,
        structural_mismatches=structural_mismatches,
    )
    inexact = [leaf for leaf in leaves if not leaf["exact"]]
    numeric_maxima = [
        leaf["max_abs_difference"]
        for leaf in leaves
        if leaf.get("max_abs_difference") is not None
    ]
    return {
        "protocol": "standard_2dgs_training_state_pair_audit_v1",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "comparison": "CPU map-location; recursive exact tensor/scalar comparison",
        "structural_mismatches": structural_mismatches,
        "tensor_or_scalar_leaf_count": len(leaves),
        "inexact_leaf_count": len(inexact),
        "inexact_leaves": inexact,
        "global_max_abs_difference": float(max(numeric_maxima, default=0.0)),
        "passed": not structural_mismatches and not inexact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training-state audit: {output}")
    audit = build_training_state_pair_audit(args.baseline, args.candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "tensor_or_scalar_leaf_count",
                    "inexact_leaf_count",
                    "global_max_abs_difference",
                    "passed",
                )
            },
            indent=2,
        )
    )
    if not audit["passed"]:
        raise SystemExit("Training-state pair audit failed; see report for details")


if __name__ == "__main__":
    main()
