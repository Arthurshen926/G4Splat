#!/usr/bin/env python3
"""Verify that a protected continuation leaves its checkpoint prefix unchanged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


MODEL_FIELDS = (
    "xyz",
    "features_dc",
    "features_rest",
    "scaling",
    "rotation",
    "opacity",
)


def _load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"Unsupported Gaussian checkpoint format: {path}")
    model, iteration = payload
    if not isinstance(model, tuple) or len(model) < 7:
        raise ValueError(f"Malformed Gaussian checkpoint model payload: {path}")
    return model, int(iteration)


def _field_report(baseline: torch.Tensor, candidate: torch.Tensor, count: int) -> dict:
    if candidate.shape[0] < count:
        raise ValueError(
            f"Candidate field has {candidate.shape[0]} rows, below protected prefix {count}"
        )
    if tuple(baseline.shape) != tuple(candidate[:count].shape):
        raise ValueError(
            "Checkpoint field shapes disagree: "
            f"baseline={tuple(baseline.shape)}, candidate_prefix={tuple(candidate[:count].shape)}"
        )
    delta = (candidate[:count] - baseline).abs()
    per_row = delta.reshape(count, -1).amax(dim=1)
    return {
        "shape": list(baseline.shape),
        "mean_absolute_delta": float(delta.float().mean().item()),
        "max_absolute_delta": float(delta.max().item()),
        "changed_row_count": int((per_row > 0).sum().item()),
        "exact": bool(torch.equal(baseline, candidate[:count])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-count",
        type=int,
        default=None,
        help="Protected prefix length; defaults to the baseline checkpoint's point count.",
    )
    args = parser.parse_args()

    baseline_path = args.baseline_checkpoint.expanduser().resolve()
    candidate_path = args.candidate_checkpoint.expanduser().resolve()
    baseline_model, baseline_iteration = _load_checkpoint(baseline_path)
    candidate_model, candidate_iteration = _load_checkpoint(candidate_path)
    available = int(baseline_model[1].shape[0])
    count = available if args.baseline_count is None else int(args.baseline_count)
    if count < 1 or count > available:
        raise ValueError(
            f"--baseline-count must lie in [1, {available}], received {count}"
        )

    fields = {
        name: _field_report(baseline_model[index], candidate_model[index], count)
        for index, name in enumerate(MODEL_FIELDS, start=1)
    }
    report = {
        "schema_version": "checkpoint-prefix-integrity-v1",
        "baseline_checkpoint": str(baseline_path),
        "candidate_checkpoint": str(candidate_path),
        "baseline_iteration": baseline_iteration,
        "candidate_iteration": candidate_iteration,
        "protected_prefix_count": count,
        "candidate_total_gaussian_count": int(candidate_model[1].shape[0]),
        "fields": fields,
        "all_fields_exact": bool(all(field["exact"] for field in fields.values())),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_fields_exact"]:
        raise SystemExit("Protected prefix changed during continuation")


if __name__ == "__main__":
    main()
