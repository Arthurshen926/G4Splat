#!/usr/bin/env python3
"""Audit that two standard 2DGS controls differ only in declared fields.

The training runner writes a renderer-compatible ``cfg_args`` representation
rather than a JSON file.  This utility parses that restricted Namespace form
without evaluating it, compares the effective command-line contract, and
cross-checks the immutable all-train inputs recorded in each manifest.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def parse_cfg_args(path: Path) -> dict[str, Any]:
    """Parse the runner's literal ``Namespace(key=value, ...)`` format safely."""
    path = path.resolve()
    try:
        expression = ast.parse(path.read_text().strip(), mode="eval").body
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"Could not parse cfg_args at {path}: {error}") from error
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "Namespace"
        and not expression.args
        and all(keyword.arg is not None for keyword in expression.keywords)
    ):
        raise RuntimeError(f"cfg_args at {path} is not a literal Namespace(...) expression")
    try:
        return {
            str(keyword.arg): ast.literal_eval(keyword.value)
            for keyword in expression.keywords
        }
    except ValueError as error:
        raise RuntimeError(f"cfg_args at {path} contains a non-literal value") from error


def _read_manifest(run: Path) -> dict[str, Any]:
    path = run / "input_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Standard-control manifest does not exist: {path}")
    return json.loads(path.read_text())


def build_pair_audit(
    baseline: Path,
    candidate: Path,
    *,
    allowed_config_differences: set[str],
    allowed_manifest_differences: set[str] | None = None,
) -> dict[str, Any]:
    """Return a machine-checkable single-variable comparison report."""
    baseline = baseline.resolve()
    candidate = candidate.resolve()
    baseline_cfg = parse_cfg_args(baseline / "cfg_args")
    candidate_cfg = parse_cfg_args(candidate / "cfg_args")
    baseline_manifest = _read_manifest(baseline)
    candidate_manifest = _read_manifest(candidate)
    allowed_manifest_differences = allowed_manifest_differences or set()

    config_differences = []
    for field in sorted(set(baseline_cfg) | set(candidate_cfg)):
        before = baseline_cfg.get(field)
        after = candidate_cfg.get(field)
        if before != after:
            config_differences.append(
                {
                    "field": field,
                    "baseline": before,
                    "candidate": after,
                    "allowed": field in allowed_config_differences,
                }
            )

    # These fields describe data identity and the optimizer trajectory. Paths
    # to each run's copied input.ply differ by construction, so compare their
    # recorded digest rather than their run-local filenames.
    immutable_manifest_fields = (
        "source_path",
        "image_count",
        "colmap_camera_count",
        "loaded_camera_count",
        "camera_set_sha256_input",
        "name_mapping_sha256",
        "effective_initialization_ply_sha256",
        "cambridge_mask_pickle_sha256",
        "supervision_profile",
        "ulfloc_native_pixel_protocol",
        "white_background",
        "opacity_reset_iterations",
        "cudnn",
        "camera_schedule",
        "mask_resolution_audit",
        "implementation",
        # Two causal branches must resume the exact same optimizer/RNG/model
        # checkpoint.  The runner records that parent under this field; treat
        # it as immutable so a pair audit cannot accidentally compare two
        # unrelated fresh trajectories.
        "training_state",
    )
    manifest_differences = []
    for field in immutable_manifest_fields:
        before = baseline_manifest.get(field)
        after = candidate_manifest.get(field)
        if before != after:
            manifest_differences.append(
                {
                    "field": field,
                    "baseline": before,
                    "candidate": after,
                    "allowed": field in allowed_manifest_differences,
                }
            )

    unexpected_config = [
        row for row in config_differences if not row["allowed"]
    ]
    unexpected_manifest = [
        row for row in manifest_differences if not row["allowed"]
    ]
    return {
        "protocol": "standard_2dgs_single_variable_pair_audit_v1",
        "baseline": str(baseline),
        "candidate": str(candidate),
        "allowed_config_differences": sorted(allowed_config_differences),
        "allowed_manifest_differences": sorted(allowed_manifest_differences),
        "config_differences": config_differences,
        "unexpected_config_differences": unexpected_config,
        "immutable_manifest_differences": manifest_differences,
        "unexpected_immutable_manifest_differences": unexpected_manifest,
        "passed": not unexpected_config and not unexpected_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--allow-config-difference",
        action="append",
        default=["model_path"],
        help=(
            "cfg_args field allowed to differ. model_path is always allowed by "
            "default; repeat this option for the tested variable."
        ),
    )
    parser.add_argument(
        "--allow-manifest-difference",
        action="append",
        default=[],
        help=(
            "Immutable manifest field allowed to differ for the declared "
            "implementation variable. This does not suppress any other "
            "data-identity or optimizer-contract difference."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pair-audit report: {output}")
    audit = build_pair_audit(
        args.baseline,
        args.candidate,
        allowed_config_differences=set(args.allow_config_difference),
        allowed_manifest_differences=set(args.allow_manifest_difference),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit("Pair audit failed; see report for unexpected differences")


if __name__ == "__main__":
    main()
