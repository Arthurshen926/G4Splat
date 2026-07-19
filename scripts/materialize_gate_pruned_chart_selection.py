#!/usr/bin/env python3
"""Materialize the actually usable Chart subset after the hard geometry gate.

The aligned-chart gate deliberately keeps camera order stable and zeros rejected
geometry.  This adapter turns those gate decisions into an explicit selection
artifact for downstream plane/Gaussian code and coverage audits.  It is
intended for the safe case where a Chart is rejected for insufficient support:
the Chart is removed from the active set rather than silently remaining an
anchor, while the full 1,487-view dense reconstruction frontend is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.select_chart_views import normalize_required_name


def _name(value: str) -> str:
    return normalize_required_name(Path(str(value)).name)


def build_gate_pruned_selection(
    selection: dict,
    gate_report: dict,
    *,
    minimum_active: int,
) -> dict:
    """Return a selection whose active Chart set excludes every gate reject."""
    original_names = [_name(name) for name in selection.get("image_names", [])]
    if not original_names:
        raise ValueError("Selection has no image_names")
    original_indices = list(selection.get("image_idx", []))
    if original_indices and len(original_indices) != len(original_names):
        raise ValueError("Selection image_idx and image_names lengths differ")
    rejected = {
        _name(record["image_name"])
        for record in gate_report.get("records", [])
        if record.get("rejected")
    }
    active_pairs = [
        (name, original_indices[offset] if original_indices else None)
        for offset, name in enumerate(original_names)
        if name not in rejected
    ]
    active_names = [name for name, _ in active_pairs]
    if len(active_names) < int(minimum_active):
        raise RuntimeError(
            f"Only {len(active_names)} gate-valid Charts remain; require "
            f"at least {minimum_active}."
        )
    active_indices = [index for _, index in active_pairs] if original_indices else []
    excluded_names = [name for name in original_names if name in rejected]
    unreported = sorted(set(original_names) - {
        _name(record["image_name"]) for record in gate_report.get("records", [])
    })
    if unreported:
        raise RuntimeError(
            "Gate report is missing selected Chart(s): " + ", ".join(unreported[:5])
        )
    result = dict(selection)
    result.update(
        {
            "image_names": active_names,
            "selected_image_names": active_names,
            "n_images": len(active_names),
            "requested_n_images": len(active_names),
            "actual_n_images": len(active_names),
            "audit_active_from_selection": True,
            "gate_pruned_selection": {
                "version": 1,
                "mode": "hard_remove_all_aligned_chart_gate_rejections",
                "input_chart_count": len(original_names),
                "active_chart_count": len(active_names),
                "excluded_chart_names": excluded_names,
                "gate_rejected_names": sorted(rejected),
                "minimum_active": int(minimum_active),
                "zero_gate_reject_active": True,
            },
        }
    )
    if original_indices:
        result["image_idx"] = active_indices
        result["run_sfm_arg"] = "--image_idx " + " ".join(map(str, active_indices))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--minimum-active", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    gate_report = json.loads(args.gate_report.read_text(encoding="utf-8"))
    result = build_gate_pruned_selection(
        selection,
        gate_report,
        minimum_active=args.minimum_active,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "active_chart_count": result["gate_pruned_selection"]["active_chart_count"],
                "excluded_chart_names": result["gate_pruned_selection"]["excluded_chart_names"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
