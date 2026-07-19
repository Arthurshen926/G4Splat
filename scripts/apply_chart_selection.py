#!/usr/bin/env python3
"""Make an audited joint Chart subset an actual geometry input.

``select_quality_aware_charts.py`` intentionally writes a sidecar rather than
mutating chart geometry.  That is useful for comparison, but it is unsafe to
call the result "selected" if later plane and Gaussian stages still consume
every hard-gated Chart.  This adapter turns that selection into a hard mask in
``charts_data.npz`` while preserving a recoverable backup and an audit record.

It does not remove camera metadata or renumber tensors: non-selected Charts are
zeroed exactly like hard-gate rejects, so every downstream tensor remains in
the original camera order.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


GEOMETRY_KEYS = ("depths", "prior_depths", "pts", "pts3d", "confs")


def _name(value: str) -> str:
    return Path(str(value)).name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--gate-report",
        type=Path,
        required=True,
        help="Aligned-chart gate used to verify the selection never reactivates a reject.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scene = args.mast3r_scene.expanduser().resolve()
    charts_path = scene / "charts_data.npz"
    if not charts_path.is_file():
        raise FileNotFoundError(charts_path)
    selection = json.loads(args.selection.expanduser().read_text(encoding="utf-8"))
    selected_names = {_name(value) for value in selection.get("image_names", [])}
    if not selected_names:
        raise ValueError("Selection has no image_names")
    cameras = json.loads((scene / "cameras.json").read_text(encoding="utf-8"))
    names = [_name(value) for value in cameras["filepaths"]]
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = sorted(selected_names - set(names))
    if missing:
        raise RuntimeError(f"Selection contains Chart(s) absent from scene: {missing[:3]}")

    gate = json.loads(args.gate_report.expanduser().read_text(encoding="utf-8"))
    rejected = {
        _name(record["image_name"])
        for record in gate.get("records", [])
        if record.get("rejected")
    }
    conflict = sorted(selected_names & rejected)
    if conflict:
        raise RuntimeError(
            "Refusing to reactivate hard-gated Chart(s): " + ", ".join(conflict[:5])
        )

    selected_mask = np.asarray([name in selected_names for name in names], dtype=bool)
    backup = scene / "charts_data.pre_quality_selection.npz"
    if backup.exists() and not args.replace:
        raise FileExistsError(
            f"Selection backup already exists: {backup}. Use --replace only after auditing it."
        )
    if not backup.exists():
        shutil.copy2(charts_path, backup)

    with np.load(backup) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    count = len(names)
    if not all(
        key not in payload or payload[key].ndim == 0 or payload[key].shape[0] == count
        for key in GEOMETRY_KEYS
    ):
        raise RuntimeError("charts_data geometry tensors do not share camera dimension")

    gated = dict(payload)
    for key in GEOMETRY_KEYS:
        if key not in payload:
            continue
        value = np.asarray(payload[key]).copy()
        if value.ndim and value.shape[0] == count:
            value[~selected_mask] = 0
            if key == "confs":
                value[~selected_mask] = -2.0
        gated[key] = value
    gated["quality_selection_active"] = selected_mask
    temporary = charts_path.with_suffix(".quality-selection.tmp.npz")
    np.savez_compressed(temporary, **gated)
    temporary.replace(charts_path)

    active_confidence = np.asarray(gated["confs"])[selected_mask]
    active_pixel_counts = (active_confidence > 0).reshape(len(active_confidence), -1).sum(axis=1)
    if np.any(active_pixel_counts < 3):
        bad = [name for name, pixels in zip(np.asarray(names)[selected_mask], active_pixel_counts) if pixels < 3]
        raise RuntimeError(
            "Quality selection kept Chart(s) with no usable geometry: " + ", ".join(bad[:5])
        )
    report = {
        "version": 1,
        "mode": "hard_zero_nonselected_quality_aware_charts",
        "scene": str(scene),
        "selection": str(args.selection.expanduser().resolve()),
        "gate_report": str(args.gate_report.expanduser().resolve()),
        "input_chart_count": count,
        "active_chart_count": int(selected_mask.sum()),
        "active_chart_names": [name for name, active in zip(names, selected_mask) if active],
        "excluded_after_quality_count": int((~selected_mask).sum()),
        "hard_rejected_chart_count": len(rejected),
        "zero_depth_conflict_active": not bool(selected_names & rejected),
        "active_min_positive_geometry_pixels": int(active_pixel_counts.min()),
        "backup": str(backup),
    }
    report_path = args.report or scene / "quality_aware_chart_filter.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
