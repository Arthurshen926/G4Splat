#!/usr/bin/env python
"""Build a reproducible, diverse fixed-view Teacher evaluation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcha.cambridge_masks import CambridgeMaskLookup


def _pick_spaced(rows, *, score, count, excluded, minimum_index_gap):
    selected = []
    for row in sorted(rows, key=lambda item: float(item[score]), reverse=True):
        index = int(row["index"])
        if index in excluded:
            continue
        if any(abs(index - int(other["index"])) < minimum_index_gap for other in selected):
            continue
        selected.append(row)
        excluded.add(index)
        if len(selected) == count:
            break
    return selected


def _coverage_rows(rows, *, count, excluded):
    available = [row for row in rows if int(row["index"]) not in excluded]
    if not available or count <= 0:
        return []
    selected = []
    for offset in range(count):
        target = offset * (len(available) - 1) / max(count - 1, 1)
        candidates = [
            row
            for _, row in sorted(
                enumerate(available), key=lambda item: abs(item[0] - target)
            )
        ]
        row = next(
            row for row in candidates if int(row["index"]) not in excluded
        )
        selected.append(row)
        excluded.add(int(row["index"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-sequence", default="seq2")
    parser.add_argument("--high-tree-count", type=int, default=6)
    parser.add_argument("--high-boundary-count", type=int, default=6)
    parser.add_argument("--coverage-count", type=int, default=6)
    parser.add_argument("--minimum-index-gap", type=int, default=8)
    parser.add_argument("--mandatory-indices", default="632,633,634")
    args = parser.parse_args()

    cameras = json.loads(args.cameras_json.read_text())
    lookup = CambridgeMaskLookup(
        args.dataset.resolve(), args.tree_mask_pickle.resolve(), [0, 1, 2, 3]
    )
    rows = []
    for camera in cameras:
        image_name = str(camera["img_name"])
        if not image_name.startswith(f"{args.canonical_sequence}__"):
            continue
        height, width = int(camera["height"]), int(camera["width"])
        masks = lookup.get_index_masks(
            image_name, (0, 1, 2, 3), (height, width), torch.device("cpu")
        ).bool()
        static = masks[0] & masks[1] & masks[2]
        tree = static & ~masks[3]
        radius = max(2, round(min(height, width) * 0.015))
        kernel = radius * 2 + 1
        tree_4d = tree[None, None].float()
        dilated = F.max_pool2d(tree_4d, kernel, 1, radius)[0, 0].bool()
        eroded = ~F.max_pool2d((~tree)[None, None].float(), kernel, 1, radius)[
            0, 0
        ].bool()
        pixels = float(height * width)
        rows.append(
            {
                "index": int(camera["id"]),
                "image_name": image_name,
                "tree_fraction": float(tree.sum()) / pixels,
                "tree_boundary_inside_fraction": float((tree & ~eroded).sum()) / pixels,
                "rigid_boundary_outside_fraction": float(
                    (static & ~tree & dilated).sum()
                )
                / pixels,
                "static_valid_fraction": float(static.sum()) / pixels,
            }
        )
    rows.sort(key=lambda row: int(row["index"]))
    if not rows:
        raise RuntimeError(f"No cameras belong to {args.canonical_sequence!r}")

    by_index = {int(row["index"]): row for row in rows}
    mandatory = [int(value) for value in args.mandatory_indices.split(",") if value]
    missing = sorted(set(mandatory) - set(by_index))
    if missing:
        raise ValueError(f"Mandatory indices are outside the canonical sequence: {missing}")
    excluded = set(mandatory)
    selected = [
        {**by_index[index], "selection_reason": "historical_anchor"}
        for index in mandatory
    ]
    for reason, score, count in (
        ("high_tree_coverage", "tree_fraction", args.high_tree_count),
        (
            "high_tree_rigid_boundary",
            "rigid_boundary_outside_fraction",
            args.high_boundary_count,
        ),
    ):
        selected.extend(
            {
                **row,
                "selection_reason": reason,
            }
            for row in _pick_spaced(
                rows,
                score=score,
                count=count,
                excluded=excluded,
                minimum_index_gap=args.minimum_index_gap,
            )
        )
    selected.extend(
        {**row, "selection_reason": "sequence_coverage_control"}
        for row in _coverage_rows(
            rows, count=args.coverage_count, excluded=excluded
        )
    )
    selected.sort(key=lambda row: int(row["index"]))
    payload = {
        "schema_version": 1,
        "contract": (
            "scene_canonical_fixed_view_suite__historical_anchors_plus_"
            "spaced_high_tree_and_tree_rigid_boundary_hard_views_plus_"
            "sequence_coverage_controls"
        ),
        "canonical_sequence": args.canonical_sequence,
        "candidate_view_count": len(rows),
        "minimum_index_gap_within_ranked_category": args.minimum_index_gap,
        "selected_indices": [int(row["index"]) for row in selected],
        "selected_views": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
