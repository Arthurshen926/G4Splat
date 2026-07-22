#!/usr/bin/env python3
"""Select Global/Local Charts from the hard-gated static structure graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from outdoor.structural_selection import select_structural_charts, validate_structural_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-graph", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--target-scene-path", type=Path, required=True)
    parser.add_argument("--base-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-charts", type=int, default=24)
    parser.add_argument("--max-charts", type=int)
    parser.add_argument("--target-structural-coverage", type=float, default=0.92)
    parser.add_argument("--target-multiview-coverage", type=float, default=0.70)
    parser.add_argument("--min-block-views", type=int, default=2)
    parser.add_argument("--min-triangulation-angle-degrees", type=float, default=1.5)
    parser.add_argument("--target-triangulation-angle-degrees", type=float, default=8.0)
    args = parser.parse_args()
    select_structural_charts(
        args.structural_graph,
        args.gate_report,
        args.target_scene_path,
        args.output,
        base_selection=args.base_selection,
        min_charts=args.min_charts,
        max_charts=args.max_charts,
        target_structural_coverage=args.target_structural_coverage,
        target_multiview_coverage=args.target_multiview_coverage,
        min_block_views=args.min_block_views,
        min_triangulation_angle_degrees=args.min_triangulation_angle_degrees,
        target_triangulation_angle_degrees=args.target_triangulation_angle_degrees,
    )
    print(json.dumps(validate_structural_selection(args.output, args.gate_report), indent=2))


if __name__ == "__main__":
    main()
