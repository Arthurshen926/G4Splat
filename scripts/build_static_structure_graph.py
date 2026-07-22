#!/usr/bin/env python3
"""Build static structure units and a view graph after the Chart hard gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from outdoor.structure_graph import build_static_structure_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--mask-dataset-path", type=Path, required=True)
    parser.add_argument("--mask-indices", nargs="*", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--voxel-bins", type=int, default=40)
    parser.add_argument("--max-units", type=int, default=4096)
    parser.add_argument("--min-unit-views", type=int, default=2)
    parser.add_argument("--block-bins", type=int, default=4)
    parser.add_argument("--min-triangulation-angle-degrees", type=float, default=1.5)
    args = parser.parse_args()
    result = build_static_structure_graph(
        args.mast3r_scene,
        args.gate_report,
        args.mask_pickle,
        args.mask_dataset_path,
        args.output,
        mask_indices=args.mask_indices,
        stride=args.stride,
        bins=args.voxel_bins,
        max_units=args.max_units,
        min_unit_views=args.min_unit_views,
        block_bins=args.block_bins,
        min_angle_degrees=args.min_triangulation_angle_degrees,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
