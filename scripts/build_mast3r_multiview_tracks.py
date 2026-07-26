#!/usr/bin/env python
"""Build and gate real multi-view tracks without any COLMAP point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.mast3r_track_graph import (  # noqa: E402
    build_multiview_tracks,
    validate_track_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=1.25)
    parser.add_argument("--minimum-overlap", type=float, default=0.03)
    parser.add_argument("--minimum-tracks", type=int, default=10_000)
    parser.add_argument("--minimum-global-rigid-tracks", type=int, default=2_000)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.replace:
        summary = json.loads(output.with_suffix(".json").read_text())
    else:
        summary = build_multiview_tracks(
            args.mast3r_scene,
            output,
            dataset=args.dataset,
            tree_mask_pickle=args.tree_mask_pickle,
            stride=args.stride,
            confidence_threshold=args.confidence_threshold,
            pair_graph_kwargs={"minimum_overlap": args.minimum_overlap},
        )
    gate = validate_track_gate(
        output,
        minimum_tracks=args.minimum_tracks,
        minimum_global_rigid_tracks=args.minimum_global_rigid_tracks,
    )
    payload = {"summary": summary, "gate": gate}
    output.with_name(output.stem + "_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
