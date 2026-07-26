#!/usr/bin/env python
"""Create disjoint rigid-surface and layered-foliage seeds from evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import load_evidence_store  # noqa: E402
from outdoor.role_aware_initialization import (  # noqa: E402
    INITIALIZATION_VERSION,
    build_foliage_seed,
    build_surface_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-chart-seeds", type=int, default=80_000)
    parser.add_argument("--chart-seeds-per-view", type=int, default=2000)
    parser.add_argument("--maximum-foliage-voxels", type=int, default=400_000)
    parser.add_argument("--selected-foliage-views", type=int, default=64)
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    store = load_evidence_store(args.evidence_store)
    surface_path = output / "surface_seed.npz"
    surface = build_surface_seed(
        args.evidence_store,
        surface_path,
        chart_seeds_per_view=args.chart_seeds_per_view,
        maximum_chart_seeds=args.maximum_chart_seeds,
        seed=args.seed,
    )
    foliage_path = output / "foliage_seed_gaussians.pth"
    foliage = build_foliage_seed(
        args.evidence_store,
        surface_path,
        foliage_path,
        voxel_size=args.voxel_size,
        maximum_voxels=args.maximum_foliage_voxels,
        selected_view_count=args.selected_foliage_views,
        seed=args.seed,
    )
    manifest = {
        "version": INITIALIZATION_VERSION,
        "evidence_hash": store["evidence_hash"],
        "surface_seed": str(surface_path),
        "foliage_seed": str(foliage_path),
        "surface": surface,
        "foliage": foliage,
        "ownership": {
            "surface_canopy_seed_count": 0,
            "trunk_branch": "static_skeleton_3dgs",
            "canonical_crown": "ray_depth_visual_hull_3dgs",
        },
        "historical_model_initialization": False,
    }
    (output / "initialization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
