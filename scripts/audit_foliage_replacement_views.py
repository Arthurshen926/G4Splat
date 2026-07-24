#!/usr/bin/env python
"""Rebuild the legacy foliage responsibility audit over a wider view set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path[:0] = [str(REPO_ROOT), str(SURFEL_ROOT)]

from arguments import ModelParams, PipelineParams  # noqa: E402
from outdoor.foliage_responsibility import (  # noqa: E402
    audit_legacy_surface_responsibility,
)
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel  # noqa: E402
from outdoor.task_fields import OutdoorTaskFieldLookup  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(data_device="cpu", white_background=True)
    parser.add_argument("--structural-ply", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--task-semantic-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-views", type=int, default=48)
    parser.add_argument("--mandatory-indices", default="408,409,410")
    parser.add_argument("--minimum-support-views", type=int, default=3)
    parser.add_argument("--minimum-support-sequences", type=int, default=2)
    parser.add_argument("--minimum-canopy-responsibility", type=float, default=0.45)
    parser.add_argument("--maximum-rigid-responsibility", type=float, default=0.20)
    parser.add_argument("--support-contribution", type=float, default=0.20)
    parser.add_argument("--residual-quantile", type=float, default=0.40)
    args = parser.parse_args()
    return args, model.extract(args), pipeline.extract(args)


@torch.no_grad()
def main():
    args, dataset, _ = _parse_args()
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    structural = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, structural, shuffle=False)
    structural.load_ply(str(args.structural_ply.resolve()))
    views = scene.getTrainCameras()
    fields = OutdoorTaskFieldLookup(
        Path(dataset.source_path),
        args.tree_mask_pickle.resolve(),
        args.task_semantic_manifest.resolve(),
        max_cached_views=0,
    )
    empty = VolumetricFoliageModel(dataset.sh_degree).cuda()
    audit = audit_legacy_surface_responsibility(
        views,
        structural,
        empty,
        fields,
        background=torch.ones(3, device="cuda"),
        view_count=args.audit_views,
        minimum_support_views=args.minimum_support_views,
        minimum_support_sequences=args.minimum_support_sequences,
        mandatory_view_indices=tuple(
            int(value)
            for value in args.mandatory_indices.split(",")
            if value.strip()
        ),
        minimum_canopy_responsibility=args.minimum_canopy_responsibility,
        maximum_rigid_responsibility=args.maximum_rigid_responsibility,
        support_contribution=args.support_contribution,
        residual_quantile=args.residual_quantile,
    )
    path = audit.save(args.output.resolve())
    print(
        {
            "output": str(path),
            "candidate_count": int(audit.candidate_indices.numel()),
            "selected_view_count": len(audit.selected_view_indices),
            "selected_sequences": len(
                {
                    name.split("__", 1)[0]
                    for name in audit.selected_view_names
                }
            ),
            "thresholds": audit.thresholds,
        }
    )


if __name__ == "__main__":
    main()
