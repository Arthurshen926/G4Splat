#!/usr/bin/env python
"""Build one immutable evidence store without training a Gaussian model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import (  # noqa: E402
    EvidenceStoreBuilder,
    build_track_evidence,
)
from outdoor.scene_contract import build_scene_contract  # noqa: E402
from outdoor.task_semantics import build_task_semantic_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--mast3r-scene", type=Path)
    parser.add_argument("--dav2-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def _add_optional(
    builder: EvidenceStoreBuilder,
    *,
    name: str,
    source_type: str,
    path: Path,
    measurement: str,
    covariance: str,
    semantic_role: str = "task_semantic_gate",
) -> None:
    if path.is_file():
        builder.add_file(
            name,
            source_type,
            path,
            measurement=measurement,
            covariance=covariance,
            semantic_role=semantic_role,
        )


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    dataset = args.dataset.expanduser().resolve()
    mask = args.mask_pickle.expanduser().resolve()
    tree_mask = args.tree_mask_pickle.expanduser().resolve()
    scene_contract = output / "scene_contract.json"
    semantic_contract = output / "task_semantics.json"
    build_scene_contract(
        dataset, scene_contract, mask_pickle=mask, split="database_train"
    )
    build_task_semantic_manifest(
        dataset,
        mask,
        semantic_contract,
        tree_mask_pickle=tree_mask,
        tree_support_policy="neutral",
    )
    builder = EvidenceStoreBuilder(
        output,
        dataset=dataset,
        scene_contract=scene_contract,
        semantic_contract=semantic_contract,
    )
    builder.add_file(
        "scene_contract",
        "fixed_camera",
        scene_contract,
        measurement="exact intrinsics, poses and image identity",
        coordinate_frame="COLMAP_world_and_camera",
        covariance="calibration_is_fixed_task_constraint",
        semantic_role="not_applicable",
        validity="all referenced database images present",
    )
    builder.add_file(
        "task_semantics",
        "cambridge_masks",
        semantic_contract,
        measurement="per-task semantic policy",
        coordinate_frame="image_pixels",
        covariance="boundary_uncertainty_field",
        semantic_role="rigid_canopy_sky_transient",
        validity="four real mask channels",
    )
    colmap_tracks = output / "colmap_tracks.npz"
    build_track_evidence(
        dataset / "sparse" / "0",
        dataset,
        tree_mask,
        colmap_tracks,
        source_type="colmap",
    )
    builder.add_file(
        "colmap_tracks",
        "colmap",
        colmap_tracks,
        measurement="sparse xyz/rgb/tracks/role posterior",
        covariance="first_order_reprojection_covariance_diag",
        semantic_role="per_track_role_posterior",
        validity="fixed_camera_reprojection_and_real_masks",
    )
    builder.add_file(
        "colmap_tracks_summary",
        "colmap",
        colmap_tracks.with_suffix(".json"),
        measurement="track evidence audit",
        coordinate_frame="metadata",
        covariance="documented_in_summary",
        semantic_role="role_count_audit",
        validity="generated_with_track_archive",
    )

    if args.mast3r_scene is not None:
        mast3r = args.mast3r_scene.expanduser().resolve()
        mast3r_sparse = mast3r / "sparse" / "0"
        if (mast3r_sparse / "points3D.bin").is_file():
            mast3r_tracks = output / "mast3r_tracks.npz"
            build_track_evidence(
                mast3r_sparse,
                dataset,
                tree_mask,
                mast3r_tracks,
                source_type="posed_mast3r",
            )
            builder.add_file(
                "mast3r_tracks",
                "posed_mast3r",
                mast3r_tracks,
                measurement="sparse xyz/rgb/tracks/role posterior",
                covariance="first_order_reprojection_covariance_diag",
                semantic_role="per_track_role_posterior",
                validity="fixed_camera_reprojection_and_real_masks",
            )
            builder.add_file(
                "mast3r_tracks_summary",
                "posed_mast3r",
                mast3r_tracks.with_suffix(".json"),
                measurement="track evidence audit",
                coordinate_frame="metadata",
                covariance="documented_in_summary",
                semantic_role="role_count_audit",
                validity="generated_with_track_archive",
            )
        _add_optional(
            builder,
            name="chart_geometry",
            source_type="mast3r_chart",
            path=mast3r / "charts_data.npz",
            measurement="aligned dense xyz/depth/confidence per Chart",
            covariance="confidence_and_cross_source_alignment_residual",
        )
        _add_optional(
            builder,
            name="chart_cameras",
            source_type="mast3r_chart",
            path=mast3r / "cameras.json",
            measurement="Chart-to-database image identity and camera",
            covariance="fixed_exact_K",
        )
        _add_optional(
            builder,
            name="chart_alignment_gate",
            source_type="mast3r_chart",
            path=mast3r / "aligned_chart_conflict_gate.json",
            measurement="per-Chart conflict/validity gate",
            covariance="cross_source_disagreement",
        )
        _add_optional(
            builder,
            name="structure_graph",
            source_type="cross_view_structure",
            path=mast3r / "static_structure_graph.npz",
            measurement="multi-view structural units and blocks",
            covariance="triangulation_support_and_view_count",
        )
        _add_optional(
            builder,
            name="structure_graph_manifest",
            source_type="cross_view_structure",
            path=mast3r / "static_structure_graph.json",
            measurement="structure graph audit",
            covariance="documented_per_unit",
        )
        plane = mast3r / "plane-refine-depths"
        _add_optional(
            builder,
            name="plane_source_manifest",
            source_type="multi_view_plane",
            path=plane / "plane_residual_source_manifest.json",
            measurement="plane depth/normal/support source contract",
            covariance="plane_confidence_and_cross_view_support",
        )
        inverse = mast3r / "inverse_depth_fusion"
        _add_optional(
            builder,
            name="inverse_depth_manifest",
            source_type="inverse_depth_cache",
            path=inverse / "inverse_depth_fusion_manifest.json",
            measurement="rho mean/variance/source bits/support cache",
            covariance="rho_variance_per_pixel",
        )
        if (plane / "plane_depth_frame000000.npy").is_file():
            builder.add_file(
                "plane_depth_root_marker",
                "multi_view_plane",
                plane / "plane_depth_frame000000.npy",
                measurement="plane array root marker; sibling frames are indexed by manifest",
                covariance="plane_confidence_frame arrays",
                semantic_role="rigid_only",
            )
        if (inverse / "rho_mean_frame000000.npy").is_file():
            builder.add_file(
                "inverse_depth_root_marker",
                "inverse_depth_cache",
                inverse / "rho_mean_frame000000.npy",
                measurement="inverse-depth cache root marker; siblings indexed by manifest",
                covariance="rho_variance_frame arrays",
            )

    if args.dav2_root is not None:
        dav2 = args.dav2_root.expanduser().resolve()
        candidates = sorted(
            path for path in dav2.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".npz"}
        )
        if candidates:
            builder.add_file(
                "dav2_manifest",
                "depth_anything_v2",
                candidates[0],
                measurement="monocular ordinal depth/index",
                coordinate_frame="camera_rays",
                covariance="ordinal_only_not_metric",
                semantic_role="rigid_depth_order",
                validity="real_view_semantic_gate",
            )

    manifest = builder.write()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
