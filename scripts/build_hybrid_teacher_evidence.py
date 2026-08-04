#!/usr/bin/env python
"""Build an immutable MASt3R-primary evidence store for the mixed Teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from outdoor.evidence_store import (  # noqa: E402
    EvidenceStoreBuilder,
    MAST3R_ONLY_GEOMETRY,
    MAST3R_PRIMARY_SFM_COVERAGE,
    build_track_evidence,
)
from outdoor.inverse_depth import (  # noqa: E402
    INVERSE_DEPTH_FUSION_VERSION,
    fuse_inverse_depth_directory,
)
from outdoor.mast3r_track_graph import validate_track_gate  # noqa: E402
from outdoor.scene_contract import build_scene_contract  # noqa: E402
from outdoor.task_semantics import build_task_semantic_manifest  # noqa: E402


def _optional(
    builder: EvidenceStoreBuilder,
    name: str,
    source_type: str,
    path: Path,
    measurement: str,
    covariance: str,
    *,
    semantic_role: str = "rigid_only",
    coordinate_frame: str | None = None,
) -> None:
    if path.is_file():
        builder.add_file(
            name,
            source_type,
            path,
            measurement=measurement,
            covariance=covariance,
            semantic_role=semantic_role,
            coordinate_frame=coordinate_frame
            or (
                "cambridge_fixed_world"
                if path.suffix != ".json"
                else "metadata_or_camera_pixels"
            ),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_array_index(
    destination: Path,
    *,
    schema_version: str,
    roots: list[Path],
) -> Path:
    records = []
    for root in roots:
        for path in sorted(root.glob("*.npy")):
            records.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not records:
        raise RuntimeError(
            f"No geometry arrays found for {schema_version}"
        )
    destination.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mask-pickle", type=Path, required=True)
    parser.add_argument("--tree-mask-pickle", type=Path, required=True)
    parser.add_argument("--mast3r-scene", type=Path, required=True)
    parser.add_argument("--mast3r-tracks", type=Path, required=True)
    parser.add_argument("--chart-consensus", type=Path, required=True)
    parser.add_argument("--dav2-root", type=Path)
    parser.add_argument(
        "--sfm-coverage-sparse",
        type=Path,
        help=(
            "Optional raw sparse/0 directory. Its points3D tracks are "
            "classified through the fixed cameras and may only fill "
            "MASt3R/MAtCha coverage gaps; they never replace the primary "
            "geometry factors or select training cameras."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    dataset = args.dataset.expanduser().resolve()
    mast3r = args.mast3r_scene.expanduser().resolve()
    tracks = args.mast3r_tracks.expanduser().resolve()
    chart_consensus = args.chart_consensus.expanduser().resolve()
    mask = args.mask_pickle.expanduser().resolve()
    tree_mask = args.tree_mask_pickle.expanduser().resolve()

    # This validation is intentionally repeated at the evidence boundary.
    # Supplying a one-observation sparse export under the expected filename
    # must never make it into a long Teacher run.
    track_gate = validate_track_gate(tracks)
    gate_path = output / "mast3r_track_gate.json"
    gate_path.write_text(
        json.dumps(track_gate, indent=2) + "\n", encoding="utf-8"
    )
    scene_contract = output / "scene_contract.json"
    semantic_contract = output / "task_semantics.json"
    build_scene_contract(
        dataset, scene_contract, mask_pickle=mask, split="database_train"
    )
    semantic = build_task_semantic_manifest(
        dataset,
        mask,
        semantic_contract,
        tree_mask_pickle=tree_mask,
        tree_support_policy="neutral",
    )
    semantic["representation_policy"] = (
        "native_chart_2dgs_plus_free_2dgs_plus_role_aware_3dgs_teacher"
    )
    semantic["implementation_scope"] = {
        "independent_semantic_fields": True,
        "separate_surface_foliage_sky_models": True,
        "tree_handling": (
            "static_skeleton_canonical_crown_sequence_conditioned_leaf"
        ),
    }
    semantic_contract.write_text(
        json.dumps(semantic, indent=2) + "\n", encoding="utf-8"
    )
    geometry_source = (
        MAST3R_PRIMARY_SFM_COVERAGE
        if args.sfm_coverage_sparse is not None
        else MAST3R_ONLY_GEOMETRY
    )
    builder = EvidenceStoreBuilder(
        output,
        dataset=dataset,
        scene_contract=scene_contract,
        semantic_contract=semantic_contract,
        geometry_source=geometry_source,
        final_model="hybrid_teacher",
    )
    builder.add_file(
        "scene_contract",
        "cambridge_fixed_camera",
        scene_contract,
        measurement="exact intrinsics, fixed poses and image identity",
        coordinate_frame="cambridge_fixed_world_and_camera",
        covariance="calibration_is_fixed_task_constraint",
        semantic_role="not_applicable",
        validity=(
            "fixed cameras are authoritative; points3D tracks are optional "
            "coverage evidence only"
            if args.sfm_coverage_sparse is not None
            else "cameras.bin/images.bin only; points3D forbidden"
        ),
    )
    builder.add_file(
        "task_semantics",
        "cambridge_masks",
        semantic_contract,
        measurement="separate RGB/geometry/plane/topology task fields",
        coordinate_frame="image_pixels",
        covariance="spatial_boundary_uncertainty",
        semantic_role="rigid_canopy_sky_transient",
        validity="real database masks",
    )
    builder.add_file(
        "mast3r_multiview_tracks",
        "mast3r_fixed_camera_track_graph",
        tracks,
        measurement=(
            "triangulated xyz, ragged pixel observations, role posterior"
        ),
        coordinate_frame="cambridge_fixed_world",
        covariance="fixed_ray_information_inverse",
        semantic_role="per_track_role_posterior",
        validity="reprojection_cycle_angle_and_sequence_gate",
    )
    builder.add_file(
        "mast3r_tracks",
        "mast3r_fixed_camera_track_graph_alias",
        tracks,
        measurement="teacher-compatible alias of multiview track archive",
        coordinate_frame="cambridge_fixed_world",
        covariance="fixed_ray_information_inverse",
        semantic_role="per_track_role_posterior",
        validity="same immutable file as mast3r_multiview_tracks",
    )
    builder.add_file(
        "mast3r_track_gate",
        "quality_gate",
        gate_path,
        measurement="hard pre-training track graph gate",
        coordinate_frame="metadata",
        covariance="not_applicable",
        semantic_role="audit",
        validity="passed_true",
    )
    if args.sfm_coverage_sparse is not None:
        sfm_sparse = args.sfm_coverage_sparse.expanduser().resolve()
        sfm_tracks = output / "colmap_tracks.npz"
        build_track_evidence(
            sfm_sparse,
            dataset,
            tree_mask,
            sfm_tracks,
            source_type="raw_sfm_coverage",
        )
        builder.add_file(
            "colmap_tracks",
            "raw_sfm_coverage_tracks",
            sfm_tracks,
            measurement=(
                "sparse xyz/rgb/track support/role posterior used only in "
                "uncovered MASt3R/MAtCha regions"
            ),
            coordinate_frame="cambridge_fixed_world",
            covariance="first_order_reprojection_covariance_diag",
            semantic_role="rigid_coverage_and_static_tree_track_candidates",
            validity=(
                "fixed-camera reprojection; multi-view role posterior; "
                "coverage-only ownership"
            ),
        )
        builder.add_file(
            "colmap_tracks_summary",
            "raw_sfm_coverage_tracks",
            sfm_tracks.with_suffix(".json"),
            measurement="content and selection audit for optional SfM tracks",
            coordinate_frame="metadata",
            covariance="documented_in_summary",
            semantic_role="audit",
            validity="generated together with colmap_tracks",
        )
    builder.add_file(
        "chart_crossview_consensus",
        "matcha_crossview_consensus",
        chart_consensus,
        measurement=(
            "camera-indexed consensus depth, independent support, "
            "per-pixel contradiction and trust"
        ),
        coordinate_frame="matcha_normalized_camera_depth",
        covariance="peer_reprojection_relative_depth_error",
        semantic_role="rigid_only",
        validity="minimum_two_independent_chart_views",
    )
    consensus_summary = chart_consensus.with_suffix(".json")
    if not consensus_summary.is_file():
        raise FileNotFoundError(consensus_summary)
    builder.add_file(
        "chart_crossview_consensus_summary",
        "matcha_crossview_consensus",
        consensus_summary,
        measurement="cross-view consensus audit and immutable input hashes",
        coordinate_frame="metadata",
        covariance="documented_in_summary",
        semantic_role="audit",
        validity="generated_with_consensus_archive",
    )

    chart_camera_path = mast3r / "cameras.json"
    pointmap_root = mast3r / "pointmaps"
    camera_payload = json.loads(chart_camera_path.read_text(encoding="utf-8"))
    pointmap_records = {}
    for value in camera_payload["filepaths"]:
        stem = Path(value).stem
        path = (pointmap_root / f"{stem}.json").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        pointmap_records[stem] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    pointmap_index = output / "mast3r_pointmap_index.json"
    pointmap_index.write_text(
        json.dumps(
            {
                "schema_version": "mast3r-pointmap-index-v1",
                "coordinate_frame": "cambridge_fixed_world",
                "camera_order": [
                    Path(value).stem
                    for value in camera_payload["filepaths"]
                ],
                "records": pointmap_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    builder.add_file(
        "mast3r_pointmap_index",
        "mast3r_dense_geometry",
        pointmap_index,
        measurement=(
            "content-addressed dense point/conf maps for rigid and "
            "sequence-local canopy evidence"
        ),
        coordinate_frame="cambridge_fixed_world",
        covariance="per_pixel_mast3r_confidence",
        semantic_role="rigid_and_sequence_local_canopy",
        validity="all camera-ordered pointmap files hashed",
    )

    _optional(
        builder,
        "chart_geometry",
        "matcha_chart_atlas",
        mast3r / "charts_data.npz",
        (
            "aligned chart points/depth/confidence/base inverse depth; "
            "metric values require division by embedded scale_factor"
        ),
        "chart_confidence_and_alignment_residual",
        coordinate_frame="matcha_normalized_world",
    )
    _optional(
        builder,
        "chart_cameras",
        "matcha_chart_atlas",
        mast3r / "cameras.json",
        "chart image identity and fixed-camera-aligned rays",
        "fixed_camera",
    )
    _optional(
        builder,
        "chart_alignment_gate",
        "matcha_chart_atlas",
        mast3r / "aligned_chart_conflict_gate.json",
        "per-chart conflict and validity gate",
        "cross_source_disagreement",
    )
    _optional(
        builder,
        "structure_graph",
        "mast3r_g4_structure",
        mast3r / "static_structure_graph.npz",
        (
            "multi-view structural units, facade blocks and planes; "
            "metric values require chart scale conversion"
        ),
        "cross_view_support_and_planarity",
        coordinate_frame="matcha_normalized_world",
    )
    _optional(
        builder,
        "structure_graph_manifest",
        "mast3r_g4_structure",
        mast3r / "static_structure_graph.json",
        "structure graph audit",
        "documented_per_unit",
    )
    plane = mast3r / "plane-refine-depths"
    # Never inherit an opaque frontend rho cache. Rebuild the cache inside the
    # immutable evidence store from source-specific metric contracts so its
    # hash proves the exact unit conversion consumed by this Teacher.
    inverse = output / "metric_inverse_depth"
    inverse_manifest = fuse_inverse_depth_directory(
        mast3r,
        plane,
        inverse,
        chart_consensus=chart_consensus,
    )
    if inverse_manifest["schema_version"] != INVERSE_DEPTH_FUSION_VERSION:
        raise RuntimeError("World-metric inverse-depth cache was not produced")
    plane_index = _write_array_index(
        output / "plane_array_index.json",
        schema_version="cambridge-plane-array-index-v1",
        roots=[plane],
    )
    inverse_index = _write_array_index(
        output / "inverse_depth_array_index.json",
        schema_version="cambridge-inverse-depth-array-index-v1",
        roots=[inverse],
    )
    _optional(
        builder,
        "plane_source_manifest",
        "g4_plane_factor",
        plane / "plane_residual_source_manifest.json",
        "plane core depth/normal/support/source arbitration",
        "plane_confidence_and_cross_view_support",
    )
    _optional(
        builder,
        "inverse_depth_manifest",
        "source_aware_inverse_depth",
        inverse / "inverse_depth_fusion_manifest.json",
        "rho mean/variance/source bits/support cache",
        "per_pixel_rho_variance",
    )
    builder.add_file(
        "plane_array_index",
        "g4_plane_factor_index",
        plane_index,
        measurement="content-addressed plane depth/confidence/support arrays",
        coordinate_frame="cambridge_fixed_camera_depth",
        covariance="per_plane_source_array",
        semantic_role="rigid_only",
        validity="every plane numpy array hashed",
    )
    builder.add_file(
        "inverse_depth_array_index",
        "source_aware_inverse_depth_index",
        inverse_index,
        measurement="content-addressed world-metric inverse-depth arrays",
        coordinate_frame="inverse_cambridge_fixed_camera_depth",
        covariance="per_pixel_rho_variance",
        semantic_role="rigid_only",
        validity="every inverse-depth numpy array hashed",
    )
    _optional(
        builder,
        "plane_depth_root_marker",
        "g4_plane_factor",
        plane / "plane_depth_frame000000.npy",
        "plane frame-array root marker in Cambridge fixed-camera depth",
        "per_pixel_plane_confidence",
        coordinate_frame="cambridge_fixed_camera_depth",
    )
    _optional(
        builder,
        "inverse_depth_root_marker",
        "source_aware_inverse_depth",
        inverse / "rho_mean_frame000000.npy",
        "inverse-depth frame-array root marker in Cambridge metric scale",
        "per_pixel_rho_variance",
        coordinate_frame="inverse_cambridge_fixed_camera_depth",
    )

    if args.dav2_root is not None:
        dav2 = args.dav2_root.expanduser().resolve()
        requested = {
            Path(record["image_name"]).stem
            for record in json.loads(scene_contract.read_text())["records"]
        }
        records = {}
        for path in sorted(dav2.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".npy", ".npz", ".png", ".tif", ".tiff"
            }:
                continue
            stem = path.stem
            for suffix in ("_depth", "_dav2", "_depth_anything_v2", "_pred"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            if stem not in requested or stem in records:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records[stem] = {
                "path": str(path),
                "sha256": digest,
                "size": path.stat().st_size,
            }
        if records:
            index = output / "dav2_index.json"
            index.write_text(
                json.dumps(
                    {
                        "version": "dav2-real-view-index-v2",
                        "root": str(dav2),
                        "scene_view_count": len(requested),
                        "indexed_view_count": len(records),
                        "records": records,
                    },
                    indent=2,
                )
                + "\n"
            )
            builder.add_file(
                "dav2_index",
                "depth_anything_v2",
                index,
                measurement="real-view monocular ordinal depth index",
                coordinate_frame="camera_rays",
                covariance="ordinal_only_not_metric",
                semantic_role="rigid_depth_order",
                validity="real_view_semantic_gate",
            )
    manifest = builder.write()
    # Keep the negative/limited dependency audit grep-able. The manifest
    # itself already hashes the optional track archive and its raw sources.
    sfm_coverage = args.sfm_coverage_sparse is not None
    manifest["forbidden_inputs_audit"] = {
        "points3D_bin_read": sfm_coverage,
        "colmap_tracks_read": sfm_coverage,
        "colmap_camera_or_pose_authority": False,
        "colmap_dense_geometry_authority": False,
        "sfm_track_usage_mode": "coverage_only" if sfm_coverage else "disabled",
        "historical_gaussian_initialization": False,
    }
    # Re-write through the builder is not possible after hashing; keep the
    # negative audit as an adjacent signed-by-artifact sidecar.
    (output / "forbidden_inputs_audit.json").write_text(
        json.dumps(manifest["forbidden_inputs_audit"], indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
