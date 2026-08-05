import json

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.spatial import cKDTree

from outdoor.chart_surface_model import (
    ChartSurfaceModel,
    LearnableInverseDepthAtlas,
)
from outdoor.evidence_store import ROLE_CANOPY, ROLE_RIGID
from outdoor.role_aware_initialization import (
    _ProjectedRigidInitializationPosterior,
    _align_tracks_to_local_hull_instances,
    _apply_missing_pointmap_posterior_fallback,
    _balanced_single_camera_sample_cap,
    _balanced_pointmap_seed_cap,
    _chart_hypothesis_authority,
    _chart_overlap_primary_ownership,
    _source_local_surface_frames,
    _compact_padded_camera_metadata,
    _cross_sequence_ray_posterior,
    _different_sequence_nearest_distance,
    _effective_posterior_budget,
    _filter_sequence_local_scene_envelope,
    _foliage_posterior_view_pool,
    _line_supported_tree_tracks,
    _local_replacement_groups,
    _mast3r_tree_tracks,
    _nearest_cross_label_neighbour,
    _load_pointmap_cross_sequence_posterior,
    _native_pointmap_shape,
    _nonchart_pointmap_foliage_samples,
    _ownership_isolated_track_frames,
    _pointmap_fixed_camera_validity,
    _pointmap_cross_sequence_authority,
    _pointmap_surface_frames,
    _rigid_scene_envelope_mask,
    _sampled_point_footprint_multiplier,
    _select_dav2_foliage_views,
    _select_foliage_posterior_views,
    _selection_mask,
    _surface_frames,
    _uniform_confidence_samples,
    SOURCE_SURFACE_DAV2,
    sparse_rigid_depth_maps,
)
from outdoor.projected_role_posterior import (
    PROJECTED_RIGID_POSTERIOR_VERSION,
)


def test_mast3r_tree_track_archive_members_are_decompressed_once(
    monkeypatch,
):
    arrays = {
        "camera_names": np.asarray(["a.png", "b.png", "c.png"]),
        "observation_offsets": np.asarray([0, 3], dtype=np.int64),
        "observation_camera_indices": np.asarray(
            [0, 1, 2], dtype=np.int32
        ),
        "observation_pixels": np.asarray(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32
        ),
        "observation_camera_depth": np.asarray(
            [4.0, 4.1, 4.2], dtype=np.float32
        ),
        "camera_image_sizes": np.asarray(
            [[8, 6], [8, 6], [8, 6]], dtype=np.int32
        ),
        "track_id": np.asarray([17], dtype=np.int64),
        "xyz": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "rgb": np.asarray([[10, 20, 30]], dtype=np.uint8),
        "reprojection_error": np.asarray([0.2], dtype=np.float32),
        "sequence_count": np.asarray([3], dtype=np.int16),
        "role_probabilities": np.asarray(
            [[0.0, 0.8, 0.0, 0.2]], dtype=np.float32
        ),
        "valid_observation_count": np.asarray([3], dtype=np.int16),
    }

    class CountingArchive:
        def __init__(self):
            self.reads = {key: 0 for key in arrays}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __getitem__(self, key):
            self.reads[key] += 1
            return arrays[key]

    archive = CountingArchive()
    monkeypatch.setattr(
        "outdoor.role_aware_initialization.np.load",
        lambda *_args, **_kwargs: archive,
    )
    images = {
        index + 1: {"name": name}
        for index, name in enumerate(("a.png", "b.png", "c.png"))
    }

    tracks = _mast3r_tree_tracks(
        "unused.npz",
        images,
        minimum_track_observations=3,
        maximum_reprojection_error=1.0,
    )

    assert len(tracks) == 1
    assert tracks[0]["id"] == 17
    assert tracks[0]["tree_image_ids"].tolist() == [1, 2, 3]
    assert all(read_count == 1 for read_count in archive.reads.values())


def test_projected_pointmap_capacity_authority_remains_continuous():
    authority = _pointmap_cross_sequence_authority(
        np.asarray([True, False, False, False]),
        np.asarray([0.01, 0.9, 0.2, 0.0], dtype=np.float32),
    )
    assert authority.tolist() == pytest.approx([1.0, 0.9, 0.2, 0.0])


def test_projected_rigid_initialization_requires_local_depth_agreement(
    tmp_path,
):
    archive = tmp_path / "projected.npz"
    tracks = tmp_path / "tracks.npz"
    np.savez_compressed(
        tracks,
        track_id=np.asarray([11, 22], dtype=np.int64),
        rgb=np.asarray([[64, 128, 192], [255, 0, 0]], dtype=np.uint8),
    )
    np.savez_compressed(
        archive,
        schema_version=np.asarray(PROJECTED_RIGID_POSTERIOR_VERSION),
        camera_names=np.asarray(["frame.png"]),
        camera_image_sizes=np.asarray([[8, 6]], dtype=np.int32),
        offsets=np.asarray([0, 2], dtype=np.int64),
        pixels=np.asarray([[2, 2], [6, 4]], dtype=np.uint16),
        geometry_support_probability=np.asarray(
            [0.8, 0.6], dtype=np.float16
        ),
        visible_observation_probability=np.asarray(
            [0.8, 0.0], dtype=np.float16
        ),
        camera_depth=np.asarray([5.0, 8.0], dtype=np.float32),
        source_track_id=np.asarray([11, 22], dtype=np.int64),
    )
    candidate_depth = np.full((6, 8), 20.0, dtype=np.float32)
    candidate_depth[1:4, 1:4] = 5.05
    posterior = _ProjectedRigidInitializationPosterior(archive, tracks)
    probability = posterior.depth_consistent_probability(
        "frame.png", candidate_depth, footprint_radius=1
    )

    assert probability[2, 2] == pytest.approx(0.8, abs=1e-3)
    assert np.count_nonzero(probability) == 9
    assert probability[4, 6] == 0.0
    assert not probability[[0, 5], :].any()

    geometry_only = posterior.depth_consistent_probability(
        "frame.png",
        candidate_depth,
        footprint_radius=1,
        require_current_visibility=False,
    )
    assert geometry_only[4, 6] == 0.0
    candidate_depth[3:6, 5:8] = 8.05
    geometry_only = posterior.depth_consistent_probability(
        "frame.png",
        candidate_depth,
        footprint_radius=1,
        require_current_visibility=False,
    )
    visible_only = posterior.depth_consistent_probability(
        "frame.png", candidate_depth, footprint_radius=1
    )
    assert geometry_only[4, 6] == pytest.approx(0.6, abs=1e-3)
    assert visible_only[4, 6] == 0.0
    geometry, visible, canonical_rgb = posterior.depth_consistent_fields(
        "frame.png", candidate_depth, footprint_radius=1
    )
    assert geometry[4, 6] == pytest.approx(0.6, abs=1e-3)
    assert visible[4, 6] == 0.0
    assert canonical_rgb[4, 6].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_projected_rigid_initialization_missing_camera_is_no_evidence(
    tmp_path,
):
    archive = tmp_path / "projected.npz"
    tracks = tmp_path / "tracks.npz"
    np.savez_compressed(
        tracks,
        track_id=np.empty(0, dtype=np.int64),
        rgb=np.empty((0, 3), dtype=np.uint8),
    )
    np.savez_compressed(
        archive,
        schema_version=np.asarray(PROJECTED_RIGID_POSTERIOR_VERSION),
        camera_names=np.asarray(["frame.png"]),
        camera_image_sizes=np.asarray([[8, 6]], dtype=np.int32),
        offsets=np.asarray([0, 0], dtype=np.int64),
        pixels=np.empty((0, 2), dtype=np.uint16),
        geometry_support_probability=np.empty(0, dtype=np.float16),
        visible_observation_probability=np.empty(0, dtype=np.float16),
        camera_depth=np.empty(0, dtype=np.float32),
        source_track_id=np.empty(0, dtype=np.int64),
    )
    posterior = _ProjectedRigidInitializationPosterior(archive, tracks)
    result = posterior.depth_consistent_probability(
        "other.png", np.ones((3, 4), dtype=np.float32)
    )
    assert result.shape == (3, 4)
    assert not result.any()


def test_source_local_surface_frames_freeze_primary_parameterization():
    primary = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    coverage = np.asarray(
        [
            [0.001, 0.001, 0.01],
            [0.002, 0.001, 0.02],
            [0.001, 0.002, 0.03],
        ],
        dtype=np.float32,
    )
    primary_scale, primary_rotation, primary_normal = (
        _source_local_surface_frames(
            primary,
            np.full(4, 0.003, dtype=np.float32),
            np.ones(4, dtype=np.int8),
            np.ones(4, dtype=bool),
            maximum_scale=0.08,
        )
    )
    combined = np.concatenate([primary, coverage])
    scale, rotation, normal = _source_local_surface_frames(
        combined,
        np.full(7, 0.003, dtype=np.float32),
        np.asarray([1, 1, 1, 1, 0, 0, 0], dtype=np.int8),
        np.ones(7, dtype=bool),
        maximum_scale=0.08,
    )

    np.testing.assert_array_equal(scale[:4], primary_scale)
    np.testing.assert_array_equal(rotation[:4], primary_rotation)
    np.testing.assert_array_equal(normal[:4], primary_normal)
from scripts.initialize_unified_outdoor_scene import (
    _validate_pointmap_posterior_contract,
    _validated_reused_foliage,
)
from scripts.augment_temporal_dav2_foliage import _recover_alignments
from scene.gaussian_model import GaussianModel


def test_initializer_rejects_base_pointmap_store_without_posterior(tmp_path):
    index = tmp_path / "pointmaps.json"
    index.write_text(
        json.dumps({"schema_version": "mast3r-pointmap-index-v1"})
    )
    store = {
        "geometry_source": "mast3r_only",
        "artifacts": [
            {"name": "mast3r_pointmap_index", "path": str(index)}
        ],
    }
    with pytest.raises(RuntimeError, match="derived Evidence Store"):
        _validate_pointmap_posterior_contract(store)


def test_initializer_accepts_nonempty_cross_sequence_posterior(tmp_path):
    index = tmp_path / "pointmaps.json"
    index.write_text(
        json.dumps(
            {
                "cross_sequence_posterior": {
                    "schema_version": (
                        "mast3r-cross-sequence-pointmap-posterior-v1"
                    ),
                    "aggregate": {
                        "supported_pixels": 23,
                        "supported_fraction": 0.25,
                        "mean_precision": 0.4,
                    },
                }
            }
        )
    )
    audit = _validate_pointmap_posterior_contract(
        {
            "geometry_source": "mast3r_primary_sfm_coverage",
            "artifacts": [
                {"name": "mast3r_pointmap_index", "path": str(index)}
            ],
        }
    )
    assert audit["present"] is True
    assert audit["supported_pixels"] == 23
    assert audit["mean_precision"] == pytest.approx(0.4)


def test_chart_overlap_partition_keeps_one_metric_renderer_owner():
    xyz = np.asarray(
        [
            [0.000, 0.0, 3.0],
            [0.006, 0.0, 3.0],
            [1.000, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    owner, audit = _chart_overlap_primary_ownership(
        xyz,
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
        np.asarray([0.6, 0.9, 1.0], dtype=np.float32),
        np.asarray([True, True, True]),
        np.asarray([True, True, True]),
        distance_threshold=0.012,
    )
    assert owner.tolist() == [False, True, True]
    assert audit["overlap_components"] == 1
    assert audit["secondary_renderer_rows_suppressed"] == 1


def test_progressive_cross_label_neighbour_matches_full_knn():
    rng = np.random.default_rng(7)
    xyz = np.concatenate(
        [
            rng.normal((0.0, 0.0, 0.0), 0.01, (24, 3)),
            rng.normal((0.04, 0.0, 0.0), 0.01, (24, 3)),
            rng.normal((0.08, 0.0, 0.0), 0.01, (24, 3)),
        ]
    ).astype(np.float32)
    labels = np.repeat(np.arange(3), 24)
    distance, index = _nearest_cross_label_neighbour(
        xyz, labels, maximum_neighbours=64
    )
    full_distance, full_index = cKDTree(xyz).query(xyz, k=64)
    expected_distance = np.full(len(xyz), np.inf)
    expected_index = np.full(len(xyz), -1, dtype=np.int64)
    for row in range(len(xyz)):
        for rank in range(1, 64):
            candidate = full_index[row, rank]
            if labels[candidate] != labels[row]:
                expected_distance[row] = full_distance[row, rank]
                expected_index[row] = candidate
                break
    np.testing.assert_allclose(distance, expected_distance, rtol=1e-6)
    np.testing.assert_array_equal(index, expected_index)


def test_temporal_dav2_alignment_recovers_half_open_pixel_centres(tmp_path):
    rows = np.asarray([3, 9, 17, 28, 41, 56, 73, 91, 108, 126])
    columns = np.asarray([5, 19, 34, 52, 77, 103, 131, 164, 199, 231])
    grid_y, grid_x = np.indices((135, 240))
    ordinal = (1.0 + 0.002 * grid_x + 0.003 * grid_y).astype(np.float32)
    depth_path = tmp_path / "frame.npy"
    np.save(depth_path, ordinal)
    alpha, beta = 0.025, 0.075
    depth = 1.0 / (alpha + beta / ordinal[rows, columns])
    payload = {
        "initialization_source": torch.full((len(rows),), 3, dtype=torch.int8),
        "observation_camera_ids": torch.ones((len(rows), 1), dtype=torch.int32),
        "observation_uv": torch.from_numpy(
            np.column_stack(
                [
                    (columns.astype(np.float32) + 0.5) / 240.0,
                    (rows.astype(np.float32) + 0.5) / 135.0,
                ]
            )[:, None]
        ),
        "observation_depth": torch.from_numpy(depth.astype(np.float32))[:, None],
        "reprojection_error": torch.full((len(rows),), 0.05),
    }
    recovered = _recover_alignments(
        payload,
        [{"image_id": 1, "image_name": "seq1__frame00010.png"}],
        {"seq1__frame00010": {"path": str(depth_path)}},
    )

    assert set(recovered) == {"seq1__frame00010.png"}
    assert recovered["seq1__frame00010.png"]["alpha"] == pytest.approx(
        alpha, abs=2e-6
    )
    assert recovered["seq1__frame00010.png"]["beta"] == pytest.approx(
        beta, abs=2e-6
    )


def test_zero_foliage_view_limit_reaches_all_fixed_camera_selection_layer():
    views = [{"image_id": value} for value in range(30)]
    observed = {2, 7, 11}

    pool, limit, contract = _foliage_posterior_view_pool(
        views, observed, 0
    )
    assert [row["image_id"] for row in pool] == list(range(30))
    assert limit == 30
    assert contract == "all_fixed_database_cameras"
    selected = _select_foliage_posterior_views(
        pool,
        limit,
        contract,
        support_by_instance={
            0: {2: 5, 7: 4, 11: 3},
        },
    )
    assert [row["image_id"] for row in selected] == list(range(30))

    pool, limit, contract = _foliage_posterior_view_pool(
        views, observed, 2
    )
    assert [row["image_id"] for row in pool] == [2, 7, 11]
    assert limit == 3
    assert contract == "bounded_observed_diverse_cameras"


def test_zero_foliage_view_limit_preserves_all_dav2_depth_cameras():
    views = [
        {
            "image_id": value,
            "image_name": f"seq{value % 2}__frame{value:05d}.png",
            "sequence_id": f"seq{value % 2}",
            "canopy_fraction": 0.1,
            "camera_center": np.asarray([value, 0.0, 0.0]),
        }
        for value in range(100)
    ]

    selected, contract = _select_dav2_foliage_views(views, 0)
    assert [row["image_id"] for row in selected] == list(range(100))
    assert contract == "all_fixed_database_cameras"

    selected, contract = _select_dav2_foliage_views(views, 12)
    assert len(selected) == 72
    assert contract == "bounded_sequence_balanced_cameras"


def test_dav2_global_cap_is_camera_balanced_not_prefix_truncated():
    samples = [
        {
            "tree_image_ids": np.asarray([image_id], dtype=np.int32),
            "sample_index": sample_index,
        }
        for image_id in (2, 7, 11)
        for sample_index in range(5)
    ]

    retained, audit = _balanced_single_camera_sample_cap(
        samples, 8, source_name="DAV2"
    )

    retained_ids = [
        int(np.asarray(row["tree_image_ids"]).reshape(-1)[0])
        for row in retained
    ]
    assert set(retained_ids) == {2, 7, 11}
    assert {
        image_id: retained_ids.count(image_id)
        for image_id in set(retained_ids)
    } == {2: 3, 7: 3, 11: 2}
    assert audit == {
        "raw_samples": 15,
        "retained_samples": 8,
        "sampled_views": 3,
        "camera_budget_truncated": True,
    }


def test_dav2_global_cap_rejects_false_all_camera_provenance():
    samples = [
        {
            "tree_image_ids": np.asarray([image_id], dtype=np.int32),
        }
        for image_id in range(5)
    ]

    with pytest.raises(RuntimeError, match="one observation per"):
        _balanced_single_camera_sample_cap(
            samples, 4, source_name="DAV2"
        )


def test_replacement_groups_are_local_not_merely_same_tree():
    centers = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.1, 0.0, 0.0],
        ]
    )
    roles = np.asarray([0, 0, 2, 2, 2], dtype=np.int8)
    # Rows 0, 2 and 3 deliberately share a coarse tree instance. Row 4 is
    # spatially local to canonical row 1 but belongs to another instance, so
    # it must remain owner-local rather than borrowing another tree's state.
    instances = np.asarray([7, 8, 7, 7, 9], dtype=np.int32)

    groups, audit = _local_replacement_groups(
        centers,
        roles,
        instances,
        maximum_center_distance=0.36,
    )

    assert groups.tolist() == [0, 1, 0, -1, -1]
    assert audit["bound_dynamic_count"] == 1
    assert audit["unbound_dynamic_count"] == 2
    assert audit["cross_instance_binding_count"] == 0
    assert audit["dynamic_instance_without_canonical_count"] == 1
    assert max(audit["binding_distance_quantiles"]) <= 0.36


def test_chart_observations_are_separate_from_renderer_birth_authority():
    independent = np.asarray([True, False, True, False])
    confirmed = np.asarray([True, False, False, False])

    admitted, persistent, opacity = _chart_hypothesis_authority(
        independent,
        confirmed,
    )

    assert admitted.tolist() == [True, False, True, False]
    assert persistent.tolist() == [True, False, False, False]
    np.testing.assert_allclose(opacity, [0.05, 0.025, 0.025, 0.025])


def test_dav2_capacity_uses_continuous_posterior_effective_sample_size():
    assert SOURCE_SURFACE_DAV2 == GaussianModel.SOURCE_DAV2_RIGID_HOLE
    assert SOURCE_SURFACE_DAV2 != GaussianModel.SOURCE_FREE_RESIDUAL

    budget, effective_count = _effective_posterior_budget(
        np.ones(10),
        maximum_total=100,
    )
    assert budget == 10
    assert effective_count == pytest.approx(10.0)

    budget, effective_count = _effective_posterior_budget(
        np.asarray([1.0, 0.0, np.nan, -1.0]),
        maximum_total=100,
    )
    assert budget == 1
    assert effective_count == pytest.approx(1.0)

    budget, effective_count = _effective_posterior_budget(
        np.ones(10),
        maximum_total=4,
    )
    assert budget == 4
    assert effective_count == pytest.approx(10.0)

    assert _effective_posterior_budget(
        np.zeros(10),
        maximum_total=100,
    ) == (0, 0.0)


def test_dav2_cross_sequence_rays_form_continuous_geometry_posterior():
    centers = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    directions = np.asarray([[1.0, 0.0, 5.0], [-1.0, 0.0, 5.0]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    depth = np.sqrt(26.0)
    xyz = centers + depth * directions

    fused, posterior, audit = _cross_sequence_ray_posterior(
        xyz,
        centers,
        directions,
        np.asarray([1, 0]),
    )

    assert np.all(posterior > 0.8)
    np.testing.assert_allclose(fused, [[0, 0, 5], [0, 0, 5]], atol=1e-5)
    assert audit["paired"] == 2
    assert audit["finite_triangulation"] == 2

    unchanged, unsupported, audit = _cross_sequence_ray_posterior(
        xyz,
        centers,
        directions,
        np.asarray([-1, -1]),
    )
    np.testing.assert_allclose(unchanged, xyz)
    np.testing.assert_allclose(unsupported, 0)
    assert audit["paired"] == 0


def test_chart_confirmation_requires_independent_support():
    with np.testing.assert_raises_regex(
        ValueError, "lack independent support"
    ):
        _chart_hypothesis_authority(
            np.asarray([False]),
            np.asarray([True]),
        )


def test_chart_scene_envelope_rejects_only_catastrophic_depth_rays():
    rigid = np.column_stack(
        [
            np.linspace(-10.0, 10.0, 1000),
            np.zeros(1000),
            np.zeros(1000),
        ]
    )
    candidates = np.asarray(
        [
            [12.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
        ]
    )

    keep, audit = _rigid_scene_envelope_mask(candidates, rigid)

    assert keep.tolist() == [True, False]
    assert audit["maximum_radius"] > audit["reference_radius"]
    assert audit["rejected"] == 1


def test_foliage_reuse_requires_same_evidence_and_rgb(tmp_path):
    foliage_seed = tmp_path / "foliage.pth"
    foliage_seed.write_bytes(b"immutable foliage")
    rgb = {
        "image_root": "/rgb",
        "image_count": 3,
        "name_set_sha256": "names",
        "content_mapping_sha256": "content",
        "canonical_image_size_wh": [640, 360],
    }
    manifest = {
        "evidence_hash": "evidence",
        "historical_model_initialization": False,
        "rgb_source": rgb,
        "foliage_seed": str(foliage_seed),
        "foliage": {
            "evidence_hash": "evidence",
            "geometry_source": "mast3r_only",
            "historical_trained_ply_used": False,
        },
    }
    manifest_path = tmp_path / "initialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    path, _, provenance = _validated_reused_foliage(
        manifest_path,
        evidence_hash="evidence",
        rgb_contract=rgb,
    )

    assert path == foliage_seed
    assert provenance["surface_front_end_is_only_changed_variable"]
    with pytest.raises(RuntimeError, match="different Evidence Store"):
        _validated_reused_foliage(
            manifest_path,
            evidence_hash="other",
            rgb_contract=rgb,
        )


def test_padded_camera_metadata_is_compacted_without_changing_rows():
    foliage = {
        "support_camera_ids": np.asarray(
            [[-1, 7, -1, 3], [2, -1, -1, -1]], dtype=np.int32
        ),
        "support_view_count": np.asarray([2, 1], dtype=np.int16),
        "observation_camera_ids": np.asarray(
            [[-1, 7, -1, 3], [-1, 2, -1, -1]], dtype=np.int32
        ),
        "observation_uv": np.asarray(
            [
                [[np.nan, np.nan], [0.7, 0.1], [np.nan, np.nan], [0.3, 0.2]],
                [[np.nan, np.nan], [0.2, 0.4], [np.nan, np.nan], [np.nan, np.nan]],
            ],
            dtype=np.float32,
        ),
        "observation_depth": np.asarray(
            [[np.nan, 7.0, np.nan, 3.0], [np.nan, 2.0, np.nan, np.nan]],
            dtype=np.float32,
        ),
    }

    audit = _compact_padded_camera_metadata(foliage)

    assert foliage["support_camera_ids"].tolist() == [[7, 3], [2, -1]]
    assert foliage["observation_camera_ids"].tolist() == [[7, 3], [2, -1]]
    np.testing.assert_allclose(
        foliage["observation_uv"][0],
        np.asarray([[0.7, 0.1], [0.3, 0.2]]),
    )
    np.testing.assert_allclose(
        foliage["observation_depth"][0],
        np.asarray([7.0, 3.0]),
    )
    assert audit["support_slot_width_before"] == 4
    assert audit["support_slot_width_after"] == 2
    assert audit["observation_slot_width_after"] == 2
    assert audit["camera_metadata_bytes_removed"] > 0


def test_hull_instances_do_not_absorb_distant_dynamic_observations():
    hull = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.5, 0.0, 0.0],
        ]
    )
    # One upstream label contains two distant accepted islands. The local
    # authority pass must split them and must not attach a third distant tree
    # merely because one sparse native label happened to be nearest.
    hull_ids = np.zeros(4, dtype=np.int32)
    tracks = np.asarray(
        [
            [0.2, 0.0, 0.0],
            [20.2, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.4, 0.0, 0.0],
        ]
    )
    local_hull, track_ids, audit = (
        _align_tracks_to_local_hull_instances(
            hull,
            hull_ids,
            tracks,
            association_radius=1.0,
            maximum_component_extent=12.0,
        )
    )
    assert local_hull[0] == local_hull[1]
    assert local_hull[2] == local_hull[3]
    assert local_hull[0] != local_hull[2]
    assert track_ids[0] == local_hull[0]
    assert track_ids[1] == local_hull[2]
    assert track_ids[2] == track_ids[3]
    assert track_ids[2] not in set(local_hull.tolist())
    assert audit["associated_track_count"] == 2
    assert audit["dynamic_only_instance_count"] == 1


def test_incremental_renderer_basis_does_not_relabel_persistent_tracks():
    hull = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    hull_ids = np.zeros(2, dtype=np.int32)
    persistent = np.asarray(
        [[0.2, 0.0, 0.0], [20.0, 0.0, 0.0], [20.4, 0.0, 0.0]]
    )
    local_base, persistent_ids, _ = _align_tracks_to_local_hull_instances(
        hull,
        hull_ids,
        persistent,
        association_radius=1.0,
        maximum_component_extent=12.0,
    )
    incremental = np.asarray(
        [
            [20.2, 0.1, 0.0],
            [40.0, 0.0, 0.0],
            [40.3, 0.0, 0.0],
        ]
    )
    local_full, full_ids, audit = _align_tracks_to_local_hull_instances(
        hull,
        hull_ids,
        np.concatenate([persistent, incremental]),
        association_radius=1.0,
        maximum_component_extent=12.0,
        persistent_track_count=len(persistent),
    )

    np.testing.assert_array_equal(local_full, local_base)
    np.testing.assert_array_equal(full_ids[: len(persistent)], persistent_ids)
    assert full_ids[len(persistent)] == persistent_ids[1]
    assert full_ids[-1] == full_ids[-2]
    assert full_ids[-1] not in set(persistent_ids.tolist())
    assert audit["incremental_inherited_track_count"] == 1
    assert audit["incremental_new_instance_count"] == 1


def test_dense_renderer_bandwidth_does_not_change_persistent_track_frames():
    persistent = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.06, 0.01, 0.0],
            [0.09, 0.01, 0.01],
        ]
    )
    base_scale, base_rotation, base_linearity = (
        _ownership_isolated_track_frames(
            persistent,
            np.zeros(len(persistent), dtype=bool),
            np.full(len(persistent), 7, dtype=np.int32),
            np.full(len(persistent), -1, dtype=np.int32),
            0.008,
            0.035,
            0.10,
        )[:3]
    )
    dense = np.asarray(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.003, 0.003, 0.003],
        ]
    )
    xyz = np.concatenate([persistent, dense])
    dense_mask = np.arange(len(xyz)) >= len(persistent)
    scale, rotation, linearity, audit = _ownership_isolated_track_frames(
        xyz,
        dense_mask,
        np.full(len(xyz), 7, dtype=np.int32),
        np.concatenate(
            [
                np.full(len(persistent), -1, dtype=np.int32),
                np.full(len(dense), 1949, dtype=np.int32),
            ]
        ),
        0.008,
        0.035,
        0.10,
    )

    np.testing.assert_array_equal(scale[: len(persistent)], base_scale)
    np.testing.assert_array_equal(rotation[: len(persistent)], base_rotation)
    np.testing.assert_array_equal(
        linearity[: len(persistent)], base_linearity
    )
    assert audit["cross_instance_frame_neighbour_count"] == 0
    assert audit["cross_owner_frame_neighbour_count"] == 0


def test_dense_frames_do_not_cross_exact_owner_or_tree_instance():
    owned = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.06, 0.01, 0.0],
        ]
    )
    base = _ownership_isolated_track_frames(
        owned,
        np.ones(len(owned), dtype=bool),
        np.full(len(owned), 5, dtype=np.int32),
        np.full(len(owned), 408, dtype=np.int32),
        0.008,
        0.035,
        0.10,
    )
    foreign = np.asarray(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.003, 0.003, 0.003],
            [0.004, 0.004, 0.004],
        ]
    )
    xyz = np.concatenate([owned, foreign, foreign + 0.001])
    dense = np.ones(len(xyz), dtype=bool)
    instance = np.concatenate(
        [
            np.full(len(owned), 5),
            np.full(len(foreign), 5),
            np.full(len(foreign), 6),
        ]
    )
    owner = np.concatenate(
        [
            np.full(len(owned), 408),
            np.full(len(foreign), 409),
            np.full(len(foreign), 408),
        ]
    )
    full = _ownership_isolated_track_frames(
        xyz,
        dense,
        instance,
        owner,
        0.008,
        0.035,
        0.10,
    )

    for actual, expected in zip(full[:3], base[:3]):
        np.testing.assert_array_equal(actual[: len(owned)], expected)
    assert full[3]["dense_owner_count"] == 2
    assert full[3]["dense_owner_instance_group_count"] == 3


def test_sequence_local_envelope_removes_only_gross_depth_outliers():
    rigid = np.stack(
        [
            np.linspace(-5.0, 5.0, 100),
            np.zeros(100),
            np.zeros(100),
        ],
        axis=1,
    )
    samples = [
        {"xyz": np.asarray([6.0, 0.0, 0.0])},
        {"xyz": np.asarray([200.0, 0.0, 0.0])},
    ]
    retained, rejected, audit = _filter_sequence_local_scene_envelope(
        samples, rigid
    )
    assert len(retained) == 1
    assert rejected == 1
    assert audit["maximum_radius"] > audit["reference_radius"]


def test_nonchart_pointmap_canopy_is_not_silently_ignored(tmp_path):
    dataset = tmp_path / "dataset"
    image_root = dataset / "images"
    image_root.mkdir(parents=True)
    image_name = "seq2__frame00075.png"
    Image.new("RGB", (2, 2), (20, 80, 30)).save(
        image_root / image_name
    )
    pointmap = tmp_path / "pointmap.json"
    pointmap.write_text(
        json.dumps(
            {
                "points": [
                    [-1.0, -1.0, 2.0],
                    [0.0, -1.0, 2.0],
                    [-1.0, 0.0, 2.0],
                    [0.0, 0.0, 2.0],
                ],
                "confs": [[2.0, 2.0], [2.0, 2.0]],
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "pointmap_index.json"
    index.write_text(
        json.dumps(
            {
                "coordinate_frame": "cambridge_fixed_world",
                "camera_order": ["seq2__frame00075"],
                "records": {
                    "seq2__frame00075": {
                        "path": str(pointmap)
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    scene_contract = tmp_path / "scene_contract.json"
    scene_contract.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_name": image_name,
                        "T_camera_to_world": np.eye(4).tolist(),
                        "camera": {
                            "width": 2,
                            "height": 2,
                            "fx": 2.0,
                            "fy": 2.0,
                            "cx": 1.0,
                            "cy": 1.0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    keep = torch.ones((2, 2), dtype=torch.bool)
    tree = torch.zeros((2, 2), dtype=torch.bool)

    class Masks:
        masks = {
            image_name: (keep, keep, keep, tree)
        }

        @staticmethod
        def source_name_for(name):
            return name

    samples = _nonchart_pointmap_foliage_samples(
        {
            "dataset": str(dataset),
            "scene_contract": str(scene_contract),
            "artifacts": [
                {
                    "name": "mast3r_pointmap_index",
                    "path": str(index),
                }
            ],
        },
        {
            7: {
                "name": image_name,
                "qvec": np.asarray([1.0, 0.0, 0.0, 0.0]),
                "tvec": np.zeros(3),
            }
        },
        Masks(),
        samples_per_view=4,
        maximum_samples=4,
    )

    assert len(samples) == 4
    assert all(row["pointmap_sequence_local_sample"] for row in samples)
    assert {int(row["tree_image_ids"][0]) for row in samples} == {7}


def test_pointmap_fixed_camera_validity_rejects_behind_and_wrong_pixel():
    points = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, -2.0],
        ],
        dtype=np.float32,
    )
    scene_record = {
        "T_world_to_camera": np.eye(4).tolist(),
        "camera": {
            "width": 2,
            "height": 2,
            "fx": 2.0,
            "fy": 2.0,
            "cx": 0.0,
            "cy": 0.0,
        },
    }
    valid = _pointmap_fixed_camera_validity(
        points, (2, 2), scene_record
    )
    assert valid.tolist() == [True, True, True, False]

    points[1, 0] = 2.0
    valid = _pointmap_fixed_camera_validity(
        points, (2, 2), scene_record
    )
    assert not valid[1]


def test_surface_selection_excludes_canopy_even_with_many_observations():
    tracks = {
        "role_probabilities": np.asarray(
            [
                [0.90, 0.02, 0.02, 0.01, 0.05],
                [0.10, 0.85, 0.01, 0.01, 0.03],
                [0.72, 0.20, 0.01, 0.01, 0.06],
            ],
            dtype=np.float32,
        ),
        "valid_observation_count": np.asarray([5, 20, 8]),
        "reprojection_error": np.asarray([0.5, 0.3, 0.8]),
    }
    keep = _selection_mask(
        tracks,
        minimum_rigid_probability=0.7,
        maximum_canopy_probability=0.15,
        minimum_observations=2,
        maximum_reprojection_error=2.0,
    )
    assert keep.tolist() == [True, False, False]


def test_surface_frames_produce_two_tangent_scales_and_unit_quaternion():
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)
    )
    xyz = np.column_stack(
        [xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)]
    ).astype(np.float32)
    scales, quaternion, normal = _surface_frames(
        xyz, np.full(len(xyz), 0.01, dtype=np.float32)
    )
    assert scales.shape == (len(xyz), 2)
    assert quaternion.shape == (len(xyz), 4)
    np.testing.assert_allclose(
        np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-5
    )
    assert np.median(np.abs(normal[:, 2])) > 0.99


def test_dense_pointmap_sampling_preserves_coverage_and_cross_sequence_support():
    valid = np.ones((8, 16), dtype=bool)
    confidence = np.ones((8, 16), dtype=np.float32)
    confidence[:2, :2] = 100
    selected = _uniform_confidence_samples(valid, confidence, 32)
    rows, columns = np.divmod(selected, 16)
    assert len(selected) == 32
    assert len(np.unique(rows // 2)) == 4
    assert len(np.unique(columns // 4)) == 4

    xyz = np.asarray(
        [[0, 0, 0], [0.02, 0, 0], [4, 0, 0], [4.03, 0, 0]],
        dtype=np.float32,
    )
    sequence = np.asarray(["seq1", "seq2", "seq1", "seq1"])
    distance = _different_sequence_nearest_distance(xyz, sequence)
    np.testing.assert_allclose(distance[:2], [0.02, 0.02], atol=1e-6)
    assert distance[2] > 3.0
    assert distance[3] > 3.0


def test_pointmap_initialization_loads_native_cross_sequence_posterior(
    tmp_path,
):
    posterior_path = tmp_path / "posterior.npz"
    precision = np.asarray(
        [[1.0, 0.25], [0.03, 0.0]], dtype=np.float32
    )
    supported = np.asarray(
        [[True, True], [False, False]], dtype=bool
    )
    distance = np.asarray(
        [[0.01, 0.10], [-1.0, np.inf]], dtype=np.float32
    )
    np.savez_compressed(
        posterior_path,
        schema_version=np.asarray(
            "mast3r-cross-sequence-pointmap-posterior-v1"
        ),
        native_shape=np.asarray([2, 2], dtype=np.int32),
        precision=precision,
        cross_sequence_supported=supported,
        cross_sequence_distance=distance,
    )
    loaded = _load_pointmap_cross_sequence_posterior(
        {
            "cross_sequence_posterior": {
                "path": str(posterior_path),
                "bytes": posterior_path.stat().st_size,
            }
        },
        (2, 2),
    )

    np.testing.assert_allclose(loaded["precision"], precision)
    assert loaded["supported"].tolist() == supported.tolist()
    assert np.isinf(loaded["distance"][1]).all()


def test_missing_pointmap_posterior_cannot_manufacture_cross_sequence_support():
    available = np.asarray([True, False, False])
    precision = np.asarray([0.8, 1.0, 0.3], dtype=np.float32)
    supported = np.asarray([True, True, True])
    distance = np.asarray([0.02, 0.01, 0.10], dtype=np.float32)

    fallback_count = _apply_missing_pointmap_posterior_fallback(
        available, precision, supported, distance
    )

    assert fallback_count == 2
    assert precision.tolist() == pytest.approx([0.8, 0.03, 0.03])
    assert supported.tolist() == [True, False, False]
    assert np.isinf(distance[1:]).all()


def test_pointmap_seed_cap_balances_views_before_score_fill():
    indices = np.arange(12)
    view_id = np.repeat(np.arange(3), 4)
    score = np.asarray(
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1],
        dtype=np.float32,
    )
    selected = _balanced_pointmap_seed_cap(
        indices,
        budget=6,
        view_id=view_id,
        score=score,
    )
    assert len(selected) == 6
    assert np.bincount(view_id[selected], minlength=3).tolist() == [2, 2, 2]


def test_dense_pointmap_shape_and_frames_follow_source_raster():
    assert _native_pointmap_shape(288 * 512, (960, 540)) == (288, 512)
    yy, xx = np.mgrid[0:5, 0:7]
    pointmap = np.stack(
        [0.01 * xx, 0.01 * yy, np.full_like(xx, 2.0)],
        axis=-1,
    ).astype(np.float32)
    rows = np.asarray([1, 2, 3])
    columns = np.asarray([1, 3, 5])
    scales, quaternion, normal = _pointmap_surface_frames(
        pointmap, rows, columns
    )
    assert scales.shape == (3, 2)
    np.testing.assert_allclose(
        np.linalg.norm(quaternion, axis=1), 1.0, atol=1e-5
    )
    assert np.median(np.abs(normal[:, 2])) > 0.99


def test_sampled_point_footprint_tracks_retained_pixel_spacing():
    dense_rows, dense_columns = np.mgrid[0:4, 0:4]
    dense = _sampled_point_footprint_multiplier(
        dense_rows.reshape(-1), dense_columns.reshape(-1)
    )
    np.testing.assert_allclose(dense, 1.0)

    sparse_rows, sparse_columns = np.mgrid[0:20:5, 0:20:5]
    sparse = _sampled_point_footprint_multiplier(
        sparse_rows.reshape(-1), sparse_columns.reshape(-1)
    )
    np.testing.assert_allclose(sparse, 2.5)


def test_sparse_rigid_zbuffer_keeps_nearest_depth():
    view = {
        "image_id": 7,
        "width": 100,
        "height": 100,
        "rotation": np.eye(3),
        "translation": np.zeros(3),
        "fx": 100.0,
        "fy": 100.0,
        "cx": 50.0,
        "cy": 50.0,
    }
    points = np.asarray([[0, 0, 4], [0, 0, 2]], dtype=np.float32)
    depth = sparse_rigid_depth_maps(
        [view],
        points,
        resolution_scale=1.0,
        dilation_pixels=1,
    )[7]
    assert depth[50, 50] == 2.0


def test_static_tree_skeleton_requires_local_line_cylinder_evidence():
    line = np.column_stack(
        [
            np.linspace(0, 0.18, 8),
            np.zeros(8),
            np.zeros(8),
        ]
    ).astype(np.float32)
    supported, linearity = _line_supported_tree_tracks(
        line,
        np.zeros(len(line), dtype=np.int32),
        np.ones(len(line), dtype=bool),
        minimum_linearity=1.8,
    )
    assert supported[2:-2].all()
    assert np.median(linearity[2:-2]) > 10

    cloud = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.02, 0.02, 0.04],
            [0.02, 0.02, -0.04],
        ],
        dtype=np.float32,
    )
    rejected, _ = _line_supported_tree_tracks(
        cloud,
        np.zeros(len(cloud), dtype=np.int32),
        np.ones(len(cloud), dtype=bool),
        minimum_linearity=1.8,
    )
    assert not rejected.any()


def test_chart_anchor_model_excludes_unsupported_bootstrap_witnesses(
    tmp_path,
):
    path = tmp_path / "surface_seed.npz"
    np.savez_compressed(
        path,
        source_type=np.asarray([2, 2, 1], dtype=np.int8),
        persistent_geometry_evidence=np.asarray(
            [True, False, True], dtype=bool
        ),
        xyz=np.asarray(
            [[0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=np.float32
        ),
        position_sigma=np.asarray([0.02, 0.35, 0.01], dtype=np.float32),
        scales=np.full((3, 2), 0.02, dtype=np.float32),
        track_id=np.asarray([-2, -3, 7], dtype=np.int64),
        chart_id=np.asarray([0, 0, -1], dtype=np.int32),
        chart_uv=np.asarray(
            [[0.25, 0.25], [0.75, 0.25], [np.nan, np.nan]],
            dtype=np.float32,
        ),
    )
    model = ChartSurfaceModel(path)
    audit = model.audit()
    assert audit["anchor_count"] == 1
    assert audit["coverage_only_seed_count"] == 1


def test_chart_anchor_factor_maps_gapped_persistent_evidence_ids(tmp_path):
    path = tmp_path / "surface_seed.npz"
    np.savez_compressed(
        path,
        source_type=np.asarray([2, 2, 2], dtype=np.int8),
        persistent_geometry_evidence=np.asarray(
            [False, False, True], dtype=bool
        ),
        xyz=np.asarray(
            [[0, 0, 2], [1, 0, 2], [2, 0, 2]], dtype=np.float32
        ),
        position_sigma=np.full(3, 0.02, dtype=np.float32),
        scales=np.full((3, 2), 0.02, dtype=np.float32),
        track_id=np.asarray([-2, -3, -4], dtype=np.int64),
        chart_id=np.asarray([0, 0, 0], dtype=np.int32),
        chart_uv=np.asarray(
            [[0.1, 0.2], [0.2, 0.2], [0.3, 0.2]], dtype=np.float32
        ),
    )
    model = ChartSurfaceModel(path)

    class Surface:
        get_xyz = torch.tensor([[2.0, 0.0, 2.0]])
        _source_type = torch.tensor([2], dtype=torch.int16)
        _track_id = torch.tensor([-4], dtype=torch.int64)

    loss, audit = model.factor(Surface(), maximum_anchors=1)
    assert audit["matched"] == 1
    assert float(loss) == 0.0


def test_chart_uv_edges_resolve_from_all_live_witnesses_not_anchor_batch(
    tmp_path,
):
    path = tmp_path / "surface_seed.npz"
    np.savez_compressed(
        path,
        source_type=np.full(3, 2, dtype=np.int8),
        persistent_geometry_evidence=np.ones(3, dtype=bool),
        xyz=np.asarray(
            [[0, 0, 2], [1, 0, 2], [2, 0, 2]], dtype=np.float32
        ),
        position_sigma=np.full(3, 0.02, dtype=np.float32),
        scales=np.full((3, 2), 0.02, dtype=np.float32),
        track_id=np.asarray([-2, -3, -4], dtype=np.int64),
        chart_id=np.zeros(3, dtype=np.int32),
        chart_uv=np.asarray(
            [[0.1, 0.2], [0.2, 0.2], [0.3, 0.2]], dtype=np.float32
        ),
    )
    model = ChartSurfaceModel(path)

    class Surface:
        get_xyz = torch.tensor(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0]]
        )
        _source_type = torch.full((3,), 2, dtype=torch.int16)
        _track_id = torch.tensor([-2, -3, -4], dtype=torch.int64)

    _, audit = model.factor(Surface(), maximum_anchors=1)

    # Only one anchor is sampled, so the former minibatch-coupled endpoint
    # lookup could not possibly match both ends of an edge.
    assert audit["matched"] == 1
    assert audit["live_chart_rows"] == 3
    assert audit["uv_edges_sampled"] == 1
    assert audit["uv_edges_matched"] == 1


def test_chart_uv_priority_robustly_bounds_screen_gradient_outliers():
    class AtlasProbe:
        def _bound_geometry(self, surface):
            rows = torch.arange(4)
            return rows, rows, (
                torch.zeros(4, 3),
                torch.ones(4, 2),
                torch.zeros(4, 4),
            )

    class Surface:
        get_xyz = torch.zeros(4, 3)
        _track_id = torch.tensor([-2, -3, -4, -5])
        _geometry_confidence = torch.ones(4)
        xyz_gradient_accum = torch.tensor(
            [[1e-3], [2e-3], [3e-3], [1e12]]
        )
        denom = torch.ones(4, 1)
        max_radii2D = torch.tensor([2.0, 4.0, 8.0, 1e9])

    audit = LearnableInverseDepthAtlas.topology_priority_snapshot(
        AtlasProbe(),
        Surface(),
        gradient_threshold=2e-4,
        radius_target=8.0,
    )
    assert torch.isfinite(audit["priority"]).all()
    assert float(audit["priority"].max()) <= 16.0
    assert float(audit["raw_gradient_maximum"]) == pytest.approx(1e12)


def test_chart_uv_priority_is_integrated_over_projected_cell_area():
    class AtlasProbe:
        def _bound_geometry(self, surface):
            rows = torch.arange(2)
            return rows, rows, (
                torch.zeros(2, 3),
                torch.ones(2, 2),
                torch.zeros(2, 4),
            )

    class Surface:
        get_xyz = torch.zeros(2, 3)
        _track_id = torch.tensor([-2, -3])
        _geometry_confidence = torch.ones(2)
        # Identical saturated gradient density, but only the first cell has
        # enough projected area for a split to reduce image error.
        xyz_gradient_accum = torch.full((2, 1), 1e3)
        denom = torch.ones(2, 1)
        max_radii2D = torch.tensor([8.0, 0.25])

    audit = LearnableInverseDepthAtlas.topology_priority_snapshot(
        AtlasProbe(),
        Surface(),
        gradient_threshold=2e-4,
        radius_target=8.0,
    )

    assert float(audit["priority"][0]) > 1.0
    assert float(audit["priority"][1]) < 1.0


def test_chart_jacobian_frame_owns_both_tangent_scales_and_orientation():
    centre = torch.zeros(1, 3, requires_grad=True)
    right = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
    # The UV-v derivative contains shear along u; only its orthogonal
    # component is the second 2DGS tangent scale.
    down = torch.tensor([[1.0, 3.0, 0.0]], requires_grad=True)

    scales, quaternion = LearnableInverseDepthAtlas._jacobian_frame(
        centre, right, down
    )

    torch.testing.assert_close(scales, torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(
        quaternion,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        atol=1e-4,
        rtol=1e-4,
    )
    (scales.sum() + quaternion.sum()).backward()
    assert torch.isfinite(centre.grad).all()
    assert torch.isfinite(right.grad).all()
    assert torch.isfinite(down.grad).all()


def test_chart_runtime_geometry_applies_physical_tangent_scale_contract():
    class AtlasProbe:
        maximum_tangent_scale = 0.5

        def _live_rows(self, surface):
            return (
                torch.tensor([0]),
                torch.tensor([0]),
                {
                    "chart_id": torch.tensor([0]),
                    "uv": torch.tensor([[0.5, 0.5]]),
                    "half_uv": torch.tensor([[0.25, 0.25]]),
                    "base_rho": torch.tensor([0.5]),
                },
            )

        def _points(self, chart_id, uv, fallback):
            del chart_id
            return torch.cat([10.0 * uv, fallback[:, None]], dim=1), fallback

        _jacobian_frame = staticmethod(
            LearnableInverseDepthAtlas._jacobian_frame
        )

    primitive, _, geometry = LearnableInverseDepthAtlas._bound_geometry(
        AtlasProbe(), object()
    )

    assert primitive.tolist() == [0]
    scales = geometry[1]
    torch.testing.assert_close(scales, torch.full((1, 2), 0.5))


def test_chart_runtime_geometry_honors_persistent_screen_scale_ceiling():
    class AtlasProbe:
        maximum_tangent_scale = 0.5

        def _live_rows(self, surface):
            return (
                torch.tensor([0]),
                torch.tensor([0]),
                {
                    "chart_id": torch.tensor([0]),
                    "uv": torch.tensor([[0.5, 0.5]]),
                    "half_uv": torch.tensor([[0.25, 0.25]]),
                    "base_rho": torch.tensor([0.5]),
                },
            )

        def _points(self, chart_id, uv, fallback):
            del chart_id
            return torch.cat([10.0 * uv, fallback[:, None]], dim=1), fallback

        _jacobian_frame = staticmethod(
            LearnableInverseDepthAtlas._jacobian_frame
        )
        _local_chart_geometry = LearnableInverseDepthAtlas._local_chart_geometry

    class Surface:
        get_scaling = torch.full((1, 2), 0.04)

    _, _, geometry = LearnableInverseDepthAtlas._bound_geometry(
        AtlasProbe(), Surface()
    )

    torch.testing.assert_close(geometry[1], torch.full((1, 2), 0.04))


def test_chart_local_frame_does_not_bridge_depth_discontinuity():
    class AtlasProbe:
        def _points(self, chart_id, uv, fallback):
            del chart_id, fallback
            rho = torch.where(
                uv[:, 0] > 0.5,
                torch.full_like(uv[:, 0], 0.05),
                torch.full_like(uv[:, 0], 0.5),
            )
            xyz = torch.stack([uv[:, 0], uv[:, 1], rho.reciprocal()], dim=1)
            return xyz, rho

        _jacobian_frame = staticmethod(
            LearnableInverseDepthAtlas._jacobian_frame
        )

    centre, scales, quaternion = (
        LearnableInverseDepthAtlas._local_chart_geometry(
            AtlasProbe(),
            torch.tensor([0]),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[0.1, 0.1]]),
            torch.tensor([0.5]),
        )
    )

    torch.testing.assert_close(centre, torch.tensor([[0.5, 0.5, 2.0]]))
    assert float(scales.max()) < 0.2
    assert torch.isfinite(quaternion).all()


def test_chart_uv_split_uses_current_jacobian_frame_at_atlas_border():
    class AtlasProbe:
        evidence_id = np.asarray([-2], dtype=np.int64)
        chart_id = np.asarray([0], dtype=np.int32)
        uv = np.asarray([[0.9, 0.9]], dtype=np.float32)
        half_uv = np.asarray([[0.2, 0.2]], dtype=np.float32)
        base_rho = np.asarray([0.5], dtype=np.float32)
        base_tangent_scale = np.asarray(
            [[0.1, 0.1]], dtype=np.float32
        )
        level = np.asarray([0], dtype=np.int16)
        next_evidence_id = -3
        quadtree_events = []
        _jacobian_frame = staticmethod(
            LearnableInverseDepthAtlas._jacobian_frame
        )

        def _bound_geometry(self, surface):
            return (
                torch.tensor([0]),
                torch.tensor([0]),
                (
                    surface.get_xyz,
                    torch.ones(1, 2),
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                ),
            )

        def _topology(self, device):
            return {
                "uv": torch.from_numpy(self.uv).to(device),
                "half_uv": torch.from_numpy(self.half_uv).to(device),
                "chart_id": torch.from_numpy(self.chart_id).to(device),
                "base_rho": torch.from_numpy(self.base_rho).to(device),
                "level": torch.from_numpy(self.level).to(device),
            }

        def _points(self, chart_id, uv, fallback):
            xyz = torch.cat(
                [uv, fallback[:, None].reciprocal()], dim=1
            )
            return xyz, fallback

        def _invalidate_cache(self):
            pass

    class Surface:
        def __init__(self):
            self.get_xyz = torch.tensor([[0.9, 0.9, 2.0]])
            self._track_id = torch.tensor([-2], dtype=torch.int64)
            self._features_dc = torch.zeros(1, 1, 3)
            self._features_rest = torch.zeros(1, 0, 3)
            self._opacity = torch.zeros(1, 1)
            self.appended = None

        def _point_metadata_from_indices(self, rows, **kwargs):
            del kwargs
            return {"track_id": self._track_id[rows].clone()}

        def densification_postfix(
            self,
            xyz,
            features_dc,
            features_rest,
            opacity,
            scaling,
            rotation,
            *,
            metadata,
        ):
            del features_dc, features_rest, opacity
            self.appended = (xyz, scaling, rotation, metadata)
            self.get_xyz = torch.cat([self.get_xyz, xyz])

        def prune_points(self, remove):
            self.get_xyz = self.get_xyz[~remove]

    atlas = AtlasProbe()
    surface = Surface()
    event = LearnableInverseDepthAtlas.adapt_uv_quadtree(
        atlas,
        surface,
        maximum_net_growth=3,
        maximum_points=10,
        gradient_threshold=1e-4,
        radius_target=8.0,
        priority_snapshot={
            "evidence_id": torch.tensor([-2]),
            "priority": torch.tensor([2.0]),
        },
    )

    assert event["parents_replaced"] == 1
    assert event["children"] == 4
    assert len(surface.get_xyz) == 4
    _, scaling, rotation, metadata = surface.appended
    assert torch.isfinite(scaling).all()
    assert torch.isfinite(rotation).all()
    assert metadata["track_id"].tolist() == [-3, -4, -5, -6]

    capped_atlas = AtlasProbe()
    capped_atlas.level = np.asarray([3], dtype=np.int16)
    capped_event = LearnableInverseDepthAtlas.adapt_uv_quadtree(
        capped_atlas,
        Surface(),
        maximum_net_growth=3,
        maximum_points=10,
        gradient_threshold=1e-4,
        radius_target=8.0,
        maximum_level=3,
        priority_snapshot={
            "evidence_id": torch.tensor([-2]),
            "priority": torch.tensor([2.0]),
        },
    )
    assert capped_event["parents_replaced"] == 0
    assert capped_event["eligible"] == 0
    assert capped_event["configured_maximum_level"] == 3
