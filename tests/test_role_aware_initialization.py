import json

import numpy as np
import pytest
import torch
from PIL import Image

from outdoor.chart_surface_model import ChartSurfaceModel
from outdoor.evidence_store import ROLE_CANOPY, ROLE_RIGID
from outdoor.role_aware_initialization import (
    _align_tracks_to_local_hull_instances,
    _balanced_pointmap_seed_cap,
    _chart_hypothesis_authority,
    _compact_padded_camera_metadata,
    _cross_sequence_ray_posterior,
    _different_sequence_nearest_distance,
    _effective_posterior_budget,
    _filter_sequence_local_scene_envelope,
    _line_supported_tree_tracks,
    _load_pointmap_cross_sequence_posterior,
    _native_pointmap_shape,
    _nonchart_pointmap_foliage_samples,
    _pointmap_fixed_camera_validity,
    _pointmap_surface_frames,
    _rigid_scene_envelope_mask,
    _sampled_point_footprint_multiplier,
    _selection_mask,
    _surface_frames,
    _uniform_confidence_samples,
    SOURCE_SURFACE_DAV2,
    sparse_rigid_depth_maps,
)
from scripts.initialize_unified_outdoor_scene import (
    _validated_reused_foliage,
)
from scene.gaussian_model import GaussianModel


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
