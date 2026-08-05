import json
import hashlib
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from outdoor.directional_sky import (
    CanonicalDirectionalSky,
    composite_white_background,
)
from outdoor.task_fields import OutdoorTaskFieldLookup
from outdoor.task_fields import _soft_boundary
from outdoor.projected_role_posterior import (
    PROJECTED_RIGID_POSTERIOR_VERSION,
)


def _task_lookup(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "name_mapping.json").write_text(
        json.dumps({"seq1__frame00001.png": "seq1/frame00001.png"})
    )
    masks = (
        # One transient pixel.
        torch.tensor([[False, True], [True, True]]),
        # One sky pixel.
        torch.tensor([[True, False], [True, True]]),
        torch.ones((2, 2), dtype=torch.bool),
        # One canopy pixel.
        torch.tensor([[True, True], [False, True]]),
    )
    mask_pickle = tmp_path / "tree_masks.pkl"
    with mask_pickle.open("wb") as handle:
        pickle.dump({"seq1/frame00001.png": masks}, handle)
    manifest = tmp_path / "semantics.json"
    manifest.write_text(
        json.dumps(
            {
                "input_hashes": {
                    "tree_mask_pickle": hashlib.sha256(
                        mask_pickle.read_bytes()
                    ).hexdigest()
                },
                "class_availability": {
                    "canopy": "tree_mask_index_3",
                    "trunk": "unlabelled",
                }
            }
        )
    )
    return OutdoorTaskFieldLookup(
        dataset,
        mask_pickle,
        manifest,
        boundary_radius=0,
        max_cached_views=1,
    )


def test_task_fields_keep_transient_sky_canopy_and_rigid_disjoint(tmp_path):
    lookup = _task_lookup(tmp_path)
    fields = lookup.fields(
        "seq1__frame00001", (2, 2), torch.device("cpu")
    )

    assert fields["p_transient"][0, 0] == 1
    assert fields["p_sky"][0, 1] == 1
    assert fields["p_canopy"][1, 0] == 1
    assert fields["p_rigid"][1, 1] == 1
    assert fields["p_trunk"].count_nonzero() == 0
    assert fields["p_branch"].count_nonzero() == 0
    assert torch.equal(
        fields["p_canopy_core"] + fields["p_canopy_boundary"],
        fields["p_canopy"],
    )
    assert fields["w_topology"][1, 0] == 1
    assert fields["w_topology"].sum() == 1
    assert fields["w_sky_rgb"].sum() == 1


def test_task_fields_explicitly_report_rigid_building_ground_proxies(tmp_path):
    audit = _task_lookup(tmp_path).audit()

    assert audit["materialization"] == "deterministic_runtime_per_sample"
    assert audit["mask_channel_contract"]["trunk"] == "unavailable_zero"
    assert audit["boundary_policy"]["type"].startswith(
        "resolution_and_foreground_scale_aware"
    )
    assert (
        audit["mask_channel_contract"]["building"]
        == "rigid_proxy_not_semantic_segmentation"
    )


def test_multiview_tracks_add_positive_tree_surface_without_relabeling_mask(
    tmp_path,
):
    lookup = _task_lookup(tmp_path)
    count = 8
    tracks = tmp_path / "tracks.npz"
    role = np.zeros((count, 5), dtype=np.float32)
    role[:, 1] = 1.0
    np.savez_compressed(
        tracks,
        xyz=np.column_stack(
            [
                np.zeros(count),
                np.arange(count) * 0.03,
                np.ones(count),
            ]
        ).astype(np.float32),
        role_probabilities=role,
        reprojection_error=np.full(count, 0.2, dtype=np.float32),
        valid_observation_count=np.full(count, 3, dtype=np.int16),
        sequence_count=np.full(count, 2, dtype=np.int16),
        triangulation_angle_median=np.full(count, 10.0, dtype=np.float32),
        cycle_consistency=np.full(count, 0.8, dtype=np.float32),
        observation_offsets=np.arange(count + 1, dtype=np.int64),
        observation_camera_indices=np.zeros(count, dtype=np.int32),
        observation_pixels=np.full((count, 2), [0.25, 1.25], dtype=np.float32),
        camera_names=np.asarray(["seq1__frame00001"]),
        camera_image_sizes=np.asarray([[2, 2]], dtype=np.int32),
    )
    lookup = OutdoorTaskFieldLookup(
        lookup.dataset_path,
        lookup.tree_mask_pickle,
        lookup.semantic_manifest_path,
        multiview_track_archive=tracks,
        boundary_radius=0,
    )

    fields = lookup.fields(
        "seq1__frame00001", (2, 2), torch.device("cpu")
    )

    # The posterior survives only at the real canopy pixel (1, 0); max-pool
    # support cannot turn rigid, sky or transient mask pixels into tree.
    assert fields["p_tree_surface"][1, 0] > 0
    assert fields["p_tree_surface"].count_nonzero() == 1
    assert fields["p_trunk"][1, 0] > 0
    assert torch.all(fields["p_crown"] <= fields["p_canopy"])


def test_multiview_rigid_tracks_rescue_facade_from_tree_mask(tmp_path):
    lookup = _task_lookup(tmp_path)
    count = 8
    tracks = tmp_path / "rigid_tracks.npz"
    role = np.zeros((count, 5), dtype=np.float32)
    role[:, 0] = 1.0
    np.savez_compressed(
        tracks,
        xyz=np.column_stack(
            [
                np.arange(count) * 0.03,
                np.zeros(count),
                np.ones(count),
            ]
        ).astype(np.float32),
        role_probabilities=role,
        reprojection_error=np.full(count, 0.2, dtype=np.float32),
        valid_observation_count=np.full(count, 3, dtype=np.int16),
        sequence_count=np.full(count, 2, dtype=np.int16),
        triangulation_angle_median=np.full(count, 10.0, dtype=np.float32),
        cycle_consistency=np.full(count, 0.8, dtype=np.float32),
        observation_offsets=np.arange(count + 1, dtype=np.int64),
        observation_camera_indices=np.zeros(count, dtype=np.int32),
        observation_pixels=np.full(
            (count, 2), [0.25, 1.25], dtype=np.float32
        ),
        camera_names=np.asarray(["seq1__frame00001"]),
        camera_image_sizes=np.asarray([[2, 2]], dtype=np.int32),
    )
    lookup = OutdoorTaskFieldLookup(
        lookup.dataset_path,
        lookup.tree_mask_pickle,
        lookup.semantic_manifest_path,
        multiview_track_archive=tracks,
        boundary_radius=0,
    )
    fields = lookup.fields(
        "seq1__frame00001", (2, 2), torch.device("cpu")
    )
    assert fields["p_multiview_rigid_rescue"][1, 0] > 0
    assert fields["p_rigid"][1, 0] > 0
    assert fields["p_canopy"][1, 0] < 1
    assert torch.allclose(
        fields["p_rigid"][1, 0] + fields["p_canopy"][1, 0],
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_projected_rigid_geometry_and_rgb_visibility_have_separate_authority(
    tmp_path,
):
    lookup = _task_lookup(tmp_path)
    projected = tmp_path / "projected_rigid.npz"
    np.savez_compressed(
        projected,
        schema_version=np.asarray(PROJECTED_RIGID_POSTERIOR_VERSION),
        camera_names=np.asarray(["seq1__frame00001.png"]),
        camera_image_sizes=np.asarray([[2, 2]], dtype=np.int32),
        offsets=np.asarray([0, 1], dtype=np.int64),
        pixels=np.asarray([[0, 0]], dtype=np.uint16),
        geometry_support_probability=np.asarray([0.8], dtype=np.float16),
        visible_observation_probability=np.asarray([0.5], dtype=np.float16),
        camera_depth=np.asarray([2.0], dtype=np.float32),
        source_track_id=np.asarray([7], dtype=np.int64),
    )
    lookup = OutdoorTaskFieldLookup(
        lookup.dataset_path,
        lookup.tree_mask_pickle,
        lookup.semantic_manifest_path,
        projected_rigid_posterior_archive=projected,
        boundary_radius=0,
    )
    fields = lookup.fields(
        "seq1__frame00001", (2, 2), torch.device("cpu")
    )

    # The transient prior is transferred continuously, not replaced by a
    # binary gate. Only visible probability restores current RGB ownership;
    # the remaining geometric support is explicit unknown/occluded evidence.
    assert torch.allclose(fields["p_rigid"][0, 0], torch.tensor(0.5))
    assert torch.allclose(fields["p_transient"][0, 0], torch.tensor(0.5))
    assert torch.allclose(
        fields["p_projected_rigid_geometry"][0, 0], torch.tensor(0.8),
        atol=5e-4,
    )
    assert torch.allclose(
        fields["p_projected_rigid_visible"][0, 0], torch.tensor(0.5),
        atol=5e-4,
    )
    assert fields["p_unknown_ownership"][0, 0] >= 0.29
    assert fields["w_gaussian_rgb"][0, 0] > 0


def test_tree_occluded_projected_geometry_is_unknown_not_rgb_visible(
    tmp_path,
):
    lookup = _task_lookup(tmp_path)
    projected = tmp_path / "projected_rigid.npz"
    np.savez_compressed(
        projected,
        schema_version=np.asarray(PROJECTED_RIGID_POSTERIOR_VERSION),
        camera_names=np.asarray(["seq1__frame00001.png"]),
        camera_image_sizes=np.asarray([[2, 2]], dtype=np.int32),
        offsets=np.asarray([0, 1], dtype=np.int64),
        pixels=np.asarray([[0, 1]], dtype=np.uint16),
        geometry_support_probability=np.asarray([0.8], dtype=np.float16),
        visible_observation_probability=np.asarray([0.0], dtype=np.float16),
        camera_depth=np.asarray([2.0], dtype=np.float32),
        source_track_id=np.asarray([7], dtype=np.int64),
    )
    lookup = OutdoorTaskFieldLookup(
        lookup.dataset_path,
        lookup.tree_mask_pickle,
        lookup.semantic_manifest_path,
        projected_rigid_posterior_archive=projected,
        boundary_radius=0,
    )
    fields = lookup.fields(
        "seq1__frame00001", (2, 2), torch.device("cpu")
    )

    assert torch.allclose(
        fields["p_projected_rigid_geometry"][1, 0],
        torch.tensor(0.8),
        atol=5e-4,
    )
    assert fields["p_projected_rigid_visible"][1, 0] == 0
    assert fields["projected_rigid_depth"][1, 0] == 2.0
    assert fields["p_rigid"][1, 0] == 0
    assert fields["p_canopy"][1, 0] == 1
    assert fields["p_unknown_ownership"][1, 0] >= 0.79

    quarantined_lookup = OutdoorTaskFieldLookup(
        lookup.dataset_path,
        lookup.tree_mask_pickle,
        lookup.semantic_manifest_path,
        projected_rigid_posterior_archive=projected,
        boundary_radius=1,
    )
    quarantined = quarantined_lookup.fields(
        "seq1__frame00001", (2, 2), torch.device("cpu")
    )
    # Hidden geometry is useful in the canopy interior, but this tiny mask is
    # entirely boundary band. With no current-view visibility it must not
    # grow a broad surfel across the tree/building edge.
    assert quarantined["p_projected_rigid_geometry"][1, 0] == 0
    assert quarantined["projected_rigid_depth"][1, 0] == 0


def test_projected_geometry_and_depth_keep_the_same_track_owner(tmp_path):
    lookup = _task_lookup(tmp_path)
    projected = tmp_path / "projected_rigid.npz"
    np.savez_compressed(
        projected,
        schema_version=np.asarray(PROJECTED_RIGID_POSTERIOR_VERSION),
        camera_names=np.asarray(["seq1__frame00001.png"]),
        camera_image_sizes=np.asarray([[7, 7]], dtype=np.int32),
        offsets=np.asarray([0, 2], dtype=np.int64),
        pixels=np.asarray([[2, 2], [4, 2]], dtype=np.uint16),
        geometry_support_probability=np.asarray(
            [0.8, 0.5], dtype=np.float16
        ),
        visible_observation_probability=np.zeros(2, dtype=np.float16),
        camera_depth=np.asarray([10.0, 2.0], dtype=np.float32),
        source_track_id=np.asarray([7, 8], dtype=np.int64),
    )
    lookup = OutdoorTaskFieldLookup(
        lookup.dataset_path,
        lookup.tree_mask_pickle,
        lookup.semantic_manifest_path,
        projected_rigid_posterior_archive=projected,
        boundary_radius=0,
    )
    fields = lookup.fields(
        "seq1__frame00001", (7, 7), torch.device("cpu")
    )

    # Both rows fall inside the five-pixel footprint at (x=3,y=2). The
    # higher-confidence row owns both authority and depth; the closer but
    # lower-confidence neighbour must not donate an unrelated depth target.
    assert torch.allclose(
        fields["p_projected_rigid_geometry"][2, 3],
        torch.tensor(0.8),
        atol=5e-4,
    )
    assert fields["projected_rigid_depth"][2, 3] == 10.0


def test_projected_depth_beyond_renderer_far_plane_has_no_authority(tmp_path):
    lookup = _task_lookup(tmp_path)
    projected = tmp_path / "projected_rigid.npz"
    np.savez_compressed(
        projected,
        schema_version=np.asarray(PROJECTED_RIGID_POSTERIOR_VERSION),
        camera_names=np.asarray(["seq1__frame00001.png"]),
        camera_image_sizes=np.asarray([[2, 2]], dtype=np.int32),
        offsets=np.asarray([0, 1], dtype=np.int64),
        pixels=np.asarray([[0, 0]], dtype=np.uint16),
        geometry_support_probability=np.asarray([0.9], dtype=np.float16),
        visible_observation_probability=np.asarray([0.8], dtype=np.float16),
        camera_depth=np.asarray([101.0], dtype=np.float32),
        source_track_id=np.asarray([7], dtype=np.int64),
    )
    lookup = OutdoorTaskFieldLookup(
        lookup.dataset_path,
        lookup.tree_mask_pickle,
        lookup.semantic_manifest_path,
        projected_rigid_posterior_archive=projected,
        boundary_radius=0,
    )
    fields = lookup.fields(
        "seq1__frame00001", (2, 2), torch.device("cpu")
    )

    assert fields["p_projected_rigid_geometry"].sum() == 0
    assert fields["p_projected_rigid_visible"].sum() == 0
    assert fields["projected_rigid_depth"].sum() == 0
    assert (
        lookup.audit()["all_camera_projected_rigid_posterior"]
        ["discarded_out_of_renderer_depth_rows"]
        == 1
    )


def test_vectorized_mask_and_boundary_fast_paths_are_exact(tmp_path):
    lookup = _task_lookup(tmp_path)
    shape = (7, 9)
    individual = torch.stack(
        [
            lookup.lookup.get_index_mask(
                "seq1__frame00001", index, shape, torch.device("cpu")
            )
            for index in range(4)
        ]
    )
    vectorized = lookup.lookup.get_index_masks(
        "seq1__frame00001",
        (0, 1, 2, 3),
        shape,
        torch.device("cpu"),
    )
    assert torch.equal(vectorized, individual)

    mask = torch.tensor(
        [
            [False, False, True, False, False],
            [False, True, True, True, False],
            [False, False, True, False, False],
        ]
    )
    for radius in (1, 2, 4):
        value = mask.float()[None, None]
        kernel = 2 * radius + 1
        legacy = (
            torch.nn.functional.max_pool2d(
                value, kernel, stride=1, padding=radius
            )
            + torch.nn.functional.max_pool2d(
                -value, kernel, stride=1, padding=radius
            )
        )[0, 0].clamp(0, 1)
        assert torch.equal(_soft_boundary(mask, radius), legacy)


def test_directional_sky_uses_world_rays_and_white_alpha_compositing():
    sky = CanonicalDirectionalSky(degree=0, initial_rgb=0.8)
    camera = SimpleNamespace(
        image_height=2,
        image_width=3,
        focal_x=4.0,
        focal_y=4.0,
        cx=1.0,
        cy=0.5,
        R=np.eye(3, dtype=np.float32),
    )
    sky_rgb = sky(camera)

    assert sky_rgb.shape == (3, 2, 3)
    assert torch.allclose(sky_rgb, torch.full_like(sky_rgb, 0.8), atol=1e-6)
    gaussian_white = torch.ones_like(sky_rgb) * 0.6
    alpha = torch.tensor([[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]])
    composite = composite_white_background(gaussian_white, alpha, sky_rgb)
    expected = gaussian_white + (1.0 - alpha) * (sky_rgb - 1.0)
    assert torch.allclose(composite, expected)
