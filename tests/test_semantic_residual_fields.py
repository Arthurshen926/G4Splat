import json
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
