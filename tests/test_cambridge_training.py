from pathlib import Path

import torch

from matcha.cambridge_training import (
    PerImageAffineColorCorrection,
    compute_rgb_loss,
    dense_depth_weight,
    geometry_iteration,
    load_depth_cache,
    rgb_supervision_weight,
    sanitize_chart_geometry,
    sanitize_depth,
    save_depth_cache,
)


def test_affine_color_correction_is_identity_and_image_specific():
    correction = PerImageAffineColorCorrection(["a.png", "b.png", "a.png"])
    image = torch.ones((3, 1, 1))

    assert torch.allclose(correction(image, "a.png"), image)
    assert torch.isclose(correction.identity_regularization(), torch.tensor(0.0))

    with torch.no_grad():
        correction.log_scales[1, :, 0, 0] = torch.log(torch.tensor([2.0, 3.0, 4.0]))
        correction.biases[1, :, 0, 0] = torch.tensor([0.1, 0.2, 0.3])

    assert torch.allclose(correction(image, "a.png"), image)
    assert torch.allclose(
        correction(image, "b.png"),
        torch.tensor([[[2.1]], [[3.2]], [[4.3]]]),
    )
    assert correction.identity_regularization() > 0


def _mean_similarity(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - (image - target).abs().mean()


def test_rgb_loss_excludes_masked_pixels_and_renormalizes_dssim():
    target = torch.zeros((3, 2, 2))
    image = torch.ones((3, 2, 2))
    mask = torch.tensor([[True, False], [False, False]])

    rgb, total = compute_rgb_loss(
        image,
        target,
        mask=mask,
        lambda_dssim=0.2,
        ssim_fn=_mean_similarity,
    )

    assert torch.isclose(rgb, torch.tensor(1.0))
    assert torch.isclose(total, torch.tensor(1.0))


def test_geometry_schedule_uses_chart_views_early_then_dense_only():
    assert geometry_iteration(5, use_dense_supervision=True, every_n=5, dense_only_from_iter=3000)
    assert not geometry_iteration(6, use_dense_supervision=True, every_n=5, dense_only_from_iter=3000)
    assert not geometry_iteration(3005, use_dense_supervision=True, every_n=5, dense_only_from_iter=3000)
    assert geometry_iteration(3005, use_dense_supervision=False, every_n=5, dense_only_from_iter=3000)


def test_chart_only_screen_keeps_real_chart_rgb_at_full_weight():
    assert rgb_supervision_weight(
        is_pseudo_view=False,
        is_geometry_view=True,
        use_dense_supervision=False,
        downweight_input_view_color_loss=True,
        pseudo_rgb_weight=0.01,
    ) == 1.0
    assert rgb_supervision_weight(
        is_pseudo_view=False,
        is_geometry_view=True,
        use_dense_supervision=True,
        downweight_input_view_color_loss=True,
        pseudo_rgb_weight=0.01,
    ) == 0.01
    assert rgb_supervision_weight(
        is_pseudo_view=True,
        is_geometry_view=True,
        use_dense_supervision=False,
        downweight_input_view_color_loss=False,
        pseudo_rgb_weight=0.01,
    ) == 0.01


def test_sanitize_depth_combines_confidence_semantics_and_outlier_limit():
    depth = torch.tensor([[1.0, float("nan")], [60.0, 2.0]])
    confidence = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    semantic = torch.tensor([[True, True], [True, True]])

    clean, valid = sanitize_depth(
        depth,
        confidence=confidence,
        semantic_mask=semantic,
        max_abs_depth=50.0,
    )

    assert torch.equal(valid, torch.tensor([[True, False], [False, False]]))
    assert torch.equal(clean, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_sanitize_chart_geometry_removes_unsupported_and_extreme_pixels():
    charts = {
        "depths": torch.tensor([[[1.0, 2.0], [60.0, 3.0]]]),
        "prior_depths": torch.tensor([[[1.1, -1.0], [2.0, 3.1]]]),
        "confs": torch.tensor([[[2.0, 2.0], [2.0, 0.0]]]),
        "pts": torch.tensor([[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                               [[1.0, 2.0, 3.0], [100.0, 2.0, 3.0]]]]),
        "scale_factor": torch.tensor(1.0),
    }

    sanitized, valid = sanitize_chart_geometry(charts)

    assert torch.equal(valid, torch.tensor([[[True, False], [False, False]]]))
    assert torch.equal(sanitized["depths"], torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))
    assert torch.count_nonzero(sanitized["pts"]) == 3
    assert sanitized["scale_factor"] is charts["scale_factor"]


def test_dense_depth_cache_checks_camera_order(tmp_path: Path):
    path = tmp_path / "depths.pt"
    save_depth_cache(path, ["a.png", "b.png"], [torch.ones(2, 2), torch.zeros(2, 2)])

    depths = load_depth_cache(path, ["a.png", "b.png"])

    assert len(depths) == 2
    assert depths[0].dtype == torch.float16


def test_strong_decay_depth_schedule_releases_monocular_prior_late():
    assert dense_depth_weight(1, "strong_decay") == 1.0
    assert dense_depth_weight(7000, "strong_decay") == 1.0
    assert dense_depth_weight(7001, "strong_decay") == 0.1
    assert dense_depth_weight(15001, "strong_decay") == 0.01
    assert dense_depth_weight(25001, "strong_decay") == 0.0001
