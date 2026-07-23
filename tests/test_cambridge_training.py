from pathlib import Path
import json

import torch

from matcha.cambridge_training import (
    apply_per_image_affine_color_correction,
    load_per_image_affine_color_correction,
    PerImageAffineColorCorrection,
    compute_masked_depth_order_loss,
    compute_rgb_loss,
    dense_depth_weight,
    densification_stats_from_view,
    fused_geometry_validity_mask,
    geometry_chart_sampling_indices,
    geometry_iteration,
    geometry_prior_schedule_weight,
    load_depth_cache,
    opacity_reset_due,
    preliminary_uses_dense_supervision,
    rgb_sampling_importance_weights,
    rgb_sampling_importance_weights_by_camera,
    resolve_active_chart_indices,
    rgb_supervision_weight,
    scale_chart_geometry_priors,
    sanitize_chart_geometry,
    sanitize_depth,
    save_per_image_affine_color_correction,
    save_depth_cache,
    ulfloc_masked_supervision,
    validate_chart_camera_order,
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


def test_affine_color_correction_checkpoint_round_trip_preserves_camera_contract(tmp_path: Path):
    correction = PerImageAffineColorCorrection(["a.png", "b.png"])
    with torch.no_grad():
        correction.log_scales[1, :, 0, 0] = torch.log(torch.tensor([2.0, 3.0, 4.0]))
        correction.biases[1, :, 0, 0] = torch.tensor([0.1, 0.2, 0.3])

    path = tmp_path / "color_correction.pth"
    save_per_image_affine_color_correction(correction, path)
    restored = load_per_image_affine_color_correction(path, device="cpu")
    image = torch.ones((3, 1, 1))

    assert torch.allclose(
        apply_per_image_affine_color_correction(restored, image, "b.png"),
        torch.tensor([[[2.1]], [[3.2]], [[4.3]]]),
    )
    # Novel cameras deliberately receive identity, rather than accidentally
    # using another training camera's photometric parameters.
    assert torch.allclose(
        apply_per_image_affine_color_correction(restored, image, "novel.png"), image
    )


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


def test_ulfloc_mask_protocol_keeps_its_legacy_sky_ordering():
    image = torch.tensor([[[0.2, 0.8]]]).repeat(3, 1, 1)
    target = torch.tensor([[[0.4, 0.1]]]).repeat(3, 1, 1)
    object_mask = torch.tensor([[True, True]])
    sky_mask = torch.tensor([[True, False]])
    distortion_mask = torch.tensor([[True, False]])

    masked_image, masked_target, keep = ulfloc_masked_supervision(
        image,
        target,
        object_mask=object_mask,
        sky_mask=sky_mask,
        distortion_mask=distortion_mask,
    )

    assert torch.equal(keep, torch.tensor([[True, False]]))
    assert torch.allclose(masked_image[:, 0, 1], torch.zeros(3))
    # ULF-Loc overwrites the sky target *after* object/distortion masking.
    assert torch.allclose(masked_target[:, 0, 1], torch.ones(3))


def test_ulfloc_mask_protocol_matches_the_released_tensor_operation_order():
    """Keep the control objective byte-for-byte equivalent to ULF's masking.

    This intentionally spells out the three released ULF-Loc tensor operations
    instead of reusing a helper: it catches a future "clean-up" that changes
    the unusual post-mask sky assignment into an ordinary semantic AND mask.
    """
    image = torch.tensor(
        [
            [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]],
            [[0.70, 0.80, 0.90], [0.15, 0.25, 0.35]],
            [[0.45, 0.55, 0.65], [0.75, 0.85, 0.95]],
        ]
    )
    target = 1.0 - image
    object_mask = torch.tensor([[True, False, True], [True, True, False]])
    sky_mask = torch.tensor([[True, False, False], [True, False, True]])
    distortion_mask = torch.tensor([[True, True, False], [True, False, True]])

    # Literal release-main ULF-Loc semantics from train.py:
    # mask = obj_mask & distort_mask; image *= mask; gt *= mask;
    # gt[sky_mask.repeat(3, 1, 1) == False] = 1.
    expected_keep = object_mask & distortion_mask
    expected_image = image * expected_keep[None]
    expected_target = target * expected_keep[None]
    expected_target[~sky_mask[None].repeat(3, 1, 1)] = 1

    actual_image, actual_target, actual_keep = ulfloc_masked_supervision(
        image,
        target,
        object_mask=object_mask,
        sky_mask=sky_mask,
        distortion_mask=distortion_mask,
    )

    assert torch.equal(actual_keep, expected_keep)
    assert torch.equal(actual_image, expected_image)
    assert torch.equal(actual_target, expected_target)


def test_masked_depth_order_requires_both_members_of_each_random_pair(monkeypatch):
    """A valid facade pixel must not be ordered against a masked sky/hole pixel."""

    # Map every 2x2 source pixel to bottom-right.  That target is masked out,
    # while the raw depth/prior ordering would otherwise contribute a loss.
    shifts = torch.tensor(
        [[[1, 1], [1, 0]], [[0, 1], [0, 0]]], dtype=torch.long
    )

    def fixed_shifts(low, high, size, *, device=None, **_kwargs):
        assert tuple(size) == (4, 2)
        return shifts.reshape(4, 2).to(device=device)

    monkeypatch.setattr(torch, "randint", fixed_shifts)
    depth = torch.tensor([[[2.0, 2.0], [2.0, 1.0]]])
    prior = torch.tensor([[[1.0, 1.0], [1.0, 2.0]]])
    keep = torch.tensor([[True, True], [True, False]])

    raw = compute_masked_depth_order_loss(
        depth=depth,
        prior_depth=prior,
        scene_extent=1.0,
        max_pixel_shift_ratio=1.0,
        normalize_loss=False,
        log_space=False,
    )
    masked = compute_masked_depth_order_loss(
        depth=depth,
        prior_depth=prior,
        mask=keep,
        scene_extent=1.0,
        max_pixel_shift_ratio=1.0,
        normalize_loss=False,
        log_space=False,
    )

    assert raw > 0
    assert torch.equal(masked, torch.tensor(0.0))


def test_geometry_schedule_uses_chart_views_early_then_dense_only():
    assert geometry_iteration(5, use_dense_supervision=True, every_n=5, dense_only_from_iter=3000)
    assert not geometry_iteration(6, use_dense_supervision=True, every_n=5, dense_only_from_iter=3000)
    assert not geometry_iteration(3005, use_dense_supervision=True, every_n=5, dense_only_from_iter=3000)
    assert geometry_iteration(3005, use_dense_supervision=False, every_n=5, dense_only_from_iter=3000)


def test_persistent_geometry_schedule_never_hard_disables_chart_supervision():
    assert geometry_iteration(
        5,
        use_dense_supervision=True,
        every_n=5,
        dense_only_from_iter=3000,
        geometry_schedule="persistent",
    )
    assert geometry_iteration(
        12_010,
        use_dense_supervision=True,
        every_n=5,
        dense_only_from_iter=3000,
        geometry_schedule="persistent",
    )
    assert geometry_iteration(
        25_020,
        use_dense_supervision=True,
        every_n=5,
        dense_only_from_iter=3000,
        geometry_schedule="persistent",
    )
    assert geometry_prior_schedule_weight(
        40_000, geometry_schedule="persistent", final_weight_floor=0.2
    ) == 0.2


def test_dense_only_densification_keeps_chart_geometry_but_not_chart_topology_stats():
    # The legacy path remains byte-for-byte semantically identical: every
    # rendered view contributes radii/gradient statistics.
    assert densification_stats_from_view(
        policy="legacy_current",
        use_dense_supervision=True,
        is_geometry_view=True,
    )
    assert densification_stats_from_view(
        policy="legacy_current",
        use_dense_supervision=True,
        is_geometry_view=False,
    )

    # With all-real dense supervision, Chart iterations still compute their
    # geometry loss but cannot dominate split/prune allocation.
    assert not densification_stats_from_view(
        policy="dense_only",
        use_dense_supervision=True,
        is_geometry_view=True,
    )
    assert densification_stats_from_view(
        policy="dense_only",
        use_dense_supervision=True,
        is_geometry_view=False,
    )

    # A chart-only control has no alternate real-view stream, so dense_only is
    # intentionally a no-op rather than silently disabling densification.
    assert densification_stats_from_view(
        policy="dense_only",
        use_dense_supervision=False,
        is_geometry_view=True,
    )


def test_opacity_reset_schedule_can_hold_resets_fixed_across_topology_horizons():
    native_short = [
        iteration
        for iteration in range(1, 7001)
        if opacity_reset_due(
            iteration,
            opacity_reset_interval=3000,
            densify_from_iter=500,
            densify_until_iter=3500,
            white_background=True,
            continue_after_densify=False,
        )
    ]
    continued_short = [
        iteration
        for iteration in range(1, 7001)
        if opacity_reset_due(
            iteration,
            opacity_reset_interval=3000,
            densify_from_iter=500,
            densify_until_iter=3500,
            white_background=True,
            continue_after_densify=True,
        )
    ]
    native_long = [
        iteration
        for iteration in range(1, 7001)
        if opacity_reset_due(
            iteration,
            opacity_reset_interval=3000,
            densify_from_iter=500,
            densify_until_iter=15000,
            white_background=True,
            continue_after_densify=False,
        )
    ]

    assert native_short == [500, 3000]
    assert continued_short == native_long == [500, 3000, 6000]


def test_chart_geometry_prior_weight_zero_removes_only_the_extra_chart_residuals():
    losses = (torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0))
    assert scale_chart_geometry_priors(losses, weight=1.0) == losses
    assert all(
        torch.equal(value, torch.tensor(0.0))
        for value in scale_chart_geometry_priors(losses, weight=0.0)
    )
    try:
        scale_chart_geometry_priors(losses, weight=-0.1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("Negative Chart geometry-prior weight must fail")


def test_dense_final_only_is_not_overridden_for_the_initial_screen():
    assert not preliminary_uses_dense_supervision(
        dense_supervision=True,
        dense_final_only=True,
    )
    assert preliminary_uses_dense_supervision(
        dense_supervision=True,
        dense_final_only=False,
    )
    assert not preliminary_uses_dense_supervision(
        dense_supervision=False,
        dense_final_only=False,
    )


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


def test_all_train_importance_corrects_chart_rgb_oversampling_without_changing_geometry_schedule():
    chart_weight, non_chart_weight = rgb_sampling_importance_weights(
        policy="all_train_importance",
        total_iterations=10,
        dense_view_count=100,
        chart_view_count=10,
        use_dense_supervision=True,
        geometry_view_every_n_iter=5,
        dense_only_from_iter=10,
    )

    # Two of ten iterations use a uniformly sampled Chart camera.  Charts are
    # still rendered more often for geometry, but their weighted RGB mass is
    # exactly the same as every other real dense camera.
    chart_probability = 0.2 / 10 + 0.8 / 100
    non_chart_probability = 0.8 / 100
    assert chart_weight < 1.0 < non_chart_weight
    assert abs(chart_probability * chart_weight - 1.0 / 100) < 1e-9
    assert abs(non_chart_probability * non_chart_weight - 1.0 / 100) < 1e-9


def test_legacy_interleaved_keeps_unmodified_rgb_weights():
    assert rgb_sampling_importance_weights(
        policy="legacy_interleaved",
        total_iterations=10,
        dense_view_count=100,
        chart_view_count=10,
        use_dense_supervision=True,
        geometry_view_every_n_iter=5,
        dense_only_from_iter=10,
    ) == (1.0, 1.0)
    assert rgb_supervision_weight(
        is_pseudo_view=True,
        is_geometry_view=True,
        use_dense_supervision=False,
        downweight_input_view_color_loss=False,
        pseudo_rgb_weight=0.01,
    ) == 0.01


def test_block_balanced_rgb_importance_uses_each_camera_probability():
    names = ["chart.png", "small_block.png", "large_a.png", "large_b.png"]
    weights = rgb_sampling_importance_weights_by_camera(
        policy="all_train_importance",
        total_iterations=10,
        dense_camera_names=names,
        chart_camera_names={"chart.png"},
        use_dense_supervision=True,
        geometry_view_every_n_iter=5,
        dense_only_from_iter=10,
        dense_view_sampling_policy="spatial_block_balanced",
        dense_view_blocks=[[0, 1], [2, 3]],
    )

    # Both occupied blocks have two cameras here, so non-Chart RGB mass is
    # equal.  The Chart's extra geometry samples are corrected separately.
    assert weights["chart.png"] < weights["small_block.png"]
    assert weights["small_block.png"] == weights["large_a.png"]
    dense_fraction = 0.8
    chart_probability = 0.2 + dense_fraction * 0.25
    assert abs(chart_probability * weights["chart.png"] - 0.25) < 1e-9


def test_checkpoint_continuation_reconditions_rgb_sampler_to_remaining_schedule():
    names = ["chart.png", "other.png"]
    fresh = rgb_sampling_importance_weights_by_camera(
        policy="all_train_importance",
        total_iterations=40_000,
        dense_camera_names=names,
        chart_camera_names={"chart.png"},
        use_dense_supervision=True,
        geometry_view_every_n_iter=5,
        dense_only_from_iter=3000,
        dense_view_sampling_policy="uniform",
        geometry_schedule="persistent",
    )
    continued = rgb_sampling_importance_weights_by_camera(
        policy="all_train_importance",
        total_iterations=40_000,
        start_iteration=7_000,
        dense_camera_names=names,
        chart_camera_names={"chart.png"},
        use_dense_supervision=True,
        geometry_view_every_n_iter=5,
        dense_only_from_iter=3000,
        dense_view_sampling_policy="uniform",
        geometry_schedule="persistent",
    )

    # 7k->40k starts in the high-frequency first phase and later crosses two
    # cadence changes, so using a fresh 0->40k mixture would be wrong.
    assert continued["chart.png"] != fresh["chart.png"]
    geometry_count = sum(
        geometry_iteration(
            iteration,
            use_dense_supervision=True,
            every_n=5,
            dense_only_from_iter=3000,
            geometry_schedule="persistent",
        )
        for iteration in range(7_001, 40_001)
    )
    geometry_fraction = geometry_count / 33_000
    chart_probability = geometry_fraction + (1.0 - geometry_fraction) * 0.5
    other_probability = (1.0 - geometry_fraction) * 0.5
    assert abs(chart_probability * continued["chart.png"] - 0.5) < 1e-9
    assert abs(other_probability * continued["other.png"] - 0.5) < 1e-9


def test_active_chart_scheduler_intersects_gate_and_quality_flags():
    active = resolve_active_chart_indices(
        5,
        alignment_gate_valid=torch.tensor([True, False, True, True, True]),
        quality_selection_active=torch.tensor([True, True, False, True, True]),
    )

    assert active == [0, 3, 4]
    assert geometry_chart_sampling_indices(
        5,
        active_chart_indices=active,
        policy="active_only",
    ) == [0, 3, 4]
    assert geometry_chart_sampling_indices(
        5,
        active_chart_indices=active,
        policy="legacy_all_input",
    ) == [0, 1, 2, 3, 4]


def test_active_chart_scheduler_rejects_stale_flag_cardinality():
    try:
        resolve_active_chart_indices(
            3,
            alignment_gate_valid=torch.tensor([True, False]),
        )
    except RuntimeError as error:
        assert "expected 3 Charts" in str(error)
    else:
        raise AssertionError("Expected stale gate cardinality to be rejected")


def test_chart_camera_order_requires_exact_tensor_to_camera_mapping(tmp_path: Path):
    (tmp_path / "cameras.json").write_text(json.dumps({
        "filepaths": ["/tmp/a.png", "/tmp/b.png"],
    }))

    assert validate_chart_camera_order(
        tmp_path,
        ["a", "b"],
        chart_tensor_count=2,
    ) == ["a", "b"]

    try:
        validate_chart_camera_order(
            tmp_path,
            ["b", "a"],
            chart_tensor_count=2,
        )
    except RuntimeError as error:
        assert "order mismatch" in str(error)
        assert "index 0" in str(error)
    else:
        raise AssertionError("Expected chart/camera ordering mismatch to be rejected")


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


def test_fused_geometry_validity_uses_verified_sources_not_confidence_scale():
    # Bits: plane=1, aligned Chart=2, calibrated mono=4. Mono-only depth is
    # useful as weak supervision but must not seed geometry without a static
    # multi-view anchor.
    source = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.uint8)

    assert torch.equal(
        fused_geometry_validity_mask(source),
        torch.tensor([[False, True, True, True], [False, True, True, True]]),
    )


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

    uncapped, uncapped_valid = sanitize_chart_geometry(
        charts,
        max_abs_depth=0.0,
        max_abs_point=0.0,
    )
    assert torch.equal(uncapped_valid, torch.tensor([[[True, False], [True, False]]]))
    assert uncapped["depths"][0, 1, 0] == 60.0


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
