import torch

from matcha.see3d_geometry import (
    build_render_support_mask,
    robust_align_inverse_depth,
)


def test_render_support_requires_visibility_alpha_and_valid_depth():
    alpha = torch.ones((7, 7))
    visibility = torch.ones((7, 7))
    depth = torch.ones((7, 7))
    visibility[3, 3] = 0
    alpha[1, 1] = 0.2
    depth[5, 5] = 0

    support = build_render_support_mask(
        alpha,
        visibility,
        depth,
        erosion_radius=0,
    )

    assert support.sum().item() == 46
    assert not support[3, 3]
    assert not support[1, 1]
    assert not support[5, 5]


def test_robust_inverse_depth_alignment_rejects_outliers():
    generator = torch.Generator().manual_seed(7)
    disparity = torch.rand((64, 64), generator=generator) * 0.8 + 0.1
    true_inverse_depth = 0.2 + 1.7 * disparity
    render_depth = true_inverse_depth.reciprocal()
    support = torch.ones_like(disparity, dtype=torch.bool)
    render_depth.flatten()[::11] *= 4.0

    aligned, diagnostics = robust_align_inverse_depth(
        disparity,
        render_depth,
        support,
        min_samples=128,
        max_relative_rmse=0.05,
    )

    assert diagnostics.accepted
    assert diagnostics.inlier_ratio < 1.0
    assert abs(diagnostics.alpha - 0.2) < 1e-3
    assert abs(diagnostics.beta - 1.7) < 1e-3
    expected = true_inverse_depth.reciprocal()
    assert torch.mean(torch.abs(aligned - expected)).item() < 1e-3


def test_robust_inverse_depth_alignment_rejects_sparse_support():
    disparity = torch.ones((16, 16))
    depth = torch.ones((16, 16))
    support = torch.zeros((16, 16), dtype=torch.bool)
    support[:2, :2] = True

    aligned, diagnostics = robust_align_inverse_depth(
        disparity,
        depth,
        support,
        min_samples=16,
    )

    assert not diagnostics.accepted
    assert "insufficient support" in diagnostics.reason
    assert torch.count_nonzero(aligned) == 0
