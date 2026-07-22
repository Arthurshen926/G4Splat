import torch

from matcha.dm_modules.matcher_3d import Matcher3D


class _IdentityTransform:
    def transform_points(self, points):
        return points


class _IdentityP3DCameras:
    def get_world_to_view_transform(self):
        return _IdentityTransform()


class _IdentityCameras:
    """Small differentiable camera contract for matcher-layout regression tests."""

    p3d_cameras = _IdentityP3DCameras()

    def transform_points_world_to_view(self, points):
        return points

    def project_points(self, points, **_kwargs):
        # (0, 0) is a valid normalized-image coordinate for grid_sample.
        return torch.zeros_like(points[..., :2])


def _points(charts=1, height=4, width=4):
    values = torch.arange(charts * height * width, dtype=torch.float32).reshape(
        charts, height, width
    )
    return torch.stack((torch.zeros_like(values), torch.zeros_like(values), values + 1.0), dim=-1)


def test_source_stride_is_exactly_the_full_matching_layout_subsample():
    points = _points()
    matcher = Matcher3D(
        _IdentityCameras(),
        reference_pts=points,
        reference_depths=torch.ones((1, 4, 4)),
    )

    full_error, full_fov = matcher.compute_reprojection_errors(points=points, source_stride=1)
    strided_error, strided_fov = matcher.compute_reprojection_errors(points=points, source_stride=2)

    assert torch.equal(strided_fov, full_fov[..., ::2, ::2])
    assert torch.equal(strided_error, full_error[..., ::2, ::2])


def test_reference_mask_filters_the_source_chart_axis_not_the_target_axis():
    points = _points(charts=2)
    source_mask = torch.ones((2, 4, 4), dtype=torch.bool)
    source_mask[0, 1, 2] = False
    source_mask[1, 3, 0] = False
    matcher = Matcher3D(
        _IdentityCameras(),
        reference_pts=points,
        reference_depths=torch.ones((2, 4, 4)),
        reference_masks=source_mask,
    )

    matcher.match(matching_thr=1e6)

    assert matcher.reference_matches.shape == (2, 2, 4, 4)
    for source_chart in range(2):
        assert torch.equal(
            matcher.reference_matches[:, source_chart],
            source_mask[source_chart].expand(2, -1, -1),
        )


def test_checkpointed_matching_loss_matches_materialized_objective_and_gradient():
    source_mask = torch.ones((2, 4, 4), dtype=torch.bool)
    source_mask[0, 1, 2] = False
    source_mask[1, 3, 0] = False
    reference_depths = torch.ones((2, 4, 4))
    matcher = Matcher3D(
        _IdentityCameras(),
        reference_pts=_points(charts=2),
        reference_depths=reference_depths,
        reference_masks=source_mask,
        projection_chunk_size=3,
    )
    matcher.match(matching_thr=1e6, source_stride=2)

    points = _points(charts=2).requires_grad_()
    target_depths = torch.ones((2, 4, 4), requires_grad=True)
    errors, fov = matcher.compute_reprojection_errors(
        points=points,
        depths=target_depths,
        source_stride=2,
    )
    valid = fov & matcher.reference_matches
    source_valid = source_mask[:, ::2, ::2][None].expand_as(errors)
    dense_loss = torch.where(valid, errors, torch.zeros_like(errors)).sum() / (
        source_valid.sum().to(errors.dtype)
    )
    dense_loss.backward()
    dense_points_grad = points.grad.detach().clone()
    dense_depths_grad = target_depths.grad

    points.grad = None
    target_depths.grad = None
    chunked_loss = matcher.compute_matching_loss(
        points=points,
        depths=target_depths,
        source_stride=2,
        source_masks=source_mask,
        checkpoint_chunks=True,
    )
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, dense_loss.detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(points.grad, dense_points_grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(target_depths.grad, dense_depths_grad, atol=1e-7, rtol=1e-6)
